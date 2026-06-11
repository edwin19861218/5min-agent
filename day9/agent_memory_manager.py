"""agent_memory_manager.py — Agent记忆管理器
支持记忆写入（带重要性评分）、检索（关键词+时间衰减）、压缩（滑动窗口/摘要/混合）
用法：python agent_memory_manager.py --provider deepseek
"""
import json
import argparse
import os
from datetime import datetime
from pathlib import Path
from openai import OpenAI

PROVIDERS = {
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "zhipu": {
        "api_key_env": "ZHIPU_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "model": "glm-5.1",
    },
}


class MemoryManager:
    """简易记忆管理器 — JSON文件存储"""

    def __init__(self, filepath="memories.json"):
        self.filepath = Path(filepath)
        self.memories = self._load()

    def _load(self):
        if self.filepath.exists():
            return json.loads(self.filepath.read_text(encoding="utf-8"))
        return []

    def _save(self):
        self.filepath.write_text(
            json.dumps(self.memories, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add(self, content, importance=5, tags=None):
        """写入一条记忆（带去重检查）"""
        if self._is_duplicate(content):
            print(f"  [去重] 跳过重复记忆: {content[:30]}...")
            return False
        memory = {
            "id": len(self.memories) + 1,
            "content": content,
            "importance": importance,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "access_count": 0,
        }
        self.memories.append(memory)
        self._save()
        print(f"  [写入] 重要性={importance}: {content[:40]}...")
        return True

    def _is_duplicate(self, new_content, threshold=0.6):
        """基于关键词重叠度的去重"""
        new_words = set(new_content.lower().split())
        if not new_words:
            return False
        for mem in self.memories:
            old_words = set(mem["content"].lower().split())
            if not old_words:
                continue
            overlap = len(new_words & old_words) / len(new_words | old_words)
            if overlap > threshold:
                return True
        return False

    def search(self, query, top_k=3, time_decay_hours=24):
        """检索：关键词匹配 + 时间衰减"""
        now = datetime.now()
        scored = []
        query_words = set(query.lower().split())
        for mem in self.memories:
            mem_words = set(mem["content"].lower().split())
            keyword_score = len(query_words & mem_words) / max(len(query_words), 1)
            hours_ago = max(
                (now - datetime.fromisoformat(mem["created_at"])).total_seconds() / 3600,
                0,
            )
            time_score = 0.5 ** (hours_ago / time_decay_hours)
            importance_score = mem["importance"] / 10
            total = 0.5 * keyword_score + 0.3 * time_score + 0.2 * importance_score
            scored.append((total, mem))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, mem in scored[:top_k]:
            if score > 0:
                mem["access_count"] += 1
                self._save()
                results.append(mem)
        return results


class ConversationCompressor:
    """会话压缩：三种策略对比"""

    def __init__(self, client, model):
        self.client = client
        self.model = model

    def sliding_window(self, messages, keep_recent=4):
        """策略1：滑动窗口 — 只保留最近N轮"""
        system_msg = [m for m in messages if m["role"] == "system"]
        conversation = [m for m in messages if m["role"] != "system"]
        kept = conversation[-keep_recent * 2 :]
        return system_msg + kept

    def summarize_old(self, messages, keep_recent=4):
        """策略2：递归摘要 — 旧对话压缩成摘要"""
        system_msg = [m for m in messages if m["role"] == "system"]
        conversation = [m for m in messages if m["role"] != "system"]
        if len(conversation) <= keep_recent * 2:
            return messages
        old_part = conversation[: -keep_recent * 2]
        recent_part = conversation[-keep_recent * 2 :]
        old_text = "\n".join(f"{m['role']}: {m['content']}" for m in old_part)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "你是摘要助手。将以下对话压缩成一段简洁的摘要，保留关键信息、用户偏好和重要决定。用中文，不超过200字。",
                },
                {"role": "user", "content": old_text},
            ],
            temperature=0.3,
        )
        summary = response.choices[0].message.content
        print(f"  [摘要] 原始{len(old_part)}条 → 压缩为1条摘要")
        return system_msg + [
            {"role": "assistant", "content": f"[历史对话摘要] {summary}"}
        ] + recent_part

    def hybrid(self, messages, keep_recent=4):
        """策略3：混合 — 摘要 + 最近原文"""
        return self.summarize_old(messages, keep_recent)


class TokenBudgetManager:
    """Token预算管理器"""

    def __init__(self, max_per_turn=4000, max_per_session=20000):
        self.max_per_turn = max_per_turn
        self.max_per_session = max_per_session
        self.session_used = 0

    def estimate_tokens(self, text):
        """粗略估算token数"""
        chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    def check_budget(self, messages):
        """检查是否会超预算"""
        total_text = " ".join(m["content"] for m in messages)
        estimated = self.estimate_tokens(total_text)
        if estimated > self.max_per_turn:
            return "compress", estimated
        if self.session_used + estimated > self.max_per_session:
            return "session_over", estimated
        return "ok", estimated

    def record_usage(self, messages, response_text=""):
        """记录本次使用的token"""
        total_text = " ".join(m["content"] for m in messages) + response_text
        self.session_used += self.estimate_tokens(total_text)


def main():
    parser = argparse.ArgumentParser(description="Agent记忆管理器Demo")
    parser.add_argument("--provider", choices=["deepseek", "zhipu"], default="deepseek")
    args = parser.parse_args()

    cfg = PROVIDERS[args.provider]
    client = OpenAI(
        api_key=os.getenv(cfg["api_key_env"]),
        base_url=cfg["base_url"],
    )
    model = cfg["model"]
    print(f"使用模型: {model}\n")

    # 1. 记忆写入Demo
    print("=" * 50)
    print("Demo 1: 记忆写入（带重要性评分和去重）")
    print("=" * 50)
    mm = MemoryManager("memories.json")
    mm.add("用户偏好使用深色主题编辑器", importance=8, tags=["偏好"])
    mm.add("项目使用Python 3.12 + uv管理依赖", importance=7, tags=["技术栈"])
    mm.add("用户偏好使用深色主题，代码用等宽字体", importance=8, tags=["偏好"])
    mm.add("寒暄：你好，今天天气不错", importance=1, tags=["闲聊"])

    # 2. 记忆检索Demo
    print(f"\n{'=' * 50}")
    print("Demo 2: 记忆检索（关键词 + 时间衰减）")
    print("=" * 50)
    results = mm.search("用户编辑器偏好")
    for r in results:
        print(f"  [{r['importance']}/10] {r['content']}")

    # 3. 会话压缩Demo
    print(f"\n{'=' * 50}")
    print("Demo 3: 会话压缩策略对比")
    print("=" * 50)
    conversation = [
        {"role": "system", "content": "你是Python编程助手。"},
        {"role": "user", "content": "我想学Python，从哪里开始？"},
        {"role": "assistant", "content": "建议从Python官方教程开始，先掌握基本语法：变量、循环、函数。"},
        {"role": "user", "content": "什么是变量？"},
        {"role": "assistant", "content": "变量就像一个盒子，可以存放数据。比如 name = 'Alice' 就是把Alice放进了name这个盒子。"},
        {"role": "user", "content": "循环怎么写？"},
        {"role": "assistant", "content": "Python有两种循环：for循环遍历序列，while循环满足条件时执行。"},
        {"role": "user", "content": "函数怎么定义？"},
        {"role": "assistant", "content": "用def关键字定义函数，比如 def greet(name): return f'Hello {name}'"},
        {"role": "user", "content": "给我写一个计算斐波那契数列的函数"},
        {"role": "assistant", "content": "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)"},
        {"role": "user", "content": "这个递归太慢了，有更好的写法吗？"},
        {"role": "assistant", "content": "用迭代法更高效：维护两个变量a, b不断更新，时间复杂度O(n)。"},
    ]

    compressor = ConversationCompressor(client, model)

    result1 = compressor.sliding_window(conversation, keep_recent=2)
    print(f"\n  滑动窗口: {len(conversation)}条 → {len(result1)}条")

    result2 = compressor.summarize_old(conversation, keep_recent=2)
    print(f"  递归摘要: {len(conversation)}条 → {len(result2)}条")
    if len(result2) > 1 and "历史对话摘要" in result2[1]["content"]:
        print(f"  摘要内容: {result2[1]['content'][:100]}...")

    result3 = compressor.hybrid(conversation, keep_recent=2)
    print(f"  混合策略: {len(conversation)}条 → {len(result3)}条")

    # 4. Token预算管理Demo
    print(f"\n{'=' * 50}")
    print("Demo 4: Token预算管理")
    print("=" * 50)
    budget = TokenBudgetManager(max_per_turn=200, max_per_session=1000)
    status, est = budget.check_budget(conversation)
    print(f"  对话预估token: {est}")
    print(f"  单轮上限: {budget.max_per_turn}")
    print(f"  状态: {status}")
    if status == "compress":
        print("  → 需要压缩！触发摘要压缩...")


if __name__ == "__main__":
    main()
