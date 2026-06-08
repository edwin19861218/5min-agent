"""
带记忆的 ReAct Agent
在基础版上增加对话历史记忆，支持多轮对话

使用方式：
  python demo_react_with_memory.py --provider deepseek

环境变量：
  export DEEPSEEK_API_KEY="sk-你的key"
  export ZHIPU_API_KEY="你的key"
"""

import os
import re
import json
import argparse
from datetime import datetime
from openai import OpenAI


# ============================================================
# 工具（同基础版）
# ============================================================

def tool_search(query: str) -> str:
    knowledge = {
        "春节": "2024年春节是2024年2月10日（农历正月初一）。",
        "2024年春节": "2024年春节是2024年2月10日（农历正月初一）。",
        "北京面积": "北京市总面积约16410平方公里。",
        "地球到月球": "地球到月球的平均距离约384400公里。",
        "Python发布": "Python最初由Guido van Rossum于1991年发布。",
    }
    for key, value in knowledge.items():
        if key in query:
            return value
    return f"搜索 '{query}' 未找到相关信息。"


def tool_calculate(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return f"不安全的表达式: {expression}"
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


def tool_weekday(date_str: str) -> str:
    try:
        date_str = date_str.strip()
        for fmt in ["%Y-%m-%d", "%Y年%m月%d日"]:
            try:
                dt = datetime.strptime(date_str, fmt)
                weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
                return f"{date_str} 是 {weekdays[dt.weekday()]}"
            except ValueError:
                continue
        return f"无法解析日期: {date_str}"
    except Exception as e:
        return f"日期查询错误: {e}"


TOOLS = {
    "search": {"func": tool_search, "description": "搜索信息，查询事实性问题。", "param_desc": "搜索关键词"},
    "calculate": {"func": tool_calculate, "description": "执行数学计算。", "param_desc": "数学表达式"},
    "weekday": {"func": tool_weekday, "description": "查询某天是星期几。", "param_desc": "日期字符串"},
}


# ============================================================
# 带记忆的 ReAct Agent
# ============================================================

REACT_SYSTEM_PROMPT = """你是一个智能Agent，通过 Thought -> Action -> Observation 的循环来回答问题。

你可以使用以下工具：
{tools_description}

请严格按照以下格式回复：

Thought: 你当前的想法和分析
Action: 要调用的工具名称
Action Input: 工具的输入参数

当你有了最终答案时，使用：

Thought: 我已经有了足够的信息
Final Answer: 你的最终答案

重要规则：
1. 每次只调用一个工具
2. 仔细思考后再决定使用什么工具
3. 得到足够信息后立即给出 Final Answer
"""


class ReActAgentWithMemory:
    """带对话记忆的 ReAct Agent"""

    MAX_HISTORY = 10  # 保留最近10轮对话

    def __init__(self, provider: str = "deepseek", max_steps: int = 6):
        self.max_steps = max_steps
        self.provider = provider

        if provider == "zhipu":
            self.client = OpenAI(
                api_key=os.environ.get("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4/"
            )
            self.model = "glm-5.1"
        else:
            self.client = OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com"
            )
            self.model = "deepseek-chat"

        tools_desc_parts = []
        for name, info in TOOLS.items():
            tools_desc_parts.append(f"- {name}: {info['description']} (参数: {info['param_desc']})")

        self.system_prompt = REACT_SYSTEM_PROMPT.format(
            tools_description="\n".join(tools_desc_parts)
        )
        # 对话记忆：保存所有历史（系统prompt + 对话）
        self.conversation_history = []

    def _trim_history(self):
        """滑动窗口：只保留最近MAX_HISTORY轮"""
        if len(self.conversation_history) > self.MAX_HISTORY * 2:
            self.conversation_history = self.conversation_history[-(self.MAX_HISTORY * 2):]

    def _parse_response(self, text: str) -> dict:
        result = {"thought": "", "action": None, "action_input": None, "final_answer": None}
        fa_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
        if fa_match:
            result["final_answer"] = fa_match.group(1).strip()
        thought_match = re.search(r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer)|$)", text, re.DOTALL)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()
        action_match = re.search(r"Action:\s*(.+)", text)
        if action_match:
            result["action"] = action_match.group(1).strip()
        input_match = re.search(r"Action Input:\s*(.+?)(?=\n\n|\Z)", text, re.DOTALL)
        if input_match:
            result["action_input"] = input_match.group(1).strip()
        return result

    def _execute_tool(self, action: str, action_input: str) -> str:
        if action not in TOOLS:
            return f"错误：未知工具 '{action}'"
        try:
            return TOOLS[action]["func"](action_input)
        except Exception as e:
            return f"工具执行错误: {e}"

    def run(self, question: str) -> str:
        """处理单个问题，保留对话记忆"""
        # 构建当前对话的messages
        messages = [{"role": "system", "content": self.system_prompt}]

        # 加入历史记忆
        for msg in self.conversation_history:
            messages.append(msg)

        # 加入当前问题
        messages.append({"role": "user", "content": question})

        print(f"\n{'='*60}")
        print(f"Agent 收到问题: {question}")
        print(f"历史记忆: {len(self.conversation_history)} 条消息")
        print(f"{'='*60}\n")

        for step in range(1, self.max_steps + 1):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
            )
            text = response.choices[0].message.content
            parsed = self._parse_response(text)

            print(f"--- Step {step} ---")
            if parsed["thought"]:
                print(f"  💭 Thought: {parsed['thought']}")

            if parsed["final_answer"]:
                print(f"  ✅ Final Answer: {parsed['final_answer']}")
                # 保存到记忆
                self.conversation_history.append({"role": "user", "content": question})
                self.conversation_history.append({"role": "assistant", "content": parsed["final_answer"]})
                self._trim_history()
                print(f"  📝 记忆已保存 ({len(self.conversation_history)} 条)")
                return parsed["final_answer"]

            if parsed["action"]:
                print(f"  🔧 Action: {parsed['action']}({parsed['action_input']})")
                observation = self._execute_tool(parsed["action"], parsed["action_input"])
                print(f"  👁️ Observation: {observation}")

                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": f"Observation: {observation}\n\n请继续思考。"})
            else:
                self.conversation_history.append({"role": "user", "content": question})
                self.conversation_history.append({"role": "assistant", "content": text})
                return text

        print(f"\n⚠️ 达到最大步数 {self.max_steps}")
        return "抱歉，达到最大推理步数限制。"


# ============================================================
# 交互式多轮对话
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="带记忆的ReAct Agent（交互式）")
    parser.add_argument("--provider", choices=["deepseek", "zhipu"], default="deepseek")
    parser.add_argument("--max-steps", type=int, default=6)
    args = parser.parse_args()

    agent = ReActAgentWithMemory(provider=args.provider, max_steps=args.max_steps)

    print("🤖 带记忆的 ReAct Agent（输入 'quit' 退出）")
    print(f"   模型: {agent.model} | 最大步数: {args.max_steps}")
    print()

    while True:
        try:
            question = input("🙋 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        answer = agent.run(question)
        print(f"\n🤖 Agent: {answer}\n")


if __name__ == "__main__":
    main()
