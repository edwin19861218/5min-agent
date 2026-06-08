"""
工具注册表：统一管理 Agent 可调用的工具
=====================================
每天5分钟学Agent开发 · 第4篇 Demo

依赖安装：
    pip install openai

使用方法：
    # 默认使用 DeepSeek
    python tool_registry.py

    # 切换到智谱
    python tool_registry.py --provider zhipu

环境变量：
    DEEPSEEK_API_KEY=sk-xxx
    ZHIPU_API_KEY=xxx
"""

import argparse
import json
import math
import os
import re
from typing import Any, Callable


# ============================================================
# 1. 工具注册表
# ============================================================

class ToolRegistry:
    """工具注册表 —— 统一管理所有可被 Agent 调用的工具"""

    def __init__(self):
        self._tools: dict[str, Callable] = {}          # name → 函数
        self._schemas: dict[str, dict] = {}             # name → JSON Schema

    def register(self, name: str, description: str, parameters: dict):
        """装饰器：注册一个工具"""
        def decorator(func: Callable):
            self._tools[name] = func
            self._schemas[name] = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
            return func
        return decorator

    def get_function(self, name: str) -> Callable:
        if name not in self._tools:
            raise ValueError(f"未知工具: {name}")
        return self._tools[name]

    def get_schema(self, name: str) -> dict:
        return self._schemas[name]

    def all_schemas(self) -> list[dict]:
        """返回所有工具的 JSON Schema（用于传给 LLM）"""
        return list(self._schemas.values())

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())


# ============================================================
# 2. 定义具体工具
# ============================================================

registry = ToolRegistry()


@registry.register(
    name="search",
    description="搜索互联网信息。输入查询关键词，返回相关结果摘要。",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词",
            },
        },
        "required": ["query"],
    },
)
def search(query: str) -> str:
    """模拟搜索工具（实际项目中接入真实搜索 API）"""
    # 这里用模拟数据演示
    mock_db = {
        "北京天气": "北京今日 28°C，晴转多云，空气质量良好，PM2.5: 35",
        "上海天气": "上海今日 31°C，多云转阴，午后有雷阵雨",
        "Python": "Python 是一种通用编程语言，由 Guido van Rossum 于 1991 年发布，"
                  "目前最新稳定版为 3.12，广泛用于 AI/ML、Web 开发、自动化等领域",
        "DeepSeek": "DeepSeek 是深度求索公司推出的大语言模型，"
                    "DeepSeek-V3 拥有 671B 参数，支持 128K 上下文，"
                    "在编程和数学推理方面表现突出",
    }
    # 模糊匹配
    for key, value in mock_db.items():
        if any(word in key for word in query) or any(word in query for word in key.split()):
            return value
    return f"搜索 '{query}'：未找到精确结果。建议尝试更具体的关键词。"


@registry.register(
    name="calculator",
    description="数学计算器。支持四则运算、幂运算、三角函数等数学表达式。",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式，如 '2 + 3 * 4' 或 'sqrt(16)'",
            },
        },
        "required": ["expression"],
    },
)
def calculator(expression: str) -> str:
    """安全的数学计算器"""
    # 白名单安全检查：只允许数字、运算符和数学函数
    allowed = set("0123456789+-*/().,%^ ")
    allowed_funcs = ["sqrt", "sin", "cos", "tan", "log", "log10", "pi", "e", "abs", "round", "pow"]

    sanitized = expression.replace("^", "**")
    for func in allowed_funcs:
        sanitized = sanitized.replace(func, "")

    for char in sanitized:
        if char not in allowed:
            return f"错误：表达式包含不允许的字符 '{char}'"

    try:
        safe_dict = {
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "log10": math.log10,
            "pi": math.pi, "e": math.e, "abs": abs, "round": round,
            "pow": pow,
        }
        result = eval(expression.replace("^", "**"), {"__builtins__": {}}, safe_dict)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


@registry.register(
    name="file_read",
    description="读取本地文件内容。支持 .txt 和 .md 格式。",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径（相对或绝对路径）",
            },
            "max_lines": {
                "type": "integer",
                "description": "最大读取行数，默认50行",
            },
        },
        "required": ["path"],
    },
)
def file_read(path: str, max_lines: int = 50) -> str:
    """读取文件内容"""
    # 安全检查：防止路径穿越
    if ".." in path:
        return "错误：不允许使用 '..' 路径"

    allowed_exts = {".txt", ".md", ".py", ".json", ".csv"}
    ext = os.path.splitext(path)[1].lower()
    if ext not in allowed_exts:
        return f"错误：不支持的文件类型 '{ext}'，仅支持 {allowed_exts}"

    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()[:max_lines]
        content = "".join(lines)
        if len(lines) >= max_lines:
            content += f"\n... (仅显示前 {max_lines} 行)"
        return content if content else "(文件为空)"
    except FileNotFoundError:
        return f"错误：文件不存在 '{path}'"
    except PermissionError:
        return f"错误：无权限读取 '{path}'"


# ============================================================
# 3. LLM 客户端封装（兼容 DeepSeek / 智谱）
# ============================================================

PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "env_key": "DEEPSEEK_API_KEY",
        "model": "deepseek-chat",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "env_key": "ZHIPU_API_KEY",
        "model": "glm-4-flash",
    },
}


def create_client(provider: str):
    """根据 provider 创建 OpenAI 兼容客户端"""
    from openai import OpenAI

    cfg = PROVIDERS[provider]
    api_key = os.getenv(cfg["env_key"])
    if not api_key:
        raise ValueError(f"请设置环境变量 {cfg['env_key']}")

    return OpenAI(base_url=cfg["base_url"], api_key=api_key), cfg["model"]


# ============================================================
# 4. Agent Loop —— Function Calling 核心流程
# ============================================================

def run_agent(user_query: str, provider: str = "deepseek", verbose: bool = True):
    """
    执行一次完整的 Agent 对话：
    1. 把用户问题发给 LLM
    2. LLM 决定是否调用工具（可能并行调用多个）
    3. 执行工具，将结果注入对话
    4. 重复直到 LLM 给出最终回答
    """
    client, model = create_client(provider)

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个智能助手，可以使用工具来帮助用户。"
                "当需要搜索信息时使用 search 工具，"
                "当需要计算时使用 calculator 工具，"
                "当需要读取文件时使用 file_read 工具。"
                "请用中文回答。"
            ),
        },
        {"role": "user", "content": user_query},
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"🤖 Agent 启动 (provider={provider}, model={model})")
        print(f"📌 用户问题: {user_query}")
        print(f"{'='*60}\n")

    round_num = 0
    max_rounds = 10  # 防止无限循环

    while round_num < max_rounds:
        round_num += 1
        if verbose:
            print(f"--- 第 {round_num} 轮 ---")

        # 调用 LLM
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=registry.all_schemas(),
            tool_choice="auto",
        )

        msg = response.choices[0].message

        # 情况1：LLM 返回最终文字回答，不需要调用工具
        if not msg.tool_calls:
            if verbose:
                print(f"\n💬 最终回答:\n{msg.content}\n")
            return msg.content

        # 情况2：LLM 请求调用工具（可能是多个并行调用）
        messages.append(msg)  # 把 assistant 的 tool_calls 消息加入历史

        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            if verbose:
                print(f"  🔧 调用工具: {tool_name}({tool_args})")

            # 执行工具
            try:
                func = registry.get_function(tool_name)
                result = func(**tool_args)
            except Exception as e:
                result = f"工具执行出错: {e}"
                if verbose:
                    print(f"  ⚠️ {result}")

            if verbose:
                preview = result[:200] + "..." if len(result) > 200 else result
                print(f"  📤 返回: {preview}\n")

            # 将工具结果注入对话
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })

    return "（达到最大轮次限制，Agent 停止）"


# ============================================================
# 5. 并行调用 + 工具组合链示例
# ============================================================

def demo_parallel_calls(provider: str = "deepseek"):
    """演示并行工具调用：LLM 一次请求调用多个工具"""
    print("\n" + "=" * 60)
    print("📋 场景：并行调用 —— 同时查两个城市的天气并计算温差")
    print("=" * 60)

    run_agent(
        "帮我同时查一下北京和上海的天气，然后计算两个城市的温差是多少。",
        provider=provider,
    )


def demo_tool_chain(provider: str = "deepseek"):
    """演示工具组合链：A 的输出作为 B 的输入"""
    print("\n" + "=" * 60)
    print("📋 场景：工具组合链 —— 搜索信息 → 计算处理")
    print("=" * 60)

    run_agent(
        "搜索 Python 的发布年份，然后计算距今已经多少年了。",
        provider=provider,
    )


def demo_error_handling(provider: str = "deepseek"):
    """演示错误处理：工具调用失败时的优雅降级"""
    print("\n" + "=" * 60)
    print("📋 场景：错误处理 —— 读取不存在的文件")
    print("=" * 60)

    run_agent(
        "帮我读取 /nonexistent/secret.txt 文件的内容，如果失败请告诉我原因。",
        provider=provider,
    )


# ============================================================
# 6. 手动演示（不需要 LLM，纯本地运行）
# ============================================================

def demo_manual():
    """手动演示工具注册表和调用，不需要 API Key"""
    print("=" * 60)
    print("🔧 工具注册表演示（本地模式，无需 API Key）")
    print("=" * 60)

    # 列出所有已注册的工具
    print(f"\n已注册工具: {registry.list_tools()}")

    # 打印工具 Schema
    print("\n📋 工具 JSON Schema:")
    for schema in registry.all_schemas():
        func = schema["function"]
        print(f"\n  工具: {func['name']}")
        print(f"  描述: {func['description']}")
        print(f"  参数: {json.dumps(func['parameters'], ensure_ascii=False, indent=4)}")

    # 直接调用工具
    print("\n" + "-" * 40)
    print("🔧 直接调用 search 工具:")
    result = registry.get_function("search")("北京天气")
    print(f"  search('北京天气') → {result}")

    print("\n🔧 直接调用 calculator 工具:")
    result = registry.get_function("calculator")("sqrt(144) + 3^2")
    print(f"  calculator('sqrt(144) + 3^2') → {result}")

    print("\n🔧 并行调用多个工具:")
    results = {}
    results["weather"] = registry.get_function("search")("上海天气")
    results["math"] = registry.get_function("calculator")("28 - 31")
    for tool, res in results.items():
        print(f"  [{tool}] → {res}")

    print("\n🔧 错误处理演示:")
    result = registry.get_function("file_read")("/nonexistent/file.txt")
    print(f"  file_read('/nonexistent/file.txt') → {result}")

    result = registry.get_function("calculator")("import os")
    print(f"  calculator('import os') → {result}")

    print("\n✅ 本地演示完成！")


# ============================================================
# 7. 主入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool Use & Function Calling Demo")
    parser.add_argument(
        "--provider",
        choices=["deepseek", "zhipu"],
        default="deepseek",
        help="LLM 提供商（默认 deepseek）",
    )
    parser.add_argument(
        "--demo",
        choices=["parallel", "chain", "error", "manual", "all"],
        default="manual",
        help="运行哪个演示（默认 manual，不需要 API Key）",
    )
    args = parser.parse_args()

    if args.demo == "manual":
        demo_manual()
    elif args.demo == "parallel":
        demo_parallel_calls(args.provider)
    elif args.demo == "chain":
        demo_tool_chain(args.provider)
    elif args.demo == "error":
        demo_error_handling(args.provider)
    elif args.demo == "all":
        demo_manual()
        demo_parallel_calls(args.provider)
        demo_tool_chain(args.provider)
        demo_error_handling(args.provider)
