#!/usr/bin/env python3
"""
LangChain Agent Demo -- 搜索+计算Agent
「从0-1成为Agent架构师」系列 · 第5篇配套代码

用法:
    pip install langchain langchain-openai langchain-core langgraph

    # 使用DeepSeek
    python demo_langchain_agent.py --provider deepseek

    # 使用智谱GLM
    python demo_langchain_agent.py --provider zhipu

环境变量:
    DEEPSEEK_API_KEY  -- DeepSeek API密钥
    ZHIPU_API_KEY     -- 智谱GLM API密钥
"""

import argparse
import os
import sys

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


# ============================================================
# 模型配置
# ============================================================

PROVIDERS = {
    "deepseek": {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "zhipu": {
        "model": "glm-5.1",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "api_key_env": "ZHIPU_API_KEY",
    },
}


# ============================================================
# 工具定义
#
# LangChain标准做法：工具的description负责告诉LLM
# "我是谁、什么时候该用我"。LLM根据description自动
# 决定调用哪个工具，不需要在system prompt里重复说明。
# ============================================================

@tool
def search(query: str) -> str:
    """当你需要查找概念解释、技术名词含义、背景知识等信息时使用此工具。
    输入一个搜索关键词或问题，返回相关的信息摘要。

    适用场景举例：
    - "LangChain是什么"
    - "ReAct模式的工作原理"
    - "Python的发布年份"
    """
    knowledge = {
        "python": "Python是一种通用编程语言，创建于1991年，广泛用于AI、数据科学和Web开发",
        "agent": "AI Agent是能够自主感知环境、做出决策并执行动作的智能体系统",
        "langchain": "LangChain是一个开源框架，用于构建基于大语言模型的应用，支持链式调用、工具集成和记忆管理",
        "deepseek": "DeepSeek是一家中国AI公司，提供DeepSeek-V3等高性能大语言模型，API兼容OpenAI格式",
        "glm": "智谱GLM是智谱AI开发的大语言模型系列，GLM-5.1是其最新版本",
        "react": "ReAct模式是Agent的核心推理范式：Thought -> Action -> Observation循环",
    }
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    return f"未找到关于'{query}'的直接结果。可尝试更具体的关键词。"


@tool
def calculate(expression: str) -> str:
    """当你需要进行数学计算时使用此工具。
    输入一个数学表达式字符串，返回计算结果。

    支持的运算：加减乘除、幂运算(**)、括号
    示例输入："2**10"、"(3+5)*7"、"2025 - 1991"
    """
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return f"不安全的表达式: '{expression}'，仅支持数字和基本运算符"
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return "错误：除数不能为零"
    except Exception as e:
        return f"计算错误: {e}"


# ============================================================
# 创建Agent
# ============================================================

def create_agent(provider: str):
    """根据provider创建ReAct Agent"""
    if provider not in PROVIDERS:
        print(f"不支持的provider: {provider}，可选: {list(PROVIDERS.keys())}")
        sys.exit(1)

    config = PROVIDERS[provider]
    api_key = os.getenv(config["api_key_env"])

    if not api_key:
        print(f"请设置环境变量: export {config['api_key_env']}=your_api_key")
        sys.exit(1)

    # 1. 创建LLM（通过openai SDK兼容国产模型）
    llm = ChatOpenAI(
        model=config["model"],
        base_url=config["base_url"],
        api_key=api_key,
        temperature=0,
    )

    # 2. 注册工具
    tools = [search, calculate]

    # 3. 创建ReAct Agent
    #
    # 注意：prompt只定义Agent的角色和行为准则，
    # 工具选择由LLM根据tool的description自动完成。
    # 不需要在这里写"用search查信息，用calculate做计算"，
    # 因为每个tool的docstring已经清楚说明了适用场景。
    system_prompt = (
        "你是一个智能助手。"
        "根据用户的问题，选择合适的工具来完成任务。"
        "可以连续调用多个工具。"
        "用中文回答。"
    )

    agent = create_react_agent(
        llm,
        tools,
        prompt=system_prompt,
    )

    return agent


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LangChain Agent Demo")
    parser.add_argument(
        "--provider",
        choices=["deepseek", "zhipu"],
        default="deepseek",
        help="选择LLM提供商 (默认: deepseek)",
    )
    args = parser.parse_args()

    print(f"使用 {args.provider} 模型启动Agent...")
    print("-" * 50)

    agent = create_agent(args.provider)

    # 演示1：纯搜索
    print("\n测试1：搜索问题")
    print("-" * 50)
    result = agent.invoke({"messages": [("user", "LangChain是什么？")]})
    last_msg = result["messages"][-1]
    print(f"回答: {last_msg.content}")

    # 演示2：纯计算
    print("\n测试2：数学计算")
    print("-" * 50)
    result = agent.invoke({"messages": [("user", "帮我算一下2的10次方加上3的5次方")]})
    last_msg = result["messages"][-1]
    print(f"回答: {last_msg.content}")

    # 演示3：混合任务（搜索+计算）
    print("\n测试3：混合任务")
    print("-" * 50)
    result = agent.invoke({"messages": [("user", "Python是哪年发布的？帮我算算距离现在多少年了")]})
    last_msg = result["messages"][-1]
    print(f"回答: {last_msg.content}")

    print("\n所有测试完成！")


if __name__ == "__main__":
    main()
