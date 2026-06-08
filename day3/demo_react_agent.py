"""
ReAct Agent 标准Python实现
不依赖任何框架，纯手工实现 Agent Loop

使用方式：
  python demo_react_agent.py "2024年春节是几月几号？那天是周几？"
  python demo_react_agent.py "北京的面积是多少平方公里？如果一个人每秒走1米，走完需要多久？" --provider zhipu

环境变量：
  export DEEPSEEK_API_KEY="sk-你的key"
  export ZHIPU_API_KEY="你的key"
"""

import os
import re
import json
import argparse
from datetime import datetime, timedelta
from openai import OpenAI


# ============================================================
# 1. 工具注册表 —— Agent 的"手"
# ============================================================

def tool_search(query: str) -> str:
    """模拟搜索工具（实际项目中可接入真实搜索API）"""
    knowledge = {
        "春节": "2024年春节是2024年2月10日（农历正月初一）。",
        "2024年春节": "2024年春节是2024年2月10日（农历正月初一）。",
        "北京面积": "北京市总面积约16410平方公里。",
        "北京 面积": "北京市总面积约16410平方公里。",
        "地球到月球": "地球到月球的平均距离约384400公里。",
        "Python发布": "Python最初由Guido van Rossum于1991年发布。",
    }
    for key, value in knowledge.items():
        if key in query:
            return value
    return f"搜索 '{query}' 未找到相关信息。"


def tool_calculate(expression: str) -> str:
    """计算工具 —— 执行数学运算"""
    try:
        # 安全执行简单数学表达式
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return f"不安全的表达式: {expression}"
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


def tool_weekday(date_str: str) -> str:
    """查询某天是星期几"""
    try:
        # 支持 YYYY-MM-DD 格式
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


# 工具注册表：name -> {func, description, param_desc}
TOOLS = {
    "search": {
        "func": tool_search,
        "description": "搜索信息，查询事实性问题。输入你想搜索的关键词。",
        "param_desc": "搜索关键词"
    },
    "calculate": {
        "func": tool_calculate,
        "description": "执行数学计算。输入一个数学表达式，如 '16410 / 1' 或 '384400 / 3600'。",
        "param_desc": "数学表达式"
    },
    "weekday": {
        "func": tool_weekday,
        "description": "查询某个日期是星期几。输入日期字符串，如 '2024-02-10'。",
        "param_desc": "日期字符串"
    }
}


# ============================================================
# 2. ReAct Prompt 模板 —— 约束 LLM 输出格式
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
3. Action 必须是上面列出的工具之一
4. 得到足够信息后立即给出 Final Answer
"""


# ============================================================
# 3. 核心组件：输出解析器 + 工具执行器 + Agent循环
# ============================================================

class ReActAgent:
    """标准 ReAct Agent 实现"""

    def __init__(self, provider: str = "deepseek", max_steps: int = 6):
        self.max_steps = max_steps
        self.provider = provider

        # 根据provider创建客户端
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

        # 构建工具描述
        tools_desc_parts = []
        for name, info in TOOLS.items():
            tools_desc_parts.append(f"- {name}: {info['description']} (参数: {info['param_desc']})")
        self.tools_description = "\n".join(tools_desc_parts)

        # 构建System Prompt
        self.system_prompt = REACT_SYSTEM_PROMPT.format(
            tools_description=self.tools_description
        )

    def _parse_response(self, text: str) -> dict:
        """解析LLM输出，提取Thought/Action/Action Input/Final Answer"""
        result = {"thought": "", "action": None, "action_input": None, "final_answer": None}

        # 检查是否有 Final Answer
        fa_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
        if fa_match:
            result["final_answer"] = fa_match.group(1).strip()

        # 提取 Thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer)|$)", text, re.DOTALL)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        # 提取 Action 和 Action Input
        action_match = re.search(r"Action:\s*(.+)", text)
        if action_match:
            result["action"] = action_match.group(1).strip()

        input_match = re.search(r"Action Input:\s*(.+?)(?=\n\n|\Z)", text, re.DOTALL)
        if input_match:
            result["action_input"] = input_match.group(1).strip()

        return result

    def _execute_tool(self, action: str, action_input: str) -> str:
        """执行工具调用"""
        if action not in TOOLS:
            return f"错误：未知工具 '{action}'。可用工具: {', '.join(TOOLS.keys())}"
        try:
            return TOOLS[action]["func"](action_input)
        except Exception as e:
            return f"工具执行错误: {e}"

    def run(self, question: str) -> str:
        """运行ReAct循环"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question}
        ]

        print(f"\n{'='*60}")
        print(f"Agent 启动 | 模型: {self.model} | 最大步数: {self.max_steps}")
        print(f"问题: {question}")
        print(f"{'='*60}\n")

        for step in range(1, self.max_steps + 1):
            # 调用LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
            )
            text = response.choices[0].message.content

            # 解析输出
            parsed = self._parse_response(text)

            # 打印推理过程
            print(f"--- Step {step} ---")
            if parsed["thought"]:
                print(f"  💭 Thought: {parsed['thought']}")

            # 检查是否已有最终答案
            if parsed["final_answer"]:
                print(f"  ✅ Final Answer: {parsed['final_answer']}")
                print(f"\n{'='*60}")
                print(f"Agent 完成 | 共 {step} 步")
                print(f"{'='*60}")
                return parsed["final_answer"]

            # 执行工具
            if parsed["action"]:
                print(f"  🔧 Action: {parsed['action']}({parsed['action_input']})")
                observation = self._execute_tool(parsed["action"], parsed["action_input"])
                print(f"  👁️ Observation: {observation}")

                # 将结果加入对话历史
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}\n\n请继续思考。"
                })
            else:
                # LLM没输出有效的Action或Final Answer，直接返回
                print(f"  ⚠️ 无法解析，返回原始输出")
                return text

        # 超过最大步数
        print(f"\n⚠️ 达到最大步数 {self.max_steps}，强制停止。")
        return "抱歉，我在处理这个问题时达到了最大推理步数限制。"


# ============================================================
# 4. 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ReAct Agent 演示")
    parser.add_argument("question", nargs="?", default="2024年春节是几月几号？那天是周几？",
                        help="要问Agent的问题")
    parser.add_argument("--provider", choices=["deepseek", "zhipu"], default="deepseek",
                        help="使用哪个模型提供商 (默认: deepseek)")
    parser.add_argument("--max-steps", type=int, default=6, help="最大推理步数 (默认: 6)")
    args = parser.parse_args()

    agent = ReActAgent(provider=args.provider, max_steps=args.max_steps)
    answer = agent.run(args.question)
    print(f"\n🎯 最终答案: {answer}")


if __name__ == "__main__":
    main()
