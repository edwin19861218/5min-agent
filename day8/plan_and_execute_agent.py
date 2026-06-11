#!/usr/bin/env python3
"""
Plan-and-Execute Agent Demo
「从0-1成为Agent架构师」系列 · 第8篇配套代码

运行方式:
  python plan_and_execute_agent.py --provider deepseek --goal "调研LangGraph框架在生产环境中的优缺点"
  python plan_and_execute_agent.py --provider zhipu --goal "分析2026年最值得学习的Agent开发技术"
"""

import argparse
import os
import re
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# ── 模型配置 ──────────────────────────────────────────────

MAX_REPLAN = 3  # 最大重规划次数

PROVIDERS = {
    "deepseek": {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    },
    "zhipu": {
        "model": "glm-5.1",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key": os.getenv("ZHIPU_API_KEY", ""),
    },
}


def create_llm(provider: str) -> ChatOpenAI:
    """根据 provider 返回 ChatOpenAI 实例（平台无关客户端）"""
    cfg = PROVIDERS[provider]
    return ChatOpenAI(
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        temperature=0,
    )


# ── 状态定义 ──────────────────────────────────────────────


class PlanExecuteState(TypedDict):
    """Plan-and-Execute Agent 的全局状态"""
    goal: str                    # 用户原始目标
    plan: list[str]              # 当前任务计划（步骤列表）
    past_steps: list[tuple]      # 已完成的步骤及其结果 [(step, result), ...]
    response: str                # 最终输出
    replan_count: int            # 已执行的重规划次数


# ── Planner 节点 ──────────────────────────────────────────

PLANNER_PROMPT = """你是一个任务规划专家。
用户目标：{goal}

请将目标拆解为 3-6 个有序步骤，每步一个具体可执行的子任务。
输出格式：每行一个步骤，编号开头（1. 2. 3. ...），不要额外解释。

示例：
1. 搜索相关资料了解基本概念
2. 收集实际使用案例
3. 对比分析优缺点
4. 总结核心观点
"""


def plan_step(state: PlanExecuteState, llm) -> PlanExecuteState:
    """Planner 节点：将目标拆解为步骤"""
    response = llm.invoke(PLANNER_PROMPT.format(goal=state["goal"]))
    steps = []
    for line in response.content.strip().split("\n"):
        line = line.strip()
        if re.match(r"^\d+\.", line):
            # 去掉编号前缀
            step = re.sub(r"^\d+\.\s*", "", line)
            if step:
                steps.append(step)
    if not steps:
        # 如果解析失败，把整段作为单步
        steps = [state["goal"]]
    print(f"📋 生成计划（{len(steps)} 步）：")
    for i, s in enumerate(steps, 1):
        print(f"   {i}. {s}")
    return {"plan": steps, "replan_count": 0}


# ── Executor 节点 ─────────────────────────────────────────

EXECUTOR_PROMPT = """你是一个任务执行者。
当前任务：{task}

以下是之前已完成步骤的结果（供参考）：
{past_context}

请完成当前任务，给出详细的结果。"""


def execute_step(state: PlanExecuteState, llm) -> PlanExecuteState:
    """Executor 节点：执行计划中的第一个步骤"""
    if not state["plan"]:
        return {"response": "计划为空，无法执行。"}

    task = state["plan"][0]
    remaining_plan = state["plan"][1:]
    done_count = len(state.get("past_steps", [])) + 1
    print(f"▶ [{done_count}] 执行步骤：{task}")

    # 构建已完成步骤的上下文
    past_context = ""
    if state["past_steps"]:
        lines = []
        for i, (s, r) in enumerate(state["past_steps"], 1):
            lines.append(f"步骤{i}「{s}」的结果：{r[:500]}")
        past_context = "\n".join(lines)
    else:
        past_context = "（暂无已完成步骤）"

    response = llm.invoke(
        EXECUTOR_PROMPT.format(task=task, past_context=past_context)
    )

    return {
        "plan": remaining_plan,
        "past_steps": state.get("past_steps", []) + [(task, response.content)],
    }


# ── Replanner 节点 ────────────────────────────────────────

REPLANNER_PROMPT = """你是一个计划审查者。
用户原始目标：{goal}

已完成的步骤：
{past_context}

剩余计划：
{remaining_plan}

请判断剩余计划是否需要调整。

规则：
- 如果计划仍然合理，只需输出：NO_CHANGE
- 如果需要调整，输出新的完整计划（每行一个步骤，编号开头）
- 新计划应该是 3-6 个步骤，涵盖剩余需要做的事
"""


def replan_step(state: PlanExecuteState, llm) -> PlanExecuteState:
    """Replanner 节点：检查是否需要重规划"""
    # 检查重规划次数上限
    current_count = state.get("replan_count", 0)
    if current_count >= MAX_REPLAN:
        return {"replan_count": current_count}

    # 构建已完成步骤上下文
    past_context = ""
    if state["past_steps"]:
        lines = []
        for i, (s, r) in enumerate(state["past_steps"], 1):
            lines.append(f"步骤{i}「{s}」的结果摘要：{r[:300]}")
        past_context = "\n".join(lines)

    remaining_plan = "\n".join(
        f"{i+1}. {s}" for i, s in enumerate(state["plan"])
    ) if state["plan"] else "（无剩余步骤）"

    response = llm.invoke(
        REPLANNER_PROMPT.format(
            goal=state["goal"],
            past_context=past_context,
            remaining_plan=remaining_plan,
        )
    )

    content = response.content.strip()

    # 判断是否需要重规划
    if "NO_CHANGE" in content.upper():
        return {"replan_count": current_count}

    # 解析新计划
    new_steps = []
    for line in content.split("\n"):
        line = line.strip()
        if re.match(r"^\d+\.", line):
            step = re.sub(r"^\d+\.\s*", "", line)
            if step:
                new_steps.append(step)

    if new_steps:
        return {"plan": new_steps, "replan_count": current_count + 1}

    print("⚠️ Replanner 解析失败，保持原计划")
    # 解析失败，保持原计划，但仍然计数以防无限触发
    return {"plan": state["plan"], "replan_count": current_count + 1}


# ── 汇总节点 ──────────────────────────────────────────────

RESPONSE_PROMPT = """你是一个总结者。
用户原始目标：{goal}

以下是所有已完成步骤的结果：
{all_results}

请根据以上信息，给用户一个完整、有条理的回答。"""


def generate_response(state: PlanExecuteState, llm) -> PlanExecuteState:
    """汇总节点：将所有步骤结果整合为最终回答"""
    all_results = ""
    for i, (step, result) in enumerate(state["past_steps"], 1):
        all_results += f"\n步骤{i}「{step}」：\n{result[:800]}\n"

    response = llm.invoke(
        RESPONSE_PROMPT.format(goal=state["goal"], all_results=all_results)
    )

    return {"response": response.content}


# ── 条件路由 ──────────────────────────────────────────────


def should_end(state: PlanExecuteState) -> str:
    """判断是否结束执行"""
    if state.get("plan") and len(state["plan"]) > 0:
        return "execute"
    return "respond"


def should_replan(state: PlanExecuteState) -> str:
    """判断是否需要重规划"""
    if not state.get("plan") or len(state["plan"]) == 0:
        return "respond"
    # 每执行 2 步触发一次重规划检查
    past_count = len(state.get("past_steps", []))
    if past_count > 0 and past_count % 2 == 0:
        current_replan = state.get("replan_count", 0)
        if current_replan < MAX_REPLAN:
            return "replan"
    return "execute"


# ── 构建工作流图 ──────────────────────────────────────────


def build_graph(llm):
    """构建 Plan-and-Execute 工作流"""
    graph = StateGraph(PlanExecuteState)

    # 添加节点
    graph.add_node("planner", lambda s: plan_step(s, llm))
    graph.add_node("executor", lambda s: execute_step(s, llm))
    graph.add_node("replanner", lambda s: replan_step(s, llm))
    graph.add_node("respond", lambda s: generate_response(s, llm))

    # 设置入口
    graph.set_entry_point("planner")

    # 连边
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges(
        "executor",
        should_replan,
        {
            "replan": "replanner",
            "execute": "executor",
            "respond": "respond",
        },
    )
    graph.add_conditional_edges(
        "replanner",
        should_end,
        {
            "execute": "executor",
            "respond": "respond",
        },
    )
    graph.add_edge("respond", END)

    return graph.compile()


# ── 主函数 ────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Plan-and-Execute Agent Demo")
    parser.add_argument(
        "--provider",
        choices=["deepseek", "zhipu"],
        default="deepseek",
        help="LLM 提供商",
    )
    parser.add_argument(
        "--goal",
        type=str,
        required=True,
        help="要完成的目标",
    )
    args = parser.parse_args()

    print(f"🔧 Provider: {args.provider}")
    print(f"🎯 Goal: {args.goal}")
    print("=" * 60)

    llm = create_llm(args.provider)
    app = build_graph(llm)

    # 初始化状态
    initial_state = {
        "goal": args.goal,
        "plan": [],
        "past_steps": [],
        "response": "",
        "replan_count": 0,
    }

    # 执行工作流
    result = app.invoke(initial_state)

    print("=" * 60)
    print("📋 最终结果：\n")
    print(result.get("response", "（无结果）"))


if __name__ == "__main__":
    main()
