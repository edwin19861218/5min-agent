"""
第7篇 Demo 3: 多步审批Agent — 断点续传 + 人工审批
用 LangGraph Checkpointer 实现暂停/恢复工作流

使用方式:
    1. pip install langgraph openai
    2. 设置环境变量: export DEEPSEEK_API_KEY="你的key"
    3. 运行: python demo3_approval_workflow.py
    或使用智谱GLM:
       export ZHIPU_API_KEY="你的key"
       python demo3_approval_workflow.py --provider zhipu

断点续传流程:
    submit(提交) → review(AI初审) → [暂停等人工] → approve/reject(审批) → done(归档)
"""

import os
import argparse
from typing import TypedDict

from openai import OpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# ── 模型配置 ─────────────────────────────────────────────


def create_client(provider: str = "deepseek"):
    """根据 provider 创建 OpenAI 兼容客户端"""
    if provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")
        return OpenAI(api_key=api_key, base_url="https://api.deepseek.com"), "deepseek-chat"
    elif provider == "zhipu":
        api_key = os.environ.get("ZHIPU_API_KEY")
        if not api_key:
            raise ValueError("请设置环境变量 ZHIPU_API_KEY")
        return (
            OpenAI(api_key=api_key, base_url="https://open.bigmodel.cn/api/paas/v4/"),
            "glm-5.1",
        )
    else:
        raise ValueError(f"不支持的 provider: {provider}")


# ── 状态定义 ─────────────────────────────────────────────


class ApprovalState(TypedDict):
    request: str       # 申请内容
    amount: float      # 申请金额
    requester: str     # 申请人
    ai_review: str     # AI初审意见
    human_decision: str  # 人工审批决定 (approve/reject)
    human_comment: str   # 人工审批备注
    final_status: str  # 最终状态
    final_comment: str # 最终备注


# ── 全局变量 ─────────────────────────────────────────────

_client = None
_model = None


# ── 节点函数 ─────────────────────────────────────────────


def submit(state: ApprovalState) -> dict:
    """节点1: 提交申请"""
    print(f"  [提交] {state['requester']} 申请: {state['request']} (金额: {state['amount']}元)")
    return {"final_status": "已提交", "final_comment": "申请已提交"}


def ai_review(state: ApprovalState) -> dict:
    """节点2: AI初审 — 用 LLM 分析申请合理性"""
    prompt = f"""你是一个费用审核助手。请用1-2句话评估以下申请的合理性。

申请人: {state['requester']}
申请内容: {state['request']}
申请金额: {state['amount']}元

只给出审核意见，不要做最终决定。"""

    resp = _client.chat.completions.create(
        model=_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    review = resp.choices[0].message.content.strip()
    print(f"  [AI初审] {review}")
    return {"ai_review": review, "final_status": "AI已初审"}


def human_approve(state: ApprovalState) -> dict:
    """节点3: 人工审批（由外部调用 update_state 注入决定）"""
    decision = state.get("human_decision", "")
    comment = state.get("human_comment", "")

    if decision == "approve":
        print(f"  [人工审批] 批准 — {comment}")
        return {
            "final_status": "已批准",
            "final_comment": f"审批通过: {comment}",
        }
    else:
        print(f"  [人工审批] 驳回 — {comment}")
        return {
            "final_status": "已驳回",
            "final_comment": f"审批驳回: {comment}",
        }


def archive(state: ApprovalState) -> dict:
    """节点4: 归档"""
    status = state.get("final_status", "")
    print(f"  [归档] 状态={status}, 已记录到系统")
    return {"final_comment": state.get("final_comment", "") + " [已归档]"}


# ── 路由函数 ─────────────────────────────────────────────


def after_human_approve(state: ApprovalState) -> str:
    """人工审批后路由: 批准则归档，驳回也归档"""
    return "archive"


# ── 构建图 ──────────────────────────────────────────────


def build_graph() -> StateGraph:
    graph = StateGraph(ApprovalState)

    # 节点
    graph.add_node("submit", submit)
    graph.add_node("ai_review", ai_review)
    graph.add_node("human_approve", human_approve)
    graph.add_node("archive", archive)

    # 边: submit → ai_review → human_approve → archive → END
    graph.add_edge(START, "submit")
    graph.add_edge("submit", "ai_review")
    graph.add_edge("ai_review", "human_approve")
    graph.add_edge("human_approve", "archive")
    graph.add_edge("archive", END)

    return graph


# ── 演示: 完整的断点续传流程 ─────────────────────────────


def demo_approve_scenario(app, thread_id: str):
    """演示: 人工批准的完整流程"""
    config = {"configurable": {"thread_id": thread_id}}
    print("\n" + "=" * 60)
    print(f"场景: 人工批准 (thread={thread_id})")
    print("=" * 60)

    # Step 1: 提交并AI初审（会在 human_approve 前暂停）
    print("\n--- Step 1: 提交申请 + AI初审 ---")
    result = app.invoke(
        {
            "request": "购买团队开发工具许可证",
            "amount": 3500,
            "requester": "小明",
            "ai_review": "",
            "human_decision": "",
            "human_comment": "",
            "final_status": "",
            "final_comment": "",
        },
        config,
    )
    print(f"  当前状态: {result['final_status']}")

    # 查看暂停位置
    state = app.get_state(config)
    print(f"  暂停位置: 下一个待执行节点 = {state.next}")
    print(f"  AI初审意见: {result.get('ai_review', '无')}")

    # Step 2: 模拟人工审批（注入决定）
    print("\n--- Step 2: 人工审批 ---")
    app.update_state(
        config,
        {"human_decision": "approve", "human_comment": "预算充足，同意购买"},
        as_node="human_approve",
    )

    # Step 3: 恢复执行
    print("\n--- Step 3: 恢复执行 ---")
    result = app.invoke(None, config)
    print(f"  最终状态: {result['final_status']}")
    print(f"  最终备注: {result['final_comment']}")


def demo_reject_scenario(app, thread_id: str):
    """演示: 人工驳回的流程"""
    config = {"configurable": {"thread_id": thread_id}}
    print("\n" + "=" * 60)
    print(f"场景: 人工驳回 (thread={thread_id})")
    print("=" * 60)

    # Step 1: 提交
    print("\n--- Step 1: 提交申请 + AI初审 ---")
    result = app.invoke(
        {
            "request": "团建旅游费用报销",
            "amount": 50000,
            "requester": "小王",
            "ai_review": "",
            "human_decision": "",
            "human_comment": "",
            "final_status": "",
            "final_comment": "",
        },
        config,
    )
    print(f"  当前状态: {result['final_status']}")
    print(f"  AI初审意见: {result.get('ai_review', '无')}")

    # Step 2: 人工驳回
    print("\n--- Step 2: 人工审批 (驳回) ---")
    app.update_state(
        config,
        {"human_decision": "reject", "human_comment": "金额过大，请提供详细明细后再申请"},
        as_node="human_approve",
    )

    # Step 3: 恢复
    print("\n--- Step 3: 恢复执行 ---")
    result = app.invoke(None, config)
    print(f"  最终状态: {result['final_status']}")
    print(f"  最终备注: {result['final_comment']}")


# ── 主程序 ──────────────────────────────────────────────


def main():
    global _client, _model

    parser = argparse.ArgumentParser(description="多步审批Agent — 断点续传")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "zhipu"])
    args = parser.parse_args()

    _client, _model = create_client(args.provider)
    print(f"模型: {_model}")

    # 构建带 Checkpointer 的图（interrupt_before="human_approve" 实现断点）
    graph = build_graph()
    checkpointer = MemorySaver()
    app = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_approve"],
    )

    # 运行两个场景
    demo_approve_scenario(app, "approval-001")
    demo_reject_scenario(app, "approval-002")

    print("\n" + "=" * 60)
    print("演示完成!")
    print("关键点:")
    print("  1. interrupt_before 让工作流在 human_approve 前暂停")
    print("  2. update_state 注入人工审批决定")
    print("  3. invoke(None, config) 从断点恢复执行")
    print("=" * 60)


if __name__ == "__main__":
    main()
