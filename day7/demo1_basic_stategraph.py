"""
第7篇 Demo 1: StateGraph基础 — 3步线性审批流
用最简的代码理解 State、Node、Edge 三件套

使用方式:
    pip install langgraph
    python demo1_basic_stategraph.py
"""


from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# ── 1. 定义状态（工作流的"共享记事本"）──────────────────────
class ApprovalState(TypedDict):
    request: str          # 申请内容
    requester: str        # 申请人
    reviewer: str         # 审核人
    status: str           # 审批状态
    comment: str          # 审批意见


# ── 2. 定义节点（每个处理步骤）────────────────────────────


def submit_request(state: ApprovalState) -> dict:
    """节点1: 提交申请"""
    print(f"[提交] {state['requester']} 提交了申请: {state['request']}")
    return {"status": "待审核", "comment": "申请已提交，等待初审"}


def auto_review(state: ApprovalState) -> dict:
    """节点2: 自动初审"""
    print(f"[初审] 自动审核中...")
    if "紧急" in state["request"]:
        return {"status": "加急待审", "comment": "检测到紧急关键词，已自动加急"}
    return {"status": "初审通过", "comment": "初审通过，等待人工审批"}


def final_approve(state: ApprovalState) -> dict:
    """节点3: 最终审批"""
    print(f"[审批] {state['reviewer']} 审批通过")
    return {"status": "已批准", "comment": f"{state['reviewer']}已批准，流程完成"}


# ── 3. 构建图 ────────────────────────────────────────────


def build_graph() -> StateGraph:
    """构建 StateGraph"""
    graph = StateGraph(ApprovalState)

    # 添加节点
    graph.add_node("submit", submit_request)
    graph.add_node("review", auto_review)
    graph.add_node("approve", final_approve)

    # 添加边：START → submit → review → approve → END
    graph.add_edge(START, "submit")
    graph.add_edge("submit", "review")
    graph.add_edge("review", "approve")
    graph.add_edge("approve", END)

    return graph


# ── 4. 运行 ──────────────────────────────────────────────


def main():
    print("=" * 50)
    print("Demo 1: StateGraph 基础 — 3步线性审批流")
    print("=" * 50 + "\n")

    # 编译图
    graph = build_graph()
    app = graph.compile()

    # 测试1: 普通申请
    print("--- 测试1: 普通申请 ---")
    result1 = app.invoke({
        "request": "申请一台外接显示器",
        "requester": "小明",
        "reviewer": "张经理",
        "status": "",
        "comment": "",
    })
    print(f"结果: {result1['status']} — {result1['comment']}\n")

    # 测试2: 紧急申请
    print("--- 测试2: 紧急申请 ---")
    result2 = app.invoke({
        "request": "服务器宕机，紧急申请备用服务器",
        "requester": "小李",
        "reviewer": "王总监",
        "status": "",
        "comment": "",
    })
    print(f"结果: {result2['status']} — {result2['comment']}\n")

    # 可视化图结构
    print("--- 图结构 (ASCII) ---")
    try:
        print(app.get_graph().print_ascii())
    except Exception:
        print("START → submit → review → approve → END")


if __name__ == "__main__":
    main()
