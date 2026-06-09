"""
第7篇 Demo 2: 条件路由客服分流
用 ConditionalEdge 实现 AI 自动分类 + 分流到不同处理节点

使用方式:
    1. pip install langgraph openai
    2. 设置环境变量: export DEEPSEEK_API_KEY="你的key"
    3. 运行: python demo2_conditional_routing.py
    或使用智谱GLM:
       export ZHIPU_API_KEY="你的key"
       python demo2_conditional_routing.py --provider zhipu
"""

import os
import argparse
from typing import TypedDict

from openai import OpenAI
from langgraph.graph import StateGraph, START, END


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


class ServiceState(TypedDict):
    query: str       # 用户问题
    category: str    # 分类结果
    response: str    # 最终回复


# ── 节点函数 ─────────────────────────────────────────────

# client 和 model 在 main() 中初始化
_client = None
_model = None


def classify(state: ServiceState) -> dict:
    """分类节点: 用 LLM 判断问题类型"""
    prompt = f"""请将以下客服问题归入一个类别，只回复类别名称，不要解释。

类别选项：
- 技术支持
- 账单问题
- 通用咨询

用户问题: {state['query']}"""

    resp = _client.chat.completions.create(
        model=_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()

    # 规范化分类结果
    if "技术" in raw:
        category = "技术支持"
    elif "账单" in raw or "费用" in raw or "扣款" in raw:
        category = "账单问题"
    else:
        category = "通用咨询"

    print(f"  [分类] {state['query'][:30]}... → {category}")
    return {"category": category}


def route(state: ServiceState) -> str:
    """路由函数: 根据分类结果返回目标节点名"""
    mapping = {"技术支持": "tech", "账单问题": "billing", "通用咨询": "general"}
    return mapping.get(state["category"], "general")


def handle_tech(state: ServiceState) -> dict:
    """技术支持节点"""
    print("  [路由] → 技术支持")
    resp = _client.chat.completions.create(
        model=_model,
        messages=[
            {"role": "system", "content": "你是技术支持专家，用2-3句话简洁回答技术问题。"},
            {"role": "user", "content": state["query"]},
        ],
    )
    return {"response": resp.choices[0].message.content}


def handle_billing(state: ServiceState) -> dict:
    """账单服务节点"""
    print("  [路由] → 账单服务")
    resp = _client.chat.completions.create(
        model=_model,
        messages=[
            {"role": "system", "content": "你是账单服务专员，用2-3句话简洁回答账单相关问题。"},
            {"role": "user", "content": state["query"]},
        ],
    )
    return {"response": resp.choices[0].message.content}


def handle_general(state: ServiceState) -> dict:
    """通用咨询节点"""
    print("  [路由] → 通用咨询")
    resp = _client.chat.completions.create(
        model=_model,
        messages=[
            {"role": "system", "content": "你是友好的客服代表，用2-3句话简洁回答咨询。"},
            {"role": "user", "content": state["query"]},
        ],
    )
    return {"response": resp.choices[0].message.content}


# ── 构建图 ──────────────────────────────────────────────


def build_graph() -> StateGraph:
    graph = StateGraph(ServiceState)

    # 节点
    graph.add_node("classify", classify)
    graph.add_node("tech", handle_tech)
    graph.add_node("billing", handle_billing)
    graph.add_node("general", handle_general)

    # 边
    graph.add_edge(START, "classify")

    # 条件路由：classify 的输出决定走哪条路
    graph.add_conditional_edges(
        "classify",
        route,
        {"tech": "tech", "billing": "billing", "general": "general"},
    )

    graph.add_edge("tech", END)
    graph.add_edge("billing", END)
    graph.add_edge("general", END)

    return graph


# ── 主程序 ──────────────────────────────────────────────


def main():
    global _client, _model

    parser = argparse.ArgumentParser(description="条件路由客服分流")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "zhipu"])
    parser.add_argument("--query", default=None)
    args = parser.parse_args()

    _client, _model = create_client(args.provider)
    print(f"模型: {_model}\n")

    # 测试问题
    queries = (
        [args.query]
        if args.query
        else [
            "登录页面一直报404，清了缓存也没用",
            "上个月多扣了我两次会员费，怎么退款",
            "你们公司周末上班吗？想去门店看看",
        ]
    )

    app = build_graph().compile()

    for q in queries:
        print(f"用户: {q}")
        result = app.invoke({"query": q, "category": "", "response": ""})
        print(f"回复: {result['response']}")
        print("-" * 50)


if __name__ == "__main__":
    main()
