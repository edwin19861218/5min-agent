"""langchain_memory_demo.py — LangChain Memory组件实战
对比 ConversationBufferMemory 和 ConversationSummaryMemory 思路
用法：python langchain_memory_demo.py --provider deepseek
"""
import argparse
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END

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


def demo_buffer_memory():
    """Demo A: 完整保留对话历史（Buffer策略）"""
    print("\n" + "=" * 50)
    print("Demo A: Buffer策略 — 完整保留所有对话")
    print("=" * 50)

    messages = [SystemMessage(content="你是一个Python编程助手。")]
    questions = [
        "什么是列表推导式？",
        "能给个例子吗？",
        "如果我只要偶数呢？",
        "怎么嵌套两层？",
        "跟map/filter比哪个好？",
    ]

    total_chars = 0
    for q in questions:
        messages.append(HumanMessage(content=q))
        total_chars += len(q)
        messages.append(AIMessage(content=f"关于'{q}'的回答...（省略）"))
        total_chars += 50

    print(f"  对话轮数: {len(questions)}")
    print(f"  消息条数: {len(messages)}")
    print(f"  总字符数: {total_chars} (约{total_chars // 3} token)")
    print("  ✅ 优点: 信息零丢失")
    print("  ❌ 缺点: 随对话增长，token消耗线性增长")


def demo_summary_memory(llm):
    """Demo B: 自动摘要压缩（Summary策略）"""
    print("\n" + "=" * 50)
    print("Demo B: Summary策略 — 旧对话自动压缩为摘要")
    print("=" * 50)

    history = [
        HumanMessage(content="我想做一个天气查询Agent"),
        AIMessage(content="好的，你需要调用天气API，比如OpenWeatherMap"),
        HumanMessage(content="用户说中文怎么办？"),
        AIMessage(content="可以用LLM翻译，或者直接用国内天气API"),
    ]

    history_text = "\n".join(f"{m.type}: {m.content}" for m in history)
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "将以下对话压缩成一段摘要，保留关键决定和用户偏好，不超过100字。"),
        ("human", "{history_text}"),
    ])

    chain = summary_prompt | llm
    result = chain.invoke({"history_text": history_text})
    print(f"  原始对话: {len(history)}条, 约{len(history_text) // 3} token")
    print(f"  压缩摘要: {result.content}")
    print("  ✅ 优点: token消耗恒定，适合长对话")
    print("  ❌ 缺点: 摘要可能丢失细节")


def demo_langgraph_memory(llm):
    """Demo C: 用LangGraph实现带记忆的对话Agent"""
    print("\n" + "=" * 50)
    print("Demo C: LangGraph实现带记忆的对话Agent")
    print("=" * 50)

    system_prompt = """你是一个Python编程助手。
如果对话历史中有相关上下文，请参考之前的对话内容作答。
回答要简洁，每次不超过50字。"""

    def chatbot(state: MessagesState):
        response = llm.invoke(
            [SystemMessage(content=system_prompt)] + state["messages"]
        )
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("chatbot", chatbot)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)
    app = graph.compile()

    test_inputs = [
        "我叫小明，我在学Python",
        "你能记住我的名字吗？",
    ]

    all_messages = []
    for user_input in test_inputs:
        all_messages.append(HumanMessage(content=user_input))
        result = app.invoke({"messages": all_messages})
        ai_response = result["messages"][-1]
        all_messages.append(ai_response)
        print(f"\n  用户: {user_input}")
        print(f"  Agent: {ai_response.content}")


def main():
    parser = argparse.ArgumentParser(description="LangChain Memory Demo")
    parser.add_argument("--provider", choices=["deepseek", "zhipu"], default="deepseek")
    args = parser.parse_args()

    cfg = PROVIDERS[args.provider]
    llm = ChatOpenAI(
        api_key=os.getenv(cfg["api_key_env"]),
        base_url=cfg["base_url"],
        model=cfg["model"],
        temperature=0.3,
    )
    print(f"使用模型: {cfg['model']}\n")

    demo_buffer_memory()
    demo_summary_memory(llm)
    demo_langgraph_memory(llm)


if __name__ == "__main__":
    main()
