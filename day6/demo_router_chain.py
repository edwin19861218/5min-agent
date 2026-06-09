"""
Demo 4: Router Chain —— 根据输入动态选择处理链

展示如何根据用户输入的内容，自动路由到不同的专业处理链。
这是构建智能客服、多技能Agent的核心模式。

用法: python demo_router_chain.py --provider deepseek|zhipu
依赖: pip install openai langchain
"""

import argparse
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ========== 模型配置 ==========
PROVIDERS = {
    "deepseek": {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    },
    "zhipu": {
        "model": "glm-5.1",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "api_key": os.getenv("ZHIPU_API_KEY", ""),
    },
}


def create_llm(provider: str) -> ChatOpenAI:
    cfg = PROVIDERS[provider]
    return ChatOpenAI(
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        temperature=0,
    )


def run_router_chain(provider: str = "deepseek"):
    llm = create_llm(provider)

    # ========== 定义3条专业处理链 ==========

    # 链1: 技术支持
    tech_prompt = ChatPromptTemplate.from_template(
        "你是技术支持工程师。请用专业但易懂的语言回答以下技术问题。\n"
        "如果涉及具体操作，请给出分步骤指导。\n\n"
        "用户问题：{question}"
    )
    tech_chain = tech_prompt | llm | StrOutputParser()

    # 链2: 销售咨询
    sales_prompt = ChatPromptTemplate.from_template(
        "你是销售顾问。请热情友好地回答以下咨询。\n"
        "突出产品优势和适用场景，必要时提供选择建议。\n\n"
        "用户问题：{question}"
    )
    sales_chain = sales_prompt | llm | StrOutputParser()

    # 链3: 投诉处理
    complaint_prompt = ChatPromptTemplate.from_template(
        "你是客户关系经理。请用同理心回应以下投诉。\n"
        "先道歉和表示理解，然后给出具体的解决方案或补偿建议。\n\n"
        "用户问题：{question}"
    )
    complaint_chain = complaint_prompt | llm | StrOutputParser()

    # ========== Router：分类 + 路由 ==========
    classify_prompt = ChatPromptTemplate.from_template(
        "判断以下用户问题属于哪个类别，只输出类别名（不要输出其他内容）：\n\n"
        "类别：\n"
        "- tech: 技术问题、故障排查、使用方法\n"
        "- sales: 产品咨询、价格、推荐、对比\n"
        "- complaint: 投诉、不满、退款、差评\n\n"
        "用户问题：{question}\n\n类别:"
    )
    classify_chain = classify_prompt | llm | StrOutputParser()

    chain_map = {
        "tech": tech_chain,
        "sales": sales_chain,
        "complaint": complaint_chain,
    }

    test_questions = [
        "我的AirPods连不上MacBook怎么办？",
        "AirPods Pro和索尼XM5哪个降噪更好？",
        "你们的客服太差了，等了三天都没人回复，要投诉！",
    ]

    for q in test_questions:
        print(f"\n用户: {q}")
        category = classify_chain.invoke({"question": q}).strip().lower()
        print(f"  [Router] 分类结果: {category}")
        target_chain = chain_map.get(category, tech_chain)
        answer = target_chain.invoke({"question": q})
        print(f"回复: {answer}")
        print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["deepseek", "zhipu"], default="deepseek")
    args = parser.parse_args()

    print(f"Router Chain Demo（模型: {PROVIDERS[args.provider]['model']}）")
    print("=" * 50)
    run_router_chain(args.provider)
