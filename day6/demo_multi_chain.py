"""
Demo 1: 多步Chain —— 情感分析 → 信息提取 → 格式化输出

展示如何用LCEL（管道操作符 |）构建3步Chain，
数据在Chain中逐步流转：原始文本 → 情感标签 → 关键信息 → 最终报告

用法: python demo_multi_chain.py --provider deepseek|zhipu
依赖: pip install openai langchain
"""

import argparse
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ========== 模型配置（国产模型，OpenAI兼容） ==========
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
    """根据provider创建LLM实例"""
    cfg = PROVIDERS[provider]
    return ChatOpenAI(
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        temperature=0,
    )


def run_multi_chain(review_text: str, provider: str = "deepseek") -> dict:
    """运行多步Chain，返回每步的结果"""

    llm = create_llm(provider)

    # --- Chain 1: 情感分析 ---
    analyze_prompt = ChatPromptTemplate.from_template(
        "分析以下用户评论的情感倾向，只输出一个词：正面、负面或中性。\n\n"
        "用户评论：{review_text}"
    )
    sentiment_chain = analyze_prompt | llm | StrOutputParser()
    sentiment = sentiment_chain.invoke({"review_text": review_text})
    print(f"[Step 1] 情感分析: {sentiment}")

    # --- Chain 2: 关键信息提取 ---
    extract_prompt = ChatPromptTemplate.from_template(
        "从以下用户评论中提取关键信息，用JSON格式输出：\n"
        "- product: 提到的产品名\n"
        "- issue: 提到的问题（如果没有则为null）\n"
        "- suggestion: 改进建议（如果没有则为null）\n\n"
        "用户评论：{review_text}"
    )
    extract_chain = extract_prompt | llm | StrOutputParser()
    key_info = extract_chain.invoke({"review_text": review_text})
    print(f"[Step 2] 关键信息: {key_info}")

    # --- Chain 3: 格式化报告 ---
    report_prompt = ChatPromptTemplate.from_template(
        "根据分析结果，生成一份简短的客服摘要报告。\n\n"
        "情感倾向：{sentiment}\n"
        "关键信息：{key_info}\n"
        "原始评论：{review_text}\n\n"
        "请输出格式化的摘要报告（不超过100字）。"
    )
    report_chain = report_prompt | llm | StrOutputParser()
    report = report_chain.invoke({
        "sentiment": sentiment,
        "key_info": key_info,
        "review_text": review_text,
    })
    print(f"[Step 3] 摘要报告: {report}")

    return {
        "sentiment": sentiment,
        "key_info": key_info,
        "report": report,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["deepseek", "zhipu"], default="deepseek")
    args = parser.parse_args()

    review = (
        "我上个月买的AirPods Pro 2，降噪效果确实不错，"
        "但是连接MacBook时经常断连，很影响使用体验。"
        "建议苹果尽快修复蓝牙固件问题，不然考虑退货了。"
    )

    print("=" * 50)
    print(f"多步Chain Demo（模型: {PROVIDERS[args.provider]['model']}）")
    print("=" * 50)
    result = run_multi_chain(review, args.provider)
    print("\n" + "=" * 50)
    print("最终结果:")
    print(f"  情感: {result['sentiment']}")
    print(f"  信息: {result['key_info']}")
    print(f"  报告: {result['report']}")
