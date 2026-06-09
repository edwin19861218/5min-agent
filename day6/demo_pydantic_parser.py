"""
Demo 2: PydanticOutputParser —— 让LLM输出结构化数据

展示如何用Pydantic模型定义期望的输出格式，
Parser自动生成格式指令注入Prompt，LLM按指令输出，Parser解析验证。

用法: python demo_pydantic_parser.py --provider deepseek|zhipu
依赖: pip install openai langchain pydantic
"""

import argparse
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

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


# ========== 定义输出结构 ==========
class ProductReview(BaseModel):
    """产品评价的结构化数据"""
    product: str = Field(description="产品名称")
    rating: int = Field(description="评分，1-5分")
    pros: list[str] = Field(description="优点列表")
    cons: list[str] = Field(description="缺点列表")
    summary: str = Field(description="一句话总结")


def analyze_review(review_text: str, provider: str = "deepseek") -> ProductReview:
    """分析一条产品评价，返回结构化数据"""

    cfg = PROVIDERS[provider]
    llm = ChatOpenAI(
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        temperature=0,
    )

    parser = PydanticOutputParser(pydantic_object=ProductReview)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个产品评价分析助手。将用户评价转换为结构化数据。"),
        ("human", "{review_text}\n\n{format_instructions}"),
    ])

    # LCEL管道：prompt → llm → parser
    chain = prompt | llm | parser

    return chain.invoke({
        "review_text": review_text,
        "format_instructions": parser.get_format_instructions(),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["deepseek", "zhipu"], default="deepseek")
    args = parser.parse_args()

    review = (
        "买了这款Sony WH-1000XM5耳机用了两周，说说感受。"
        "降噪效果绝对是业界顶级，在地铁上几乎听不到外界声音。"
        "音质也很棒，低频有力度但不会轰头。"
        "不过价格确实有点贵，而且耳机不能折叠，携带不太方便。"
        "另外触摸控制偶尔会误触。"
        "总体来说如果你预算充足，还是很推荐的。"
    )

    print(f"PydanticOutputParser Demo（模型: {PROVIDERS[args.provider]['model']}）")
    print("=" * 50)
    result = analyze_review(review, args.provider)

    print(f"产品: {result.product}")
    print(f"评分: {'⭐' * result.rating} ({result.rating}/5)")
    print(f"优点: {', '.join(result.pros)}")
    print(f"缺点: {', '.join(result.cons)}")
    print(f"总结: {result.summary}")
