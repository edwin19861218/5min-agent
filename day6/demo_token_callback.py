"""
Demo 3: 自定义Callback —— 追踪Token用量

展示如何继承BaseCallbackHandler，在Chain执行的各阶段自动统计Token消耗。
这是成本控制和性能调试的核心技术。

用法: python demo_token_callback.py --provider deepseek|zhipu
依赖: pip install openai langchain
"""

import argparse
import os
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler

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


class TokenTracker(BaseCallbackHandler):
    """自定义Callback：追踪每次LLM调用的Token用量"""

    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.call_count = 0

    def on_llm_end(self, response, **kwargs: Any) -> Any:
        """LLM调用结束时触发"""
        llm_output = response.llm_output or {}
        token_usage = llm_output.get("token_usage", {})

        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        total_tokens = token_usage.get("total_tokens", 0)

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.call_count += 1

        print(f"  [Token #{self.call_count}] "
              f"输入: {prompt_tokens} | 输出: {completion_tokens} | "
              f"合计: {total_tokens}")

    def report(self) -> str:
        """生成Token用量汇总报告"""
        total = self.total_prompt_tokens + self.total_completion_tokens
        # 参考价格: DeepSeek ¥1/百万input, ¥2/百万output
        estimated_cost = (
            self.total_prompt_tokens * 1 / 1_000_000
            + self.total_completion_tokens * 2 / 1_000_000
        )
        return (
            f"\n{'=' * 50}\n"
            f"Token用量汇总\n"
            f"{'=' * 50}\n"
            f"调用次数:     {self.call_count}\n"
            f"输入Token:    {self.total_prompt_tokens}\n"
            f"输出Token:    {self.total_completion_tokens}\n"
            f"总Token:      {total}\n"
            f"预估费用(元): {estimated_cost:.6f}\n"
            f"{'=' * 50}"
        )


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--provider", choices=["deepseek", "zhipu"], default="deepseek")
    args = cli_parser.parse_args()

    tracker = TokenTracker()
    cfg = PROVIDERS[args.provider]

    llm = ChatOpenAI(
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        temperature=0,
        callbacks=[tracker],  # 挂载Callback
    )

    prompt = ChatPromptTemplate.from_template("用一句话解释：{concept}")
    chain = prompt | llm | StrOutputParser()

    print(f"Token追踪Callback Demo（模型: {cfg['model']}）")
    print("=" * 50)

    # 连续调用3次，观察Token消耗
    concepts = ["什么是Agent", "什么是RAG", "什么是LangChain"]
    for concept in concepts:
        print(f"\n提问: {concept}")
        answer = chain.invoke({"concept": concept})
        print(f"回答: {answer}")

    print(tracker.report())
