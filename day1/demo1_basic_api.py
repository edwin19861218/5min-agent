"""
第1篇 Demo 1: 最简API调用
5行代码和AI对话，支持 DeepSeek 和 智谱GLM-5.1

使用方式:
    1. 设置环境变量: export DEEPSEEK_API_KEY="你的key"
    2. 运行: python demo1_basic_api.py

    或使用智谱GLM-5.1:
    1. 设置环境变量: export ZHIPU_API_KEY="你的key"
    2. 运行: python demo1_basic_api.py --provider zhipu
"""

import os
import argparse


def create_client(provider: str = "deepseek"):
    """根据 provider 创建 OpenAI 兼容客户端"""
    from openai import OpenAI

    if provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")
        return OpenAI(api_key=api_key, base_url="https://api.deepseek.com"), "deepseek-chat"
    elif provider == "zhipu":
        api_key = os.environ.get("ZHIPU_API_KEY")
        if not api_key:
            raise ValueError("请设置环境变量 ZHIPU_API_KEY")
        return OpenAI(api_key=api_key, base_url="https://open.bigmodel.cn/api/paas/v4/"), "glm-5.1"
    else:
        raise ValueError(f"不支持的 provider: {provider}，请选择 deepseek 或 zhipu")


def main():
    parser = argparse.ArgumentParser(description="第一个LLM API调用")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "zhipu"],
                        help="选择模型提供商 (默认: deepseek)")
    args = parser.parse_args()

    client, model = create_client(args.provider)
    print(f"使用模型: {model}\n")

    # --- 核心: 5行代码完成一次API调用 ---
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "用一句话解释什么是Agent"}]
    )

    answer = response.choices[0].message.content
    print(f"AI回答: {answer}")

    # 打印token使用量
    if hasattr(response, 'usage') and response.usage:
        print(f"\n--- Token统计 ---")
        print(f"输入tokens: {response.usage.prompt_tokens}")
        print(f"输出tokens: {response.usage.completion_tokens}")
        print(f"总计tokens: {response.usage.total_tokens}")


if __name__ == "__main__":
    main()
