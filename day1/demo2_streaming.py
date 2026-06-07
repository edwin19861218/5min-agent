"""
第1篇 Demo 2: 流式输出
让AI"边想边说"，像网页版一样逐字打印

使用方式:
    1. 设置环境变量: export DEEPSEEK_API_KEY="你的key"
    2. 运行: python demo2_streaming.py

    或使用智谱GLM-5.1:
    1. 设置环境变量: export ZHIPU_API_KEY="你的key"
    2. 运行: python demo2_streaming.py --provider zhipu
"""

import os
import argparse
import sys


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


def chat_stream(client, model, user_message):
    """流式输出: 逐字打印AI的回答"""
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_message}],
        stream=True  # 关键: 开启流式输出
    )

    print("AI: ", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()  # 最后换行


def main():
    parser = argparse.ArgumentParser(description="流式输出Demo")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "zhipu"],
                        help="选择模型提供商 (默认: deepseek)")
    args = parser.parse_args()

    client, model = create_client(args.provider)
    print(f"使用模型: {model} (流式输出模式)")
    print("输入你的问题，输入 q 退出\n")

    while True:
        user_input = input("你: ").strip()
        if user_input.lower() == "q":
            print("再见！")
            break
        if not user_input:
            continue

        chat_stream(client, model, user_input)
        print()


if __name__ == "__main__":
    main()
