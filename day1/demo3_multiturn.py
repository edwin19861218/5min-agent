"""
第1篇 Demo 3: 多轮对话
让AI记住你说过的话，实现上下文连续对话

使用方式:
    1. 设置环境变量: export DEEPSEEK_API_KEY="你的key"
    2. 运行: python demo3_multiturn.py

    或使用智谱GLM-5.1:
    1. 设置环境变量: export ZHIPU_API_KEY="你的key"
    2. 运行: python demo3_multiturn.py --provider zhipu
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
    parser = argparse.ArgumentParser(description="多轮对话Demo")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "zhipu"],
                        help="选择模型提供商 (默认: deepseek)")
    args = parser.parse_args()

    client, model = create_client(args.provider)
    print(f"使用模型: {model} (多轮对话模式)")
    print("输入你的问题，输入 q 退出，输入 c 清空对话历史\n")

    # 对话历史: 一直追加，LLM靠这个"记住"之前说的话
    messages = []

    # 系统提示词: 设定AI的角色
    messages.append({
        "role": "system",
        "content": "你是一个友好的AI助手，回答简洁明了。"
    })

    while True:
        user_input = input("你: ").strip()
        if user_input.lower() == "q":
            print("再见！")
            break
        if user_input.lower() == "c":
            messages = [messages[0]]  # 保留system prompt，清空对话
            print("(对话历史已清空)\n")
            continue
        if not user_input:
            continue

        # 把用户消息加入历史
        messages.append({"role": "user", "content": user_input})

        # 调用API，传入完整对话历史
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )

        assistant_message = response.choices[0].message.content
        print(f"AI: {assistant_message}\n")

        # 把AI回答也加入历史，这样下一轮对话AI就能"记住"了
        messages.append({"role": "assistant", "content": assistant_message})

        # 显示当前对话轮数和token消耗
        turn_count = (len(messages) - 1) // 2  # 减去system prompt
        if hasattr(response, 'usage') and response.usage:
            print(f"  [第{turn_count}轮对话, 本次消耗{response.usage.total_tokens} tokens]")
        else:
            print(f"  [第{turn_count}轮对话]")


if __name__ == "__main__":
    main()
