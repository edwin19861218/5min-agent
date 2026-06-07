# 5min-agent

「从0-1成为Agent架构师」系列配套代码

每天5分钟，20天从零学会用Python设计并构建AI Agent系统。

## 目录结构

```
day1/   - Agent概念入门 + Python调用LLM API
day2/   - Prompt工程（待更新）
day3/   - ReAct模式（待更新）
...
day20/  - 完整项目：从零到上线（待更新）
```

## 环境准备

```bash
pip install openai
```

设置API Key（二选一）：
```bash
# DeepSeek
export DEEPSEEK_API_KEY="你的key"

# 智谱GLM-5.1
export ZHIPU_API_KEY="你的key"
```

## 技术栈

- **模型**：DeepSeek（deepseek-chat）+ 智谱GLM-5.1（glm-5.1）
- **SDK**：openai（国产模型兼容OpenAI格式）
- **语言**：Python 3.10+
