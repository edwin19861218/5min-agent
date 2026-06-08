# 第4篇 Demo：Tool Use 与 Function Calling

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置 API Key（二选一）
export DEEPSEEK_API_KEY=sk-xxx
# 或
export ZHIPU_API_KEY=xxx

# 3. 本地演示（不需要 API Key）
python tool_registry.py

# 4. 使用 LLM Agent 演示
python tool_registry.py --demo all
python tool_registry.py --demo parallel --provider zhipu
python tool_registry.py --demo chain --provider deepseek
```

## 演示场景

| 场景 | 命令 | 说明 |
|------|------|------|
| 本地演示 | `--demo manual` | 不需要 API Key，直接测试工具注册表 |
| 并行调用 | `--demo parallel` | LLM 同时调用多个工具 |
| 工具链 | `--demo chain` | 工具 A 的输出作为工具 B 的输入 |
| 错误处理 | `--demo error` | 工具调用失败时的优雅处理 |

## 代码结构

```
demo/
├── tool_registry.py     # 主文件：工具注册表 + Agent Loop + Demo
├── requirements.txt     # 依赖
└── README.md           # 本文件
```

## 核心知识点

1. **ToolRegistry**：用装饰器 + JSON Schema 统一管理工具
2. **Function Calling**：LLM 返回 tool_calls → 解析参数 → 执行 → 注入结果
3. **并行调用**：LLM 一次返回多个 tool_call，逐个执行后全部注入
4. **工具链**：LLM 根据第一个工具的结果，决定下一步调用什么
5. **错误处理**：工具执行异常时，将错误信息返回给 LLM 让其自行判断
