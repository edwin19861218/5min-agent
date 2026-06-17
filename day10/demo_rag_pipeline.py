#!/usr/bin/env python3
"""
RAG Pipeline 完整Demo - 让Agent读你的私有文档
「从0-1成为Agent架构师」系列 · 第10篇

功能：
  - 加载 PDF 和 Markdown 文档
  - 文档分块（RecursiveCharacterTextSplitter）
  - 向量化（智谱 embedding-3）
  - Chroma + SQLite 持久化
  - 混合检索（向量 + BM25）
  - RAG 问答链
  - 交互式提问

用法：
  # 先创建示例文档
  python demo_rag_pipeline.py --init

  # 构建 RAG Pipeline（加载文档 → 分块 → 向量化 → 存储）
  python demo_rag_pipeline.py --build --provider zhipu

  # 交互式问答
  python demo_rag_pipeline.py --ask --provider zhipu

  # 单次提问
  python demo_rag_pipeline.py --query "退货流程是什么？" --provider deepseek

  # 查看系统状态
  python demo_rag_pipeline.py --status

依赖安装：
  pip install langchain langchain-openai langchain-community langchain-text-splitters langchain-chroma chromadb pymupdf rank_bm25
"""

import argparse
import os
import sys
from pathlib import Path

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────

PROVIDERS = {
    "deepseek": {
        "chat_model": "deepseek-chat",
        "api_base": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "zhipu": {
        "chat_model": "glm-4-flash",
        "api_base": "https://open.bigmodel.cn/api/paas/v4/",
        "api_key_env": "ZHIPU_API_KEY",
    },
}

# 智谱 Embedding 配置（两种 Provider 共用智谱 Embedding）
EMBEDDING_CONFIG = {
    "model": "embedding-3",
    "api_base": "https://open.bigmodel.cn/api/paas/v4/",
    "api_key_env": "ZHIPU_API_KEY",
}

CHUNK_SIZE = 600
CHUNK_OVERLAP = 150
PERSIST_DIR = "./rag_db"
SAMPLE_DOCS_DIR = "./sample_docs"


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def get_api_key(env_var: str) -> str:
    key = os.environ.get(env_var, "")
    if not key:
        print(f"错误：请设置环境变量 {env_var}")
        print(f"  export {env_var}='your-api-key-here'")
        sys.exit(1)
    return key


def get_llm(provider: str):
    from langchain_openai import ChatOpenAI

    cfg = PROVIDERS[provider]
    return ChatOpenAI(
        model=cfg["chat_model"],
        openai_api_key=get_api_key(cfg["api_key_env"]),
        openai_api_base=cfg["api_base"],
        temperature=0.3,
    )


def get_embeddings():
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=EMBEDDING_CONFIG["model"],
        openai_api_key=get_api_key(EMBEDDING_CONFIG["api_key_env"]),
        openai_api_base=EMBEDDING_CONFIG["api_base"],
    )


# ──────────────────────────────────────────────
# 混合检索器（加权 RRF，替代 langchain 1.x 已移除的 EnsembleRetriever）
# ──────────────────────────────────────────────

class HybridRetriever:
    """加权倒数秩融合（Reciprocal Rank Fusion）检索器。

    langchain 1.x 移除了 EnsembleRetriever，这里用 RRF 复刻其核心能力：
    对每个子检索器返回的结果按排名累加加权得分，再按总分降序输出。

        score(d) = Σ weight_i / (rrf_k + rank_i(d))

    子检索器只需实现 ``.invoke(query) -> list[Document]``
    （Chroma 的 as_retriever() 和 BM25Retriever 都满足）。
    """

    def __init__(self, retrievers, weights, rrf_k=60):
        if len(retrievers) != len(weights):
            raise ValueError("retrievers 和 weights 长度必须一致")
        self.retrievers = retrievers
        self.weights = weights
        self.rrf_k = rrf_k

    def invoke(self, query):
        scores = {}  # page_content -> {"doc": Document, "score": float}
        for retriever, weight in zip(self.retrievers, self.weights):
            for rank, doc in enumerate(retriever.invoke(query)):
                key = doc.page_content
                if key not in scores:
                    scores[key] = {"doc": doc, "score": 0.0}
                scores[key]["score"] += weight / (self.rrf_k + rank + 1)
        ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in ranked]

    # 兼容旧式检索器 API
    def get_relevant_documents(self, query):
        return self.invoke(query)


# ──────────────────────────────────────────────
# 创建示例文档
# ──────────────────────────────────────────────

def create_sample_docs():
    docs_dir = Path(SAMPLE_DOCS_DIR)
    docs_dir.mkdir(exist_ok=True)

    # 示例 Markdown 文档
    faq_md = docs_dir / "faq.md"
    if not faq_md.exists():
        faq_md.write_text("""# 产品常见问题 FAQ

## 退货政策

### 退货条件
1. 商品签收后7天内可申请退货
2. 商品需保持原包装完好，未拆封使用
3. 定制类商品不支持退货
4. 食品、内衣等特殊商品不支持退货

### 退货流程
1. 打开APP → 我的订单 → 选择要退货的订单
2. 点击"申请退货"，选择退货原因
3. 上传商品照片（可选）
4. 等待商家审核（1-2个工作日）
5. 审核通过后，系统分配快递员上门取件
6. 商家收到退货后3个工作日内退款

### 退款方式
- 原路退回：退款至原支付方式，3-5个工作日到账
- 余额退款：立即到账，可用于下次消费

## 客服联系方式

- 客服热线：400-8888-9999
- 工作时间：每天 9:00 - 21:00
- 在线客服：APP内 → 我的 → 在线客服
- 邮箱：support@example.com

## 配送说明

### 配送范围
全国范围配送（偏远地区可能额外收取运费）。

### 配送时间
- 一线城市：下单后1-2天送达
- 二三线城市：下单后2-4天送达
- 偏远地区：下单后5-7天送达

### 配送费用
- 订单满99元免运费
- 未满99元收取8元运费
- 偏远地区加收15元偏远附加费

## 会员体系

### 会员等级
| 等级 | 条件 | 权益 |
|------|------|------|
| 普通会员 | 注册即得 | 基础权益 |
| 银卡会员 | 年消费满500元 | 95折 + 生日礼包 |
| 金卡会员 | 年消费满2000元 | 9折 + 专属客服 + 免运费 |
| 钻石会员 | 年消费满5000元 | 85折 + 所有权益 + 新品优先购 |

### 积分规则
- 每消费1元积1分
- 评价订单额外获得10分
- 分享商品获得5分
- 积分可抵扣现金，100积分=1元
""", encoding="utf-8")
        print(f"  创建示例文档: {faq_md}")

    # 示例文本文件（模拟产品手册片段）
    manual_txt = docs_dir / "product_manual.txt"
    if not manual_txt.exists():
        manual_txt.write_text("""产品使用手册 - 智能空气净化器 AP-2000

第一章 产品概述

AP-2000智能空气净化器是一款面向家庭和办公场景的高效空气净化设备。
采用HEPA H13级滤芯，CADR值达到500m³/h，适用面积30-60平方米。

核心参数：
- 颗粒物CADR：500m³/h
- 甲醛CADR：200m³/h
- 噪音范围：28-55dB
- 额定功率：45W
- 滤芯寿命：6-12个月（视使用环境而定）

第二章 使用指南

2.1 首次使用
1. 拆除包装，取出净化器和滤芯
2. 打开背板，取出滤芯并去除塑料密封袋
3. 将滤芯装回，关闭背板
4. 插上电源，长按开机键3秒启动
5. 首次使用建议先运行30分钟再进入房间

2.2 日常操作
- 自动模式：根据空气质量自动调节风速，推荐日常使用
- 睡眠模式：最低噪音运行，适合夜间使用
- 强力模式：快速净化，适合刚回家或空气质量较差时使用
- 定时功能：支持1/2/4/8小时定时关机

2.3 滤芯更换
当滤芯指示灯亮红灯时，需要更换滤芯。
更换步骤：
1. 关机并拔掉电源
2. 打开背板取出旧滤芯
3. 拆除新滤芯的密封袋
4. 装入新滤芯，关闭背板
5. 长按复位键5秒重置滤芯寿命

第三章 常见问题

Q: 净化器显示红灯但空气质量正常？
A: 可能是滤芯需要更换。请检查滤芯指示灯。

Q: 运行时有异味？
A: 新滤芯首次使用可能有轻微气味，属于正常现象，运行1-2天后消失。

Q: 可以24小时开机吗？
A: 可以。AP-2000支持24小时连续运行，自动模式下功耗仅约20W。

Q: 滤芯在哪里购买？
A: 官方商城、授权经销商或APP内直接购买。原装滤芯型号：FC-2000。
""", encoding="utf-8")
        print(f"  创建示例文档: {manual_txt}")

    print("示例文档创建完成！")


# ──────────────────────────────────────────────
# 加载文档
# ──────────────────────────────────────────────

def load_documents(docs_dir: str):
    from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
    from langchain_core.documents import Document

    all_docs = []
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        print(f"错误：文档目录不存在 {docs_dir}")
        print("请先运行 --init 创建示例文档")
        sys.exit(1)

    for f in docs_path.iterdir():
        if f.suffix == ".md":
            try:
                loader = UnstructuredMarkdownLoader(str(f))
                docs = loader.load()
                all_docs.extend(docs)
                print(f"  加载 Markdown: {f.name} ({len(docs)} 段)")
            except ImportError:
                # 如果没有 unstructured，用 TextLoader 代替
                loader = TextLoader(str(f), encoding="utf-8")
                docs = loader.load()
                all_docs.extend(docs)
                print(f"  加载 Markdown (文本模式): {f.name} ({len(docs)} 段)")
        elif f.suffix == ".txt":
            loader = TextLoader(str(f), encoding="utf-8")
            docs = loader.load()
            all_docs.extend(docs)
            print(f"  加载文本: {f.name} ({len(docs)} 段)")
        elif f.suffix == ".pdf":
            try:
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(str(f))
                docs = loader.load()
                all_docs.extend(docs)
                print(f"  加载 PDF: {f.name} ({len(docs)} 页)")
            except ImportError:
                print(f"  跳过 PDF（需要 pymupdf）: {f.name}")

    if not all_docs:
        print("错误：未加载到任何文档")
        sys.exit(1)

    return all_docs


# ──────────────────────────────────────────────
# 分块
# ──────────────────────────────────────────────

def split_documents(docs):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    )

    chunks = splitter.split_documents(docs)
    print(f"  分块完成：{len(docs)} 段文档 → {len(chunks)} 个文本块")
    print(f"  chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")
    return chunks


# ──────────────────────────────────────────────
# 构建 Pipeline
# ──────────────────────────────────────────────

def build_pipeline(docs_dir: str, provider: str):
    from langchain_chroma import Chroma

    print("\n=== 构建 RAG Pipeline ===\n")

    # 1. 加载文档
    print("[1/4] 加载文档...")
    docs = load_documents(docs_dir)

    # 2. 分块
    print("\n[2/4] 文档分块...")
    chunks = split_documents(docs)

    # 3. 向量化 + 存储
    print("\n[3/4] 向量化并存储（调用 Embedding API，请稍候）...")
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="rag_docs",
    )
    # langchain-chroma 在指定 persist_directory 后会自动持久化，无需显式 persist()
    print(f"  向量存储完成，持久化到 {PERSIST_DIR}")

    # 4. 构建检索器
    print("\n[4/4] 构建混合检索器...")
    from langchain_community.retrievers import BM25Retriever

    bm25_retriever = BM25Retriever.from_documents(chunks, k=3)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    ensemble_retriever = HybridRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )
    print("  混合检索器就绪（向量 60% + BM25 40%）")

    print("\n=== Pipeline 构建完成 ===\n")
    return vectorstore, ensemble_retriever, chunks


# ──────────────────────────────────────────────
# 问答
# ──────────────────────────────────────────────

def ask_question(query: str, provider: str):
    from langchain_chroma import Chroma
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    print(f"\n提问：{query}")
    print("正在检索和生成答案...\n")

    # 加载已有的向量存储
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="rag_docs",
    )

    # 获取文档块用于 BM25
    all_docs = vectorstore.get(include=["documents"])
    chunks = [
        Document(page_content=text, metadata={"source": "stored"})
        for text in all_docs["documents"]
    ]

    bm25_retriever = BM25Retriever.from_documents(chunks, k=3)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retriever = HybridRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )

    # 先检索来源文档，再喂给 LCEL 链生成答案
    source_docs = retriever.invoke(query)

    # 纯 langchain_core LCEL 链（langchain 1.x 已移除 RetrievalQA）
    system_prompt = (
        "你是一个严谨的问答助手。请只根据下面提供的「上下文」回答用户问题。"
        "如果上下文中没有相关信息，请直接回答「我不知道」，不要编造。\n\n"
        "上下文：\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    llm = get_llm(provider)
    rag_chain = (
        {
            "context": lambda _: "\n\n".join(d.page_content for d in source_docs),
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = rag_chain.invoke(query)

    print(f"回答：\n{answer}\n")
    print("--- 来源文档片段 ---")
    for i, doc in enumerate(source_docs[:3], 1):
        content = doc.page_content[:150].replace("\n", " ")
        source = doc.metadata.get("source", "unknown")
        print(f"  [{i}] ({source}) {content}...")
    print()


def interactive_mode(provider: str):
    print("\n=== RAG 问答（交互模式）===")
    print(f"模型：{PROVIDERS[provider]['chat_model']}")
    print("输入问题后回车，输入 'quit' 或 'exit' 退出\n")

    while True:
        try:
            query = input("你的问题 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("退出。")
            break

        ask_question(query, provider)


def show_status():
    import chromadb

    print("\n=== RAG Pipeline 状态 ===\n")

    if not Path(PERSIST_DIR).exists():
        print("尚未构建 Pipeline。请先运行 --build")
        return

    client = chromadb.PersistentClient(path=PERSIST_DIR)
    try:
        collection = client.get_collection("rag_docs")
        count = collection.count()
        print(f"向量数据库：{PERSIST_DIR}")
        print(f"存储的文档块数：{count}")
        print(f"分块参数：chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")
        print(f"Embedding 模型：{EMBEDDING_CONFIG['model']}")
    except Exception as e:
        print(f"读取状态失败：{e}")

    print()


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RAG Pipeline Demo - 让Agent读你的私有文档",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python demo_rag_pipeline.py --init
  python demo_rag_pipeline.py --build --provider zhipu
  python demo_rag_pipeline.py --ask --provider zhipu
  python demo_rag_pipeline.py --query "退货流程是什么？" --provider deepseek
        """,
    )

    parser.add_argument("--init", action="store_true", help="创建示例文档")
    parser.add_argument("--build", action="store_true", help="构建 RAG Pipeline")
    parser.add_argument("--ask", action="store_true", help="交互式问答模式")
    parser.add_argument("--query", type=str, help="单次提问")
    parser.add_argument("--status", action="store_true", help="查看 Pipeline 状态")
    parser.add_argument(
        "--provider",
        choices=["deepseek", "zhipu"],
        default="zhipu",
        help="LLM 提供方（默认 zhipu）",
    )
    parser.add_argument(
        "--docs-dir",
        default=SAMPLE_DOCS_DIR,
        help="文档目录路径（默认 ./sample_docs）",
    )

    args = parser.parse_args()

    if args.init:
        create_sample_docs()
    elif args.build:
        build_pipeline(args.docs_dir, args.provider)
    elif args.ask:
        interactive_mode(args.provider)
    elif args.query:
        ask_question(args.query, args.provider)
    elif args.status:
        show_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
