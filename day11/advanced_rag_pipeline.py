"""advanced_rag_pipeline.py — 高级RAG Pipeline（智谱真实API版）
包含：Multi-Query查询改写、HyDE假设文档、智谱embedding-3真实向量、智谱rerank精排
用法：python advanced_rag_pipeline.py --provider deepseek
依赖：pip install openai numpy

环境变量：
  ZHIPU_API_KEY     —— 必填，embedding 和 rerank 调用智谱 API（智谱专属能力）
  DEEPSEEK_API_KEY  —— 当 --provider deepseek 时必填，对话/改写用 DeepSeek
说明：--provider 只控制对话用的 LLM；embedding 和 rerank 始终走智谱，因为
      DeepSeek 不提供向量化/重排序接口，这正是国内常见的"DeepSeek对话 + 智谱检索"组合。
"""
import argparse
import json
import os
import urllib.request
import urllib.error

import numpy as np
from openai import OpenAI

# ---------- 配置 ----------
PROVIDERS = {
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "zhipu": {
        "api_key_env": "ZHIPU_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "model": "glm-4.6",
    },
}

ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
EMBED_MODEL = "embedding-3"   # 智谱向量化模型
RERANK_MODEL = "rerank"       # 智谱重排序模型

# ---------- 模拟知识库（8篇技术文档） ----------
DOCUMENTS = [
    {"id": 1, "title": "模型量化技术", "content": "模型量化是将FP32权重转换为INT8或INT4格式，减少模型体积和计算量。常见方法包括训练后量化（PTQ）和量化感知训练（QAT）。GPTQ和AWQ是目前主流的权重量化算法。"},
    {"id": 2, "title": "KV Cache优化", "content": "KV Cache是Transformer推理中的关键优化技术。通过缓存注意力层的Key和Value矩阵，避免重复计算。PagedAttention和FlashAttention分别从内存管理和计算效率两个维度优化了KV Cache。"},
    {"id": 3, "title": "批处理与吞吐量", "content": "批处理是提高推理吞吐量的核心手段。Continuous Batching允许不同长度的请求在同一批次中处理，相比Static Batching显著提高GPU利用率。vLLM和TGI都采用了这一技术。"},
    {"id": 4, "title": "模型蒸馏", "content": "知识蒸馏是将大模型的知识转移到小模型的技术。通过让学生模型模仿教师模型的输出分布，可以在保持大部分能力的前提下将模型体积缩小数倍。"},
    {"id": 5, "title": "推理框架选型", "content": "主流LLM推理框架：vLLM吞吐量高、支持Continuous Batching；TensorRT-LLM延迟低、NVIDIA官方支持；llama.cpp轻量、支持CPU推理。选型需要根据场景权衡吞吐量和延迟。"},
    {"id": 6, "title": "模型训练最佳实践", "content": "大语言模型训练的关键要素：高质量数据清洗（去重、过滤低质量文本）、学习率调度（Warmup+Cosine Decay）、分布式训练策略（ZeRO-3、FSDP）。DeepSpeed和Megatron-LM是主流训练框架。"},
    {"id": 7, "title": "Prompt工程技巧", "content": "编写高质量Prompt的核心原则：指令明确、提供示例（Few-shot）、结构化输出（要求JSON或Markdown格式）。Chain-of-Thought可以引导模型进行推理。"},
    {"id": 8, "title": "RAG系统优化", "content": "RAG系统优化可以从三个方向入手：查询改写（Multi-Query、HyDE、Step-Back）提升检索召回率；重排序提升检索准确率；混合检索兼顾语义和精确匹配。"},
]


# ---------- Embedding：调用智谱 embedding-3 ----------
class Embedder:
    """调用智谱 embedding-3 生成真实向量（带内存缓存，避免重复计费）"""

    def __init__(self, client, model=EMBED_MODEL):
        self.client = client
        self.model = model
        self._cache = {}

    def embed(self, text):
        """文本 → 归一化向量"""
        if text in self._cache:
            return self._cache[text]
        resp = self.client.embeddings.create(model=self.model, input=text)
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        vec = vec / norm if norm > 0 else vec
        self._cache[text] = vec
        return vec


def cosine_sim(a, b):
    """两个归一化向量的余弦相似度 = 点积"""
    return float(np.dot(a, b))


# ---------- 查询改写 ----------
class QueryRewriter:
    """查询改写：Multi-Query + HyDE"""

    def __init__(self, client, model):
        self.client = client
        self.model = model

    def multi_query(self, question, n=3):
        """Multi-Query：生成多个不同角度的子问题"""
        prompt = f"""请从不同角度将以下问题改写为{n}个独立的问题。
要求：
1. 每个问题一行，不要编号
2. 保留原问题的核心意图
3. 从不同侧面表达

原始问题：{question}

改写后的问题："""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        queries = [q.strip() for q in resp.choices[0].message.content.strip().split("\n") if q.strip()]
        return queries[:n]

    def hyde(self, question):
        """HyDE：生成假设答案用于检索"""
        prompt = f"""请根据以下问题，写一段假设性的回答（即使你不确定也要写）。
要求：包含具体的技术术语和细节，200字以内。

问题：{question}

假设回答："""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        return resp.choices[0].message.content.strip()


# ---------- 重排序：调用智谱 rerank API ----------
class ZhipuReranker:
    """调用智谱 rerank 模型，对候选文档按 query 相关性精排。

    rerank 不是 OpenAI 标准接口，智谱走自有的 /paas/v4/rerank，
    所以这里用标准库 urllib 发 POST（不引入 requests，保持依赖最小）。
    """

    def __init__(self, api_key, base_url=ZHIPU_BASE_URL, model=RERANK_MODEL):
        self.api_key = api_key
        self.endpoint = base_url.rstrip("/") + "/rerank"
        self.model = model

    def rerank(self, query, documents, top_k=5):
        """返回 [(relevance_score, doc), ...]，已按相关性降序"""
        # 拼上标题，给 rerank 模型更多上下文
        docs_text = [f"{d['title']}：{d['content']}" for d in documents]
        payload = {
            "model": self.model,
            "query": query,
            "documents": docs_text,
            "top_n": min(top_k, len(docs_text)),
            "return_documents": False,
        }
        data = self._post(payload)
        scored = []
        for item in data.get("results", []):
            idx = item["index"]
            score = item.get("relevance_score", item.get("score", 0.0))
            scored.append((float(score), documents[idx]))
        # 兜底再排一次（接口已排，这里保险）
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def _post(self, payload):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"智谱 rerank 调用失败 [{e.code}]: {body}") from None


# ---------- 基础向量检索（用真实 embedding） ----------
def naive_retrieve(query, documents, embedder, top_k=8):
    """基础向量检索：用 embedding-3 算余弦相似度"""
    query_emb = embedder.embed(query)
    scored = []
    for doc in documents:
        doc_emb = embedder.embed(doc["content"])
        score = max(0, cosine_sim(query_emb, doc_emb))
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


# ---------- 多路检索 + 去重合并 ----------
def multi_route_retrieve(question, sub_queries, hyde_answer, documents, embedder):
    """三路检索（原始问题 + Multi-Query + HyDE），合并去重"""
    all_candidates = []
    seen_ids = set()

    def _add_docs(docs):
        for doc in docs:
            if doc["id"] not in seen_ids:
                all_candidates.append(doc)
                seen_ids.add(doc["id"])

    _add_docs(naive_retrieve(question, documents, embedder, top_k=5))
    for sq in sub_queries:
        _add_docs(naive_retrieve(sq, documents, embedder, top_k=3))
    _add_docs(naive_retrieve(hyde_answer, documents, embedder, top_k=3))

    return all_candidates


# ---------- LLM生成答案 ----------
def generate_answer(question, top_docs, client, model):
    """用检索到的top文档生成答案"""
    context = "\n\n".join(f"【{d['title']}】{d['content']}" for d in top_docs)
    prompt = f"""基于以下参考资料回答问题。如果资料中没有相关信息，请说明。

参考资料：
{context}

问题：{question}

回答："""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


# ---------- 完整Advanced RAG Pipeline ----------
def advanced_rag(question, documents, chat_client, model, embedder, reranker):
    """完整的Advanced RAG Pipeline"""
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"{'='*60}")

    rewriter = QueryRewriter(chat_client, model)

    # Step 1: Multi-Query改写
    print("\n[Step 1] Multi-Query改写:")
    sub_queries = rewriter.multi_query(question, n=3)
    for i, sq in enumerate(sub_queries, 1):
        print(f"  子问题{i}: {sq}")

    # Step 2: HyDE生成假设答案
    print("\n[Step 2] HyDE假设答案:")
    hyde_answer = rewriter.hyde(question)
    print(f"  {hyde_answer[:80]}...")

    # Step 3: 多路检索 + 合并去重（用 embedding-3 真实向量）
    print("\n[Step 3] 多路检索（embedding-3）:")
    all_candidates = multi_route_retrieve(question, sub_queries, hyde_answer, documents, embedder)
    print(f"  候选文档数: {len(all_candidates)} (去重后)")

    # Step 4: 智谱 rerank 精排
    print("\n[Step 4] 智谱 rerank 重排序结果:")
    reranked = reranker.rerank(question, all_candidates, top_k=5)
    for score, doc in reranked:
        print(f"  [{score:.3f}] {doc['title']}")

    # Step 5: 生成答案
    top_docs = [doc for _, doc in reranked[:3]]
    answer = generate_answer(question, top_docs, chat_client, model)
    print(f"\n[最终回答]\n{answer}")
    return answer


# ---------- 对比实验：基础RAG vs 高级RAG ----------
def compare_rag(question, documents, chat_client, model, embedder, reranker):
    """对比基础RAG vs 高级RAG的检索质量"""
    print(f"\n{'#'*60}")
    print(f"对比实验: {question}")
    print(f"{'#'*60}")

    # 基础RAG检索结果（单路向量检索，无改写无重排）
    print("\n--- 基础RAG检索结果（embedding-3 余弦相似度） ---")
    basic_results = naive_retrieve(question, documents, embedder, top_k=5)
    for i, doc in enumerate(basic_results, 1):
        print(f"  {i}. {doc['title']}")

    # 高级RAG完整Pipeline
    print("\n--- 高级RAG检索结果 ---")
    advanced_rag(question, documents, chat_client, model, embedder, reranker)


# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser(description="高级RAG Pipeline Demo（智谱真实API版）")
    parser.add_argument("--provider", choices=["deepseek", "zhipu"], default="deepseek",
                        help="对话用LLM：deepseek 或 zhipu（embedding/rerank 始终用智谱）")
    args = parser.parse_args()

    # embedding + rerank 始终用智谱（国产模型里只有智谱提供这两个接口）
    zhipu_key = os.getenv("ZHIPU_API_KEY")
    if not zhipu_key:
        raise SystemExit("缺少 ZHIPU_API_KEY —— embedding 和 rerank 都依赖智谱 API，请先设置。")

    zhipu_client = OpenAI(api_key=zhipu_key, base_url=ZHIPU_BASE_URL)
    embedder = Embedder(zhipu_client, model=EMBED_MODEL)
    reranker = ZhipuReranker(api_key=zhipu_key)

    # 对话用 --provider 指定的模型
    cfg = PROVIDERS[args.provider]
    chat_key = os.getenv(cfg["api_key_env"])
    if not chat_key:
        raise SystemExit(f"缺少 {cfg['api_key_env']}（--provider {args.provider}）")
    chat_client = OpenAI(api_key=chat_key, base_url=cfg["base_url"])
    model = cfg["model"]
    print(f"对话模型: {model} | 向量化: {EMBED_MODEL} | 重排序: {RERANK_MODEL}")

    # 运行对比实验
    compare_rag("怎么优化模型性能", DOCUMENTS, chat_client, model, embedder, reranker)


if __name__ == "__main__":
    main()
