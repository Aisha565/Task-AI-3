from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.sparse_retriever import SparseRetriever
from app.retrieval.reranker import Reranker


class HybridRetriever:

    def __init__(self):

        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()
        self.reranker = Reranker()

    # =========================
    # RRF FUSION (IMPROVED)
    # =========================
    def rrf(self, dense_results, sparse_results, k=60):

        scores = {}

        # Dense results
        for rank, r in enumerate(dense_results):

            key = r.get("content", "").strip()

            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)

        # Sparse results
        for rank, r in enumerate(sparse_results):

            key = r.get("content", "").strip()

            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)

        fused = [
            {
                "content": content,
                "score": score
            }
            for content, score in scores.items()
        ]

        return fused

    # =========================
    # SEARCH PIPELINE
    # =========================
    def search(self, query, top_k=3):

        # 1. Dense retrieval
        dense_results = self.dense.search(query, k=5)

        # 2. Sparse retrieval
        sparse_results = self.sparse.search(query, k=5)

        # 3. RRF fusion
        fused = self.rrf(dense_results, sparse_results)

        query_terms = set(query.lower().split())

        filtered = []

        # =========================
        # STRONGER RELEVANCE FILTER
        # =========================
        for r in fused:

            content = r["content"].lower()
            content_words = set(content.split())

            overlap = len(query_terms.intersection(content_words))

            # 🔥 stricter threshold
            if overlap >= 2 or query.lower() in content:

                r["score"] += overlap * 0.5
                filtered.append(r)

        # fallback safety
        if not filtered:
            filtered = fused[:top_k]

        # sort
        filtered = sorted(
            filtered,
            key=lambda x: x["score"],
            reverse=True
        )

        # =========================
        # RERANKING STEP
        # =========================
        docs = [r["content"] for r in filtered]

        reranked_docs = self.reranker.rerank(
            query,
            docs
        )

        return reranked_docs[:top_k]