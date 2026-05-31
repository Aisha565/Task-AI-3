import pickle
from rank_bm25 import BM25Okapi

BM25_PATH = "vectorstore/bm25.pkl"


class SparseRetriever:

    def __init__(self):

        with open(BM25_PATH, "rb") as f:
            self.bm25, self.texts = pickle.load(f)

    def search(self, query, k=5):

        tokenized_query = query.split()

        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            list(enumerate(scores)),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        results = []

        for idx, score in ranked:

            results.append({
                "content": self.texts[idx],
                "score": float(score),
                "source": "sparse"
            })

        return results