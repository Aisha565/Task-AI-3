from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self):

        # Cross encoder model (light + fast)
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(self, query, docs, top_k=5):

        if not docs:
            return []

        # Create query-doc pairs
        pairs = [
            (query, doc)
            for doc in docs
        ]

        # Predict relevance scores
        scores = self.model.predict(pairs)

        # Combine docs with scores
        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Keep top_k only
        ranked = ranked[:top_k]

        # Return ONLY content (clean output for chatbot)
        return [
            doc for doc, score in ranked
        ]