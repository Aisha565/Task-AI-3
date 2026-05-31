import time
from app.retrieval.hybrid_retriever import HybridRetriever
from app.llm.prompt_builder import build_prompt
from app.llm.gemini_client import generate_answer
from app.query.query_rewriter import rewrite_query
from app.security.guardrails import is_safe


class RAGChatbot:

    def __init__(self):
        self.retriever = HybridRetriever()

    # =========================
    # SMALL TALK HANDLER
    # =========================
    def handle_small_talk(self, query: str):

        query = query.lower().strip()

        responses = {
            "hi": "Hello! How can I help you today?",
            "hello": "Hi! How can I assist you today?",
            "hey": "Hey! What can I help you with?",
            "how are you": "I'm doing great! How can I help you today?",
            "thanks": "You're welcome!",
            "thank you": "Glad to help!",
            "good morning": "Good morning! How can I assist you today?",
            "good evening": "Good evening! How can I help you?"
        }

        return responses.get(query)

    # =========================
    # RELEVANCE SCORING
    # =========================
    def score_doc(self, query: str, doc: str):

        query = query.lower()
        doc = doc.lower()

        score = 0

        # exact match boost
        if query in doc:
            score += 10

        # keyword overlap
        for word in query.split():
            if len(word) <= 2:
                continue
            if word in doc:
                score += 2

        return score

    # =========================
    # MAIN CHAT FUNCTION
    # =========================
    def chat(self, query: str):

        query = query.strip()

        # empty query
        if not query:
            return {
                "answer": "Query is empty.",
                "sources": [],
                "latency": {"retrieval_time": 0, "generation_time": 0, "total_time": 0}
            }

        # small talk first
        small_talk = self.handle_small_talk(query)
        if small_talk:
            return {
                "answer": small_talk,
                "sources": [],
                "latency": {"retrieval_time": 0, "generation_time": 0, "total_time": 0}
            }

        # security check
        if not is_safe(query):
            return {
                "answer": "Unsafe query detected.",
                "sources": []
            }

        # query rewrite
        rewritten_query = rewrite_query(query)

        # retrieval
        start = time.time()
        retrieved_docs = self.retriever.search(rewritten_query)
        retrieval_time = time.time() - start

        # scoring
        scored_docs = [
            (doc, self.score_doc(rewritten_query, doc))
            for doc in retrieved_docs
        ]

        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # filter relevant docs
        filtered_docs = [
            doc for doc, score in scored_docs if score > 0
        ][:3]

        # fallback
        if len(filtered_docs) == 0 or all(len(d.strip()) < 20 for d in filtered_docs):
            filtered_docs = retrieved_docs[:2]

        filtered_docs = filtered_docs[:2]

        # clean sources
        clean_sources = list(dict.fromkeys(
            [doc.strip() for doc in filtered_docs]
        ))

        # prompt
        prompt = build_prompt(rewritten_query, clean_sources)

        if not prompt:
            return {
                "answer": "No prompt generated.",
                "sources": clean_sources,
                "latency": {
                    "retrieval_time": round(retrieval_time, 2),
                    "generation_time": 0,
                    "total_time": round(retrieval_time, 2)
                }
            }

        # generation
        start_gen = time.time()
        answer = generate_answer(prompt)
        generation_time = time.time() - start_gen

        # final response
        return {
            "answer": answer,
            "sources": clean_sources,
            "latency": {
                "retrieval_time": round(retrieval_time, 2),
                "generation_time": round(generation_time, 2),
                "total_time": round(retrieval_time + generation_time, 2)
            }
        }