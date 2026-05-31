import json
from app.chatbot import RAGChatbot


class FallbackEvaluator:

    def __init__(self):

        self.chatbot = RAGChatbot()

    # -------------------------
    # SIMPLE RELEVANCE SCORE
    # -------------------------
    def keyword_score(self, text1, text2):

        t1 = set(text1.lower().split())
        t2 = set(text2.lower().split())

        if not t1 or not t2:
            return 0.0

        overlap = len(t1.intersection(t2))
        return overlap / len(t1)

    # -------------------------
    # MAIN EVALUATION FUNCTION
    # -------------------------
    def evaluate(self, test_questions):

        results = []

        for q in test_questions:

            response = self.chatbot.chat(q)

            answer = response["answer"]
            contexts = " ".join(response["sources"])

            # -------------------------
            # METRICS (CUSTOM RAGAS STYLE)
            # -------------------------

            faithfulness = self.keyword_score(answer, contexts)
            answer_relevancy = self.keyword_score(q, answer)
            context_recall = self.keyword_score(q, contexts)

            results.append({
                "question": q,
                "answer": answer,
                "faithfulness": round(faithfulness, 3),
                "answer_relevancy": round(answer_relevancy, 3),
                "context_recall": round(context_recall, 3)
            })

        return results

    # -------------------------
    # SAVE REPORT
    # -------------------------
    def save_report(self, results, filename="evaluation_report.json"):

        with open(filename, "w") as f:
            json.dump(results, f, indent=4)

        print(f"✅ Report saved to {filename}")