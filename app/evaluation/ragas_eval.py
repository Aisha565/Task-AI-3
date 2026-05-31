from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset

from app.chatbot import RAGChatbot


chatbot = RAGChatbot()


test_questions = [
    "What is leave policy?",
    "What is onboarding process?",
    "What are cybersecurity rules?",
    "How is performance evaluated?",
    "What is remote work policy?"
]


data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": [
        "", "", "", "", ""
    ]
}


# =========================
# RUN CHATBOT ON TEST SET
# =========================

for q in test_questions:

    response = chatbot.chat(q)

    data["question"].append(q)
    data["answer"].append(response["answer"])
    data["contexts"].append(response["sources"])


dataset = Dataset.from_dict(data)


# =========================
# EVALUATION
# =========================

result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_recall
    ]
)


print("\n🔥 RAGAS RESULTS:\n")
print(result)