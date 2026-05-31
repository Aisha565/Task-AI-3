from app.evaluation.fallback_eval import FallbackEvaluator


test_questions = [
    "What is leave policy?",
    "What is onboarding process?",
    "What are cybersecurity rules?",
    "How is performance evaluated?",
    "What is remote work policy?"
]

evaluator = FallbackEvaluator()

results = evaluator.evaluate(test_questions)

evaluator.save_report(results)

print("\n🔥 EVALUATION RESULTS:\n")
for r in results:
    print(r)