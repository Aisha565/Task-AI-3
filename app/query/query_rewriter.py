from app.llm.gemini_client import generate_answer


def rewrite_query(query):

    prompt = f"""
Rewrite the following query for better document retrieval.

Query:
{query}

Improved Query:
"""

    improved = generate_answer(prompt)

    return improved.strip()