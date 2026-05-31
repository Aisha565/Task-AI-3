BLOCKED_PATTERNS = [
    "ignore previous instructions",
    "reveal system prompt",
    "bypass",
    "hack",
    "jailbreak"
]


def is_safe(query):

    query = query.lower()

    for pattern in BLOCKED_PATTERNS:

        if pattern in query:
            return False

    return True