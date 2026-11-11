"""Pure decision logic for the agent (no LangGraph imports, so it is easy to unit-test)."""
from textutils import extract_citations

REFUSAL = "I don't know based on the provided documents."
REFUSAL_PHRASE = "i don't know based on the provided documents"


def is_refusal(answer: str) -> bool:
    return REFUSAL_PHRASE in answer.lower().replace("’", "'")


def check_answer(answer: str, n_passages: int, searches: int):
    """Return a message describing what is wrong with the answer, or None if it is acceptable."""
    if is_refusal(answer):
        return None
    if searches == 0:
        return "You answered without calling search_docs. Call it first and answer only from its results."
    cited = extract_citations(answer)
    if not cited:
        return "Your answer has no citations. Cite the passages you used as [n]."
    bad = [n for n in cited if not 1 <= n <= n_passages]
    if bad:
        return f"These citation numbers do not exist: {bad}. Valid numbers are 1 to {n_passages}."
    return None


def should_refuse(passages, searches: int, min_score: float) -> bool:
    """Refuse when the agent searched but nothing relevant came back."""
    if searches == 0:
        return False
    return not passages or max(p["score"] for p in passages) < min_score