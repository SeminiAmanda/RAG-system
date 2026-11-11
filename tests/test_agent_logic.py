from agent_logic import REFUSAL, check_answer, is_refusal, should_refuse


def test_refusal_is_accepted_without_searching():
    assert check_answer(REFUSAL, n_passages=0, searches=0) is None


def test_refusal_detection_handles_curly_apostrophes():
    assert is_refusal("I don’t know based on the provided documents")


def test_answer_without_a_search_is_flagged():
    assert "search_docs" in check_answer("Use save_pretrained [1].", n_passages=0, searches=0)


def test_answer_without_citation_is_flagged():
    assert "no citations" in check_answer("Use save_pretrained.", n_passages=3, searches=1)


def test_invalid_citation_number_is_flagged():
    msg = check_answer("Use save_pretrained [5].", n_passages=3, searches=1)
    assert "5" in msg and "1 to 3" in msg


def test_valid_answer_passes_and_code_indexing_is_ignored():
    answer = "Use it:\n```py\nx = data[7]\n```\nSee [1][2]."
    assert check_answer(answer, n_passages=3, searches=2) is None


def test_gate_only_applies_after_a_search():
    assert not should_refuse([], searches=0, min_score=0.0)
    assert should_refuse([], searches=1, min_score=0.0)
    assert should_refuse([{"score": -1.0}, {"score": -3.0}], searches=2, min_score=0.0)
    assert not should_refuse([{"score": -1.0}, {"score": 2.5}], searches=2, min_score=0.0)