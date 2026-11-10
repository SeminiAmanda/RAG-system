import json
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parent.parent / "eval"
FILES = sorted(EVAL.glob("questions*.jsonl"))


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_there_are_question_files():
    assert FILES, "no eval/questions*.jsonl files found"


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_question_file_is_well_formed(path):
    questions = load(path)
    assert questions
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids)), "duplicate ids"
    for q in questions:
        assert isinstance(q["question"], str) and q["question"].strip()
        assert isinstance(q["answerable"], bool)
        if q["answerable"]:
            assert isinstance(q.get("source"), str), f"{q['id']}: 'source' must be a string"
            if "sources" in q:
                assert isinstance(q["sources"], list)
                assert all(isinstance(s, str) for s in q["sources"])
        else:
            assert not q.get("source")