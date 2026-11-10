from textutils import extract_citations, limit_per_source, rrf, tokenize


def test_tokenize_keeps_identifier_and_its_parts():
    toks = tokenize("What is weight_sync_steps?")
    assert "weight_sync_steps" in toks
    assert {"weight", "sync", "steps"} <= set(toks)


def test_tokenize_drops_stopwords_and_lowercases():
    toks = tokenize("How do I use LoRA with PEFT?")
    assert "how" not in toks and "do" not in toks
    assert "lora" in toks and "peft" in toks


def test_tokenize_handles_leading_underscore():
    assert "supports" in tokenize("_supports_assign_param_buffer")


def test_extract_citations_basic_formats():
    assert extract_citations("Use X [1]. Also Y [2][3].") == [1, 2, 3]
    assert extract_citations("Both agree [1, 3].") == [1, 3]
    assert extract_citations("no citations here") == []


def test_extract_citations_ignores_code_indexing():
    text = "Run it:\n```py\nrow = dataset[0]\n```\nand `items[2]` too. [1]"
    assert extract_citations(text) == [1]


def test_extract_citations_flags_zero_in_prose():
    assert extract_citations("see [0]") == [0]


def test_rrf_prefers_items_found_by_both_lists():
    vec = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    kw = [("c", 12.0), ("d", 9.0), ("a", 5.0)]
    fused = [cid for cid, _ in rrf([vec, kw])]
    assert set(fused[:2]) == {"a", "c"}
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_handles_empty_input():
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_limit_per_source_keeps_order_and_caps_each_file():
    cands = [{"source": s, "id": i} for i, s in enumerate(["a", "a", "b", "a", "b"])]
    kept = limit_per_source(cands, 2)
    assert [c["id"] for c in kept] == [0, 1, 2, 4]