import ingest


def test_clean_text_removes_license_comment_and_wrapper_tags():
    raw = "<!-- Copyright 2024 -->\n\n# Title\n\n<Tip>\n\nUse this.\n\n</Tip>\n"
    cleaned = ingest.clean_text(raw)
    assert "Copyright" not in cleaned
    assert "<Tip>" not in cleaned and "</Tip>" not in cleaned
    assert "Use this." in cleaned


def test_split_sections_ignores_hash_comments_inside_code_blocks():
    text = "# Title\n\nIntro.\n\n## Install\n\n```bash\n# install the package\npip install x\n```\n"
    paths = [path for path, _ in ingest.split_sections(text, "fallback")]
    assert paths == ["Title", "Title > Install"]


def test_split_sections_builds_heading_paths():
    text = "# A\n\ntext a\n\n## B\n\ntext b\n\n### C\n\ntext c\n\n## D\n\ntext d\n"
    paths = [p for p, _ in ingest.split_sections(text, "x")]
    assert paths == ["A", "A > B", "A > B > C", "A > D"]


def test_chunk_section_keeps_code_blocks_whole():
    code = "```py\n" + "\n".join(f"x{i} = {i}" for i in range(40)) + "\n```"
    body = "Some intro words. " * 30 + "\n\n" + code + "\n\nTail paragraph."
    chunks = ingest.chunk_section(body)
    assert any(code in c for c in chunks)


def test_chunk_section_respects_the_size_ceiling():
    body = "\n\n".join(" ".join(["word"] * 120) for _ in range(10))
    chunks = ingest.chunk_section(body)
    assert len(chunks) > 1
    assert all(len(c.split()) <= ingest.MAX_WORDS + ingest.OVERLAP_MAX_WORDS for c in chunks)