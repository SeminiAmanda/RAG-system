"""Pure helper functions with no heavy dependencies, so they can be unit-tested quickly."""
import re

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "to", "of", "in", "on", "for", "and", "or",
    "do", "does", "i", "my", "me", "can", "how", "what", "which", "with", "it", "this",
    "that", "be", "by", "as", "at", "from", "if", "so", "you", "your", "we", "should",
}
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def tokenize(text: str):
    """Lowercase word tokens. snake_case identifiers are kept whole AND split into
    parts, so 'weight_sync_steps' matches the identifier and the words in it."""
    tokens = []
    for tok in TOKEN_RE.findall(text):
        low = tok.lower()
        if low in STOPWORDS:
            continue
        tokens.append(low)
        parts = low.split("_")
        if len(parts) > 1:
            tokens.extend(p for p in parts if p and p not in STOPWORDS)
    return tokens


def extract_citations(answer: str):
    """Citation numbers like [1], [2][3] or [1, 3]. Code is stripped first so that
    indexing such as dataset[0] is not mistaken for a citation."""
    text = re.sub(r"```.*?```", " ", answer, flags=re.S)  # fenced code blocks
    text = re.sub(r"`[^`\n]*`", " ", text)                # inline code
    nums = set()
    for group in CITATION_RE.findall(text):
        nums.update(int(n) for n in re.split(r"\s*,\s*", group))
    return sorted(nums)


def rrf(ranked_lists, k: int = 60):
    """Reciprocal Rank Fusion: score = sum over lists of 1 / (k + rank)."""
    fused = {}
    for lst in ranked_lists:
        for rank, (cid, _) in enumerate(lst, 1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def limit_per_source(cands, max_per_source: int):
    """Keep at most max_per_source candidates from any one file, preserving order."""
    counts, out = {}, []
    for c in cands:
        counts[c["source"]] = counts.get(c["source"], 0) + 1
        if counts[c["source"]] <= max_per_source:
            out.append(c)
    return out