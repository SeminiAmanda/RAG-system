"""
Ingest: read cleaned Markdown docs from data/raw, split by headings,
chunk by size, and write data/processed/chunks.jsonl.
"""
import json
import re
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/chunks.jsonl")

TARGET_WORDS = 200      # aim for chunks of about this many words
MAX_WORDS = 350         # hard ceiling for any single piece
MIN_WORDS = 30          # drop chunks smaller than this
OVERLAP_MAX_WORDS = 60  # carry the last paragraph into the next chunk if it is this short

HF_TAGS = (
    r"Tip|tip|frameworkcontent|pt|tf|jax|hfoptions|hfoption|Youtube|"
    r"EditOnGithub|Deprecated|Warning|Note|Question|Answer|"
    r"literalinclude|iframe|script|div|span"
)


# ----------------------------------------------------------------------
# 1. Cleaning
# ----------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)           # HTML comments / license header
    text = re.sub(r"\[\[open-in-colab\]\]", "", text)
    text = re.sub(r"(?m)^\[\[autodoc\]\].*$", "", text)           # API autodoc markers
    text = re.sub(r"(?m)^\s*(?:-\s*)?(?:members|inherited-members|exclude-members):.*$", "", text)
    text = re.sub(rf"</?(?:{HF_TAGS})\b[^>]*>", "", text)         # wrapper tags, keep inner text
    text = re.sub(r"(?m)^---\n.*?\n---\n", "", text, count=1, flags=re.S)  # YAML front matter
    text = re.sub(r"[ \t]+\n", "\n", text)                        # trailing spaces
    text = re.sub(r"\n{3,}", "\n\n", text)                        # extra blank lines
    return text.strip()


# ----------------------------------------------------------------------
# 2. Split into sections by heading (ignoring '#' inside code blocks)
# ----------------------------------------------------------------------
HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")


def split_sections(text: str, doc_title_fallback: str):
    """Return a list of (heading_path, body) tuples."""
    sections = []
    stack = []            # list of (level, title)
    body_lines = []
    in_code = False

    def flush():
        body = "\n".join(body_lines).strip()
        if body:
            path = " > ".join(t for _, t in stack) or doc_title_fallback
            sections.append((path, body))

    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_code = not in_code
            body_lines.append(line)
            continue
        m = None if in_code else HEADING_RE.match(line)
        if m:
            flush()
            body_lines = []
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            body_lines.append(line)
    flush()
    return sections


# ----------------------------------------------------------------------
# 3. Chunk a section
# ----------------------------------------------------------------------
def word_count(s: str) -> int:
    return len(s.split())


def hard_split(text: str, max_words: int):
    """Split an oversized block on line boundaries, then on words if needed."""
    pieces, current, count = [], [], 0
    for line in text.split("\n"):
        n = word_count(line)
        if n > max_words:  # one gigantic line: split on words
            if current:
                pieces.append("\n".join(current))
                current, count = [], 0
            words = line.split()
            for i in range(0, len(words), max_words):
                pieces.append(" ".join(words[i:i + max_words]))
            continue
        if current and count + n > max_words:
            pieces.append("\n".join(current))
            current, count = [], 0
        current.append(line)
        count += n
    if current:
        pieces.append("\n".join(current))
    return pieces


def to_units(body: str):
    """Break a section into units: whole code blocks and paragraphs.
    Returns a list of (text, is_code)."""
    units = []
    for block in re.split(r"(```.*?```)", body, flags=re.S):
        if not block.strip():
            continue
        if block.startswith("```"):
            if word_count(block) > MAX_WORDS:
                units.extend((p, True) for p in hard_split(block, MAX_WORDS))
            else:
                units.append((block.strip(), True))
        else:
            for para in re.split(r"\n\s*\n", block):
                para = para.strip()
                if not para:
                    continue
                if word_count(para) > MAX_WORDS:
                    units.extend((p, False) for p in hard_split(para, MAX_WORDS))
                else:
                    units.append((para, False))
    return units


def chunk_section(body: str):
    units = to_units(body)
    chunks, current, count = [], [], 0

    for text, is_code in units:
        n = word_count(text)
        if current and count + n > TARGET_WORDS:
            chunks.append("\n\n".join(t for t, _ in current))
            # overlap: carry over the last unit if it is a short, non-code paragraph
            last_text, last_is_code = current[-1]
            if not last_is_code and word_count(last_text) <= OVERLAP_MAX_WORDS:
                current, count = [(last_text, False)], word_count(last_text)
            else:
                current, count = [], 0
        current.append((text, is_code))
        count += n

    if current:
        chunks.append("\n\n".join(t for t, _ in current))
    return chunks


# ----------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------
def ingest():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in RAW_DIR.rglob("*.md"))
    if not files:
        raise SystemExit(f"No .md files found in {RAW_DIR.resolve()}")

    total_chunks, dropped = 0, 0
    sizes = []

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for path in files:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            text = clean_text(raw)
            if not text:
                continue

            rel = path.relative_to(RAW_DIR).as_posix()   # e.g. peft/conceptual_guides/lora.md
            library = rel.split("/")[0]
            doc_title = path.stem.replace("_", " ").replace("-", " ")

            for heading_path, body in split_sections(text, doc_title):
                for chunk in chunk_section(body):
                    n = word_count(chunk)
                    if n < MIN_WORDS:
                        dropped += 1
                        continue
                    record = {
                        "id": f"{rel}#{total_chunks}",
                        "source": rel,
                        "library": library,
                        "section": heading_path,
                        "text": chunk,
                        "n_words": n,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    sizes.append(n)
                    total_chunks += 1

    print(f"Files processed : {len(files)}")
    print(f"Chunks written  : {total_chunks}")
    print(f"Tiny chunks dropped: {dropped}")
    if sizes:
        sizes.sort()
        print(
            f"Words per chunk : min={sizes[0]}  median={sizes[len(sizes)//2]}  "
            f"max={sizes[-1]}"
        )
    print(f"Output          : {OUT_PATH.resolve()}")


if __name__ == "__main__":
    ingest()