"""
Streamlit demo: ask questions about the Hugging Face docs.

  source scripts/llm_env.sh      # sets the LLM_* variables and asks for your key
  streamlit run src/app.py
"""
import json
import os
import time
from pathlib import Path

import streamlit as st

from generate import generate
from retrieve import Retriever
from textutils import extract_citations

REFUSAL = "I don't know based on the provided documents."
DOCS_BASE = "https://huggingface.co/docs"
LIBRARIES = ["all", "transformers", "peft", "datasets", "accelerate", "trl"]
EXAMPLES = [
    "How do I save a processor?",
    "What does max_inflight_tasks control?",
    "How do I run Weights & Biases hyperparameter search?",
    "How do I use groupby in pandas?",
]

st.set_page_config(page_title="Ask the Hugging Face docs", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading models and index (first start only)...")
def load_retriever():
    retriever = Retriever()
    retriever.search("warm up", k=1, mode="hybrid", rerank=True)  # loads the reranker too
    return retriever


@st.cache_data(show_spinner=False, ttl=3600, max_entries=200)
def cached_generate(question: str, passages: list) -> str:
    """Cached so that moving a slider does not call the language model again."""
    return generate(question, passages)


def docs_url(source: str) -> str:
    """transformers/main_classes/model.md -> https://huggingface.co/docs/transformers/main_classes/model"""
    return f"{DOCS_BASE}/{source[:-3] if source.endswith('.md') else source}"


def log_query(record: dict):
    """Optional local logging. Off unless RAG_LOG_QUERIES=1."""
    if os.getenv("RAG_LOG_QUERIES") != "1":
        return
    path = Path("logs/queries.jsonl")
    path.parent.mkdir(exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def set_question(q: str):
    st.session_state["question"] = q


with st.sidebar:
    st.header("Settings")
    library = st.selectbox("Search only in", LIBRARIES)
    k = st.slider("Passages given to the model", 3, 8, 5)
    min_score = st.slider(
        "Refuse if the best passage scores below", -5.0, 5.0, 0.0, 0.5,
        help="Reranker score. If even the best passage scores below this, the app refuses "
             "without calling the language model.",
    )
    st.caption("Index: Hugging Face docs for transformers, peft, datasets, accelerate and trl.")

st.title("📚 Ask the Hugging Face docs")
st.caption(
    "BM25 + embeddings, then a cross-encoder rerank, then a cited answer. Answers come only "
    "from the retrieved passages; if they don't cover the question, the app says so."
)

cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    col.button(example, on_click=set_question, args=(example,))

question = st.text_input("Your question", key="question", placeholder="e.g. How do I save a processor?")

if question.strip():
    question = question.strip()
    retriever = load_retriever()

    t0 = time.time()
    passages = retriever.search(
        question, k=k, mode="hybrid", rerank=True,
        library=None if library == "all" else library,
    )
    t_retrieve = time.time() - t0
    top1 = passages[0]["score"] if passages else -99.0

    gated = (not passages) or top1 < min_score
    t_generate = 0.0
    if gated:
        answer = REFUSAL
    else:
        try:
            t1 = time.time()
            with st.spinner("Asking the language model..."):
                answer = cached_generate(question, passages)
            t_generate = time.time() - t1
        except KeyError as e:
            st.error(f"Missing setting {e}. Set the LLM_* environment variables and restart the app.")
            st.stop()
        except Exception as e:  # network error, rate limit, bad key ...
            st.error(f"The language model call failed: {e}")
            st.stop()

    st.subheader("Answer")
    if gated:
        st.info(
            f"The language model was not called: the best passage scored {top1:.2f}, "
            f"below your threshold of {min_score:.1f}."
        )
    st.markdown(answer)
    st.caption(
        f"Retrieval {t_retrieve * 1000:.0f} ms · generation {t_generate:.1f} s · "
        f"best reranker score {top1:.2f}"
    )

    cited = set(extract_citations(answer))
    st.subheader("Sources")
    if not passages:
        st.write("No passages found.")
    for n, p in enumerate(passages, 1):
        mark = "✅ cited · " if n in cited else ""
        with st.expander(f"[{n}] {mark}{p['source']} — {p['section']}  (score {p['score']:.2f})"):
            st.markdown(
                f"[Open the docs page]({docs_url(p['source'])}) "
                "(link built from the file name; a few pages may not exist online)"
            )
            st.text(p["text"])

    log_query({
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "library": library,
        "gated": gated,
        "refused": "i don't know based on the provided documents" in answer.lower().replace("’", "'"),
        "top1_score": round(top1, 2),
        "sources": [p["source"] for p in passages],
        "retrieval_ms": round(t_retrieve * 1000),
        "generation_s": round(t_generate, 1),
    })