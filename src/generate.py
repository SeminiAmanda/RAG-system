# src/generate.py
import os

import requests

SYSTEM_V1 = """You are a technical assistant for Hugging Face library documentation.
Answer the question using ONLY the numbered context passages provided.
Cite the passages you use like [1] or [2][3].
If the context does not contain the answer, reply exactly:
I don't know based on the provided documents.
Be concise. Include code only if it appears in the context."""

SYSTEM_V2 = """You are a technical assistant for Hugging Face library documentation.
Answer the question using ONLY the numbered context passages provided.
Cite the passages you use as [1] or [2][3] at the end of the sentence that uses them.
Every answer must contain at least one citation.
If the passages answer only part of the question, answer that part and say what the passages do not cover.
If the passages contain nothing relevant to the question, reply exactly:
I don't know based on the provided documents.
Do not add claims, recommendations or code that are not in the passages. Be concise."""

SYSTEM = SYSTEM_V1 if os.getenv("PROMPT_VERSION") == "1" else SYSTEM_V2

def build_prompt(question, passages):
    ctx = "\n\n".join(
        f"[{i}] (source: {p['source']} | {p['section']})\n{p['text']}"
        for i, p in enumerate(passages, 1)
    )
    return f"Context:\n{ctx}\n\nQuestion: {question}"


def generate(question, passages, backend=None):
    backend = backend or os.getenv("LLM_BACKEND", "ollama")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_prompt(question, passages)},
    ]

    if backend == "ollama":
        url = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/chat"
        r = requests.post(
            url,
            json={
                "model": os.getenv("LLM_MODEL", "qwen2.5:3b"),
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    # OpenAI-compatible endpoint (Groq, Gemini, GitHub Models, ...)
    r = requests.post(
        os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"},
        json={
            "model": os.environ["LLM_MODEL"],
            "messages": messages,
            "temperature": 0.1,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]