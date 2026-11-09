import argparse

from generate import generate
from retrieve import Retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--mode", default="hybrid", choices=["vector", "bm25", "hybrid"])
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--library", default=None, help="restrict to one library, e.g. peft")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    question = " ".join(args.question) or input("Question: ")
    passages = Retriever().search(
        question,
        k=args.k,
        mode=args.mode,
        rerank=not args.no_rerank,
        max_per_source=None,
        library=args.library,
    )
    print(generate(question, passages))
    print("\nSources:")
    for i, p in enumerate(passages, 1):
        print(f"  [{i}] {p['source']} | {p['section']}  (score {p['score']:.2f})")


if __name__ == "__main__":
    main()