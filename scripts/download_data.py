
import subprocess
from pathlib import Path

# Hugging Face repositories and their documentation folders
REPOS = {
    "transformers": "docs/source/en",
    "peft": "docs/source",
    "datasets": "docs/source",
    "accelerate": "docs/source",
    "trl": "docs/source",
}

# Project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL = PROJECT_ROOT / "data" / "external"
RAW = PROJECT_ROOT / "data" / "raw"

# Create folders if they do not exist
EXTERNAL.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)


def run(cmd, cwd=None):
    """Run a terminal command."""
    subprocess.run(cmd, cwd=cwd, check=True)


# STEP 1: Download documentation repositories
print("\nSTEP 1: Downloading Hugging Face documentation...\n")

for name, subdir in REPOS.items():
    dest = EXTERNAL / name

    if dest.exists():
        print(f"{name}: Already downloaded. Skipping clone.")
        continue

    print(f"\nCloning {name}...")

    run([
        "git",
        "clone",
        "--depth", "1",
        "--filter=blob:none",
        "--sparse",
        f"https://github.com/huggingface/{name}.git",
        str(dest),
    ])

    run([
        "git",
        "sparse-checkout",
        "set",
        subdir,
    ], cwd=dest)

    print(f"{name}: Download complete!")


# STEP 2: Copy documentation into data/raw
print("\nSTEP 2: Collecting Markdown documents...\n")

total_count = 0

for name, subdir in REPOS.items():
    src = EXTERNAL / name / subdir

    if not src.exists():
        print(f"Missing documentation folder: {src}")
        continue

    repo_count = 0

    for p in sorted(src.rglob("*")):

        # Only Markdown files
        if p.suffix.lower() not in {".md", ".mdx"}:
            continue

        # Skip repetitive model documentation
        if "model_doc" in p.parts:
            continue

        # Skip missing files or broken symbolic links
        if not p.is_file():
            print(f"Skipping missing file: {p}")
            continue

        # Read file safely
        try:
            text = p.read_text(
                encoding="utf-8",
                errors="ignore"
            )
        except OSError as e:
            print(f"Skipping unreadable file: {p} ({e})")
            continue

        # Skip very short documents
        if len(text) < 500:
            continue

        # Preserve folder structure
        rel = p.relative_to(src).with_suffix("")
        out = RAW / name / f"{rel}.md"

        # Create output folders
        out.parent.mkdir(parents=True, exist_ok=True)

        # Save document
        try:
            out.write_text(text, encoding="utf-8")
        except OSError as e:
            print(f"Could not save file: {out} ({e})")
            continue

        repo_count += 1

    print(f"{name}: {repo_count} files")
    total_count += repo_count


# STEP 3: Print results
print("\n" + "=" * 45)
print("DOWNLOAD AND COLLECTION COMPLETE")
print("=" * 45)
print(f"Total documents: {total_count}")
print(f"Saved in: {RAW}")
print("=" * 45)

if total_count == 0:
    print("WARNING: No documents were collected.")
elif total_count < 200:
    print("NOTE: Fewer than 200 documents collected.")
elif total_count > 500:
    print("NOTE: More than 500 documents collected.")
else:
    print("SUCCESS: Document count is within target range!")

print("\nNext step: Run your ingestion script.")
