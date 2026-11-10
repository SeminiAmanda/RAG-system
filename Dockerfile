FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    ANONYMIZED_TELEMETRY=False \
    HF_HOME=/home/user/.cache/huggingface \
    INDEX_DIR=/home/user/index \
    CHUNKS_PATH=corpus/chunks.jsonl

# Hugging Face Spaces runs containers as user id 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"
WORKDIR /home/user/app

# CPU-only PyTorch first (the default PyPI wheel is the huge CUDA build)
RUN pip install --user torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt

COPY --chown=user src ./src
COPY --chown=user corpus ./corpus

# Build the vector index and cache the models at build time, so startup is fast
RUN python src/embed.py && \
    python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

EXPOSE 7860
CMD ["streamlit", "run", "src/app.py", "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]