FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs as user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Install CPU-optimized dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application source, tests, and web frontend
COPY --chown=user:user src/ ./src/
COPY --chown=user:user web/ ./web/
COPY --chown=user:user tests/ ./tests/
COPY --chown=user:user run_pipeline.py README.md AGENTS.md ./

# Default Hugging Face Spaces port
EXPOSE 7860

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
