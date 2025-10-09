FROM public.ecr.aws/lambda/python:3.11

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_PREFER_BINARY=1 \
    HF_HUB_DISABLE_TELEMETRY=1

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
 && pip install --only-binary=:all: -r requirements.txt

ENV NLTK_DATA=/var/task/nltk_data
ENV MODEL_DIR=/var/task/models

RUN python -m nltk.downloader -d $NLTK_DATA punkt stopwords

ARG HF_TOKEN
ENV ENV_HF_TOKEN=${HF_TOKEN}

ARG USERNAME
ENV ENV_USERNAME=${USERNAME}

# Create a fixed folder for the models
RUN mkdir -p "$MODEL_DIR"

RUN python - <<'PY'
from huggingface_hub import snapshot_download
import os
hf_token = os.environ.get("ENV_HF_TOKEN")
username = os.environ.get("ENV_USERNAME", 'yamazakikakuyo')
repos = [
    f"{username}/MBTI-bert-base-uncased-energy",
    f"{username}/MBTI-bert-base-uncased-information",
    f"{username}/MBTI-bert-base-uncased-decision",
    f"{username}/MBTI-bert-base-uncased-execution",
]
dest_root = "/var/task/models"

snapshot_download(repo_id="bert-base-uncased", local_dir=os.path.join(dest_root, "bert-base-uncased"), repo_type="model")

for rid in repos:
    local_dir = os.path.join(dest_root, rid.split("/")[-1])
    snapshot_download(
        repo_id=rid,
        local_dir=local_dir,
        token=hf_token,
        revision=None,        # or pin a commit/tag for reproducibility
        allow_patterns=None,  # or restrict to model files
    )
    print(local_dir)
    print(os.listdir(local_dir))
PY

RUN set -euo pipefail \
 && echo "== Listing models in $MODEL_DIR ==" \
 && find "$MODEL_DIR" -mindepth 1 -maxdepth 2 -type d | sed "s|$MODEL_DIR/||" | sort || true \
 && echo "== Sizes ==" \
 && du -sh "$MODEL_DIR"/* 2>/dev/null || true


ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

COPY *.py .

CMD ["app.handler"]


