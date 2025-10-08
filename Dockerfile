FROM public.ecr.aws/lambda/python:3.11

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_PREFER_BINARY=1 \
    HF_HOME=/opt/hf-home \
    TRANSFORMERS_CACHE=/opt/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
 && pip install --only-binary=:all: -r requirements.txt

ENV NLTK_DATA=/var/task/nltk_data

RUN python -m nltk.downloader -d $NLTK_DATA punkt stopwords

# ARG HF_TOKEN
# ENV HF_TOKEN=${HF_TOKEN}

# ARG USERNAME
# ENV USERNAME=${USERNAME}

# # Create a fixed folder for the models
# RUN mkdir -p /opt/models

# RUN python - <<'PY'
# from huggingface_hub import snapshot_download
# import os
# print(os.environ.get("HF_TOKEN"))
# hf_token = os.environ.get("HF_TOKEN")
# username = os.environ.get("USERNAME", 'yamazakikakuyo')
# repos = [
#     f"{username}/MBTI-bert-base-uncased-energy",
#     f"{username}/MBTI-bert-base-uncased-information",
#     f"{username}/MBTI-bert-base-uncased-decision",
#     f"{username}/MBTI-bert-base-uncased-execution",
# ]
# dest_root = "/opt/models"
# for rid in repos:
#     local_dir = os.path.join(dest_root, rid.split("/")[-1])
#     snapshot_download(
#         repo_id=rid,
#         local_dir=local_dir,
#         local_dir_use_symlinks=False,
#         token=hf_token,
#         revision=None,        # or pin a commit/tag for reproducibility
#         allow_patterns=None,  # or restrict to model files
#     )
# PY

# ENV TRANSFORMERS_OFFLINE=1

COPY *.py .

CMD ["app.handler"]

