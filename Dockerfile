FROM public.ecr.aws/lambda/python:3.11

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_PREFER_BINARY=1

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
 && pip install --only-binary=:all: -r requirements.txt

ENV NLTK_DATA=/var/task/nltk_data

RUN python -m nltk.downloader -d $NLTK_DATA punkt stopwords

COPY *.py .

CMD ["app.handler"]

