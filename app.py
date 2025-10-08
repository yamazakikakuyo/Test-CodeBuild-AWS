import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, constr
from mangum import Mangum

import string
import contextlib
from collections import defaultdict

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from captum.attr import IntegratedGradients
import nltk

from nltk.corpus import stopwords

try:
    nltk.data.path.insert(0,'/var/task/nltk_data')
    _ = stopwords.words("english")
except Exception:
    nltk.download("stopwords", quiet=True, download_dir='./nltk_data')

app = FastAPI(
    title="MBTI + Integrated Gradients XAI API",
    version="1.0.0",
    description="POST text → get MBTI classification + word attributions (score & percentage) via Integrated Gradients."
)

@app.get("/health")
def health():
    # Don't touch models; purely a liveness probe
    return {"status": "ok"}

handler = Mangum(app)