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

class MBTIPipeline:
    def __init__(self, model_choice="bert-base-uncased", user="yamazakikakuyo", use_gpu=False):
        self.categories = ["energy", "information", "decision", "execution"]
        self.user = user
        self.model_choice = model_choice

        self.MODEL_ROOT = "/opt/models"

        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        hf_device = 0 if self.device.type == "cuda" else -1

        for category in self.categories:
            # repo_id = f"{self.user}/MBTI-{self.model_choice}-{category}"
            folder_name = f"MBTI-bert-base-uncased-{category}"
            model_dir = os.path.join(self.MODEL_ROOT, folder_name)
            tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            model_category = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
            model_category.to(self.device).eval()
            self.tokenizers[category] = tokenizer
            self.models[category] = model_category
            self.pipelines[category] = pipeline(
                "text-classification", model=model_category, tokenizer=tokenizer, device=hf_device
            )

    def post_process(self, category, output):
        pred = output[0]
        label, score = pred["label"], pred["score"]

        mappings = {
            "energy": {"LABEL_0": "E", "LABEL_1": "I"},       # { 0:"extraversion"  1:"introversion"}
            "information": {"LABEL_0": "S", "LABEL_1": "N"},  # { 0:"sensing"       1:"intuition"   }
            "decision": {"LABEL_0": "T", "LABEL_1": "F"},     # { 0:"thinking"      1:"feeling"     }
            "execution": {"LABEL_0": "J", "LABEL_1": "P"},    # { 0:"judging"       1:"perceiving"  }
        }

        letter = mappings[category].get(label, label)
        return {"category": category, "mbti_letter": letter, "confidence": round(score, 3)}

    def predict_mbti_only(self, text):
        results = []
        for category, clf in self.pipelines.items():
            raw = clf(text)
            processed = self.post_process(category, raw)
            results.append(processed)

        mbti_type = "".join([r["mbti_letter"] for r in results])
        return {"mbti_type": mbti_type, "details": results}
    
    def aggregate_tokens(self, cleaned):
        agg = defaultdict(float)
        for item in cleaned:
            word = item["word"]
            score = item["score"]
            agg[word] += score

        total_abs = sum(abs(s) for s in agg.values()) or 1.0

        aggregated = [
            {
                "word": w,
                "score": float(s),
                "percentage": float(abs(s) / total_abs)
            }
            for w, s in agg.items()
        ]

        return aggregated

    def clean_tokens(self, tokens, scores, remove_stopwords=True):
        merged = []
        current_word, current_score = "", 0.0

        for tok, score in zip(tokens, scores):
            if tok in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            if tok.startswith("##"):
                current_word += tok[2:]
                current_score += float(score)
            else:
                if current_word:
                    merged.append({"word": current_word, "score": current_score})
                current_word = tok
                current_score = float(score)

        if current_word:
            merged.append({"word": current_word, "score": current_score})

        fixed = []
        skip = False
        for i, w in enumerate(merged):
            if skip:
                skip = False
                continue

            if i+2 < len(merged) and merged[i+1]["word"] == "'" and merged[i+2]["word"] in ["re","s","m","ll","ve","d","t"]:
                new_word = w["word"] + "'" + merged[i+2]["word"]
                new_score = w["score"] + merged[i+1]["score"] + merged[i+2]["score"]
                fixed.append({"word": new_word, "score": new_score})
                skip = True
            else:
                fixed.append(w)

        sw = set(stopwords.words("english")) if remove_stopwords else set()
        cleaned = [
            w for w in fixed
            if not all(ch in string.punctuation for ch in w["word"])
            and (not remove_stopwords or w["word"].lower() not in sw)
        ]
        return cleaned

    def _ig_raw(self, text, category, n_steps=64, method="gausslegendre"):
        model = self.models[category]
        tok = self.tokenizers[category]

        # Tokenize on device
        inputs = tok(text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
        am = inputs["attention_mask"]

        with torch.no_grad():
            logits = model(**inputs).logits
            pred_class = torch.argmax(logits, dim=1).item()

        # Build differentiable embeddings
        embed = model.get_input_embeddings()
        emb = embed(input_ids).detach().clone().requires_grad_(True)
        baseline = torch.zeros_like(emb)

        ig = IntegratedGradients(lambda e: model(inputs_embeds=e, attention_mask=am).logits)
        use_amp = (self.device.type == "cuda")
        ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else contextlib.nullcontext()
        with ctx:
            attrs, delta = ig.attribute(
                emb,
                baselines=baseline,
                target=pred_class,
                n_steps=int(n_steps),
                method=method,
                internal_batch_size=min(64, int(n_steps)),
                return_convergence_delta=True
            )

        token_scores = attrs.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        tokens = tok.convert_ids_to_tokens(input_ids[0].tolist())
        # delta may be tensor or tensor-like; coerce to float
        try:
            delta_val = float(delta.mean().item())
        except Exception:
            delta_val = float(delta)
        return tokens, token_scores, delta_val, pred_class

    def report_ig(self, text, category, n_steps=120, top_k=10, method="gausslegendre"):
        tokens, token_scores, delta, _ = self._ig_raw(text, category, n_steps=n_steps, method=method)
        cleaned = self.clean_tokens(tokens, token_scores)
        aggregated = self.aggregate_tokens(cleaned)
        aggregated.sort(key=lambda x: x["score"], reverse=True)

        explanation = {
            "delta": float(delta),
            "top_words": aggregated[:top_k],
            "all_words": aggregated
        }

        return {
            "category": category,
            "explanation": explanation,
        }


TOP_K = int(os.getenv("TOP_K", "10"))
IG_STEPS = int(os.getenv("IG_STEPS", "120"))
USE_GPU = os.getenv("USE_GPU", "false").lower() in {"1", "true", "yes"}
MODEL_CHOICE = os.getenv("MODEL_CHOICE", "bert-base-uncased")
HF_USER = os.getenv("HF_USER", "yamazakikakuyo")

app = FastAPI(
    title="MBTI + Integrated Gradients XAI API",
    version="1.0.0",
    description="POST text → get MBTI classification + word attributions (score & percentage) via Integrated Gradients."
)

class PredictIn(BaseModel):
    text: constr(min_length=1)
    n_steps: int = Field(IG_STEPS, ge=1, le=1024)
    top_k: int = Field(TOP_K, ge=1, le=100)
    explain: bool = True
    hf_token: str | None = None
    hf_user: str | None = None

class PredictOut(BaseModel):
    mbti_result: Dict[str, Any]
    explanation_result: Any

@app.get("/health")
def health():
    # Don't touch models; purely a liveness probe
    return {"status": "ok"}

# --- Lazy model init ---
mbti_obj: MBTIPipeline | None = None
def get_mbti() -> MBTIPipeline:
    global mbti_obj
    if mbti_obj is None:
        # If your models are private, make sure HF_TOKEN is set on the Lambda env.
        mbti_obj = MBTIPipeline(model_choice=MODEL_CHOICE, user=HF_USER, use_gpu=USE_GPU)
    return mbti_obj

def _build_response(text: str, n_steps: int, top_k: int, explain: bool) -> Dict[str, Any]:
    obj = get_mbti()
    mbti_result = obj.predict_mbti_only(text)
    explanation_result = []
    if explain:
        for category in obj.categories:
            explanation_result.append(
                obj.report_ig(text, category, n_steps=n_steps, top_k=top_k)
            )
    return {"mbti_result": mbti_result, "explanation_result": explanation_result}

@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn):
    global HF_USER, mbti_obj
    try:
        # If token/user provided in body, set and force reinit (optional)
        if body.hf_token and not os.getenv("HF_TOKEN"):
            os.environ["HF_TOKEN"] = body.hf_token
            mbti_obj = None  # recreate with token

        if body.hf_user and body.hf_user != HF_USER:
            HF_USER = body.hf_user
            mbti_obj = None  # recreate with new owner

        return _build_response(body.text, body.n_steps, body.top_k, body.explain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e

handler = Mangum(app)