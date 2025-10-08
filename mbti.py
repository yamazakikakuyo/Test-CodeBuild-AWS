import string
import contextlib
from collections import defaultdict

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from captum.attr import IntegratedGradients
import nltk
nltk.data.path.append(0,'/usr/share/nltk_data')

from nltk.corpus import stopwords

try:
    _ = stopwords.words("english")
except Exception:
    nltk.download("stopwords", quiet=True, download_dir='/usr/share/nltk_data')

class MBTIPipeline:
    def __init__(self, model_choice="bert-base-uncased", user="yamazakikakuyo", use_gpu=False):
        self.categories = ["energy", "information", "decision", "execution"]
        self.user = user
        self.model_choice = model_choice

        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        hf_device = 0 if self.device.type == "cuda" else -1

        for category in self.categories:
            repo_id = f"{self.user}/MBTI-{self.model_choice}-{category}"
            tokenizer = AutoTokenizer.from_pretrained(repo_id)
            model_category = AutoModelForSequenceClassification.from_pretrained(repo_id)
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
