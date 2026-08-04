"""طبقه‌بندی embedding-based معنایی با multi-prototype.

روش:
1. امبدینگ همه‌ی prototypeها (فکت‌های A–G به انگلیسی و فارسی) با مدل چندزبانه.
2. امبدینگ همه‌ی نمونه‌های دیتاست.
3. ماتریس cosine similarity نمونه × prototype.
4. امتیاز هر دسته = max (یا میانگین top-k) شباهت روی prototypeهای همان دسته.
5. لیبل top-1 = دسته‌ای با بیشترین امتیاز؛ multi-label = دسته‌هایی بالای آستانه.

این روش «معنا» را مقایسه می‌کند، نه کلمه؛ بنابراین متون انگلیسی دیتاست و
فکت‌های فارسی/انگلیسی را در یک فضای مشترک می‌نشاند.
"""
from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config
from facts import CATEGORY_KEYS, CATEGORIES, get_prototypes


class SemanticLabeler:
    def __init__(self, model_name: str = config.EMBED_MODEL_NAME,
                 device: str | None = None):
        # روی CPU اجرا می‌شود مگر اینکه GPU در دسترس باشد.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"[labeler] Loading embedding model '{model_name}' on {device} ...")
        self.model = SentenceTransformer(model_name, device=device,
                                         cache_folder=str(config.CACHE_DIR))
        self.cat_keys = CATEGORY_KEYS
        self.prototypes = get_prototypes()
        # ماسک دسته‌ی هر prototype برای محاسبه‌ی گروهی امتیاز دسته
        self.proto_texts = [p["text"] for p in self.prototypes]
        self.proto_cats = [p["category"] for p in self.prototypes]
        self.proto_cat_indices = {
            cat: np.fromiter((i for i, c in enumerate(self.proto_cats) if c == cat),
                             dtype=np.int64)
            for cat in self.cat_keys
        }
        self._proto_emb = self._embed_prototypes()

    # --- امبدینگ ---
    def _embed(self, texts: list[str], batch_size: int,
               desc: str = "embedding") -> np.ndarray:
        embs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=config.NORMALIZE_EMBEDDINGS,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return embs.astype(np.float32)

    def _embed_prototypes(self) -> np.ndarray:
        emb = self._embed(self.proto_texts, batch_size=32, desc="prototypes")
        print(f"[labeler] Prototype embeddings: {emb.shape}")
        return emb

    def embed_samples(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, batch_size=config.EMBED_BATCH_SIZE,
                           desc="samples")

    # --- امتیازدهی دسته‌ها ---
    def score_samples(self, sample_embs: np.ndarray,
                      top_k: int = config.TOP_K_PROTOTYPES_PER_CAT
                      ) -> np.ndarray:
        """برگرداند ماتریس (n_samples, n_categories) از امتیاز cosine."""
        # چون امبدینگ‌ها نرمالایز شده‌اند، ضرب داخلی = cosine similarity.
        sim = sample_embs @ self._proto_emb.T  # (N, P)
        n_samples = sim.shape[0]
        scores = np.zeros((n_samples, len(self.cat_keys)), dtype=np.float32)

        # برای هر دسته، روی prototypeهای آن: max یا میانگین top-k
        for ci, cat in enumerate(self.cat_keys):
            cols = self.proto_cat_indices[cat]
            sub = sim[:, cols]  # (N, P_cat)
            if top_k <= 1:
                scores[:, ci] = sub.max(axis=1)
            else:
                k = min(top_k, sub.shape[1])
                part = np.partition(sub, -k, axis=1)[:, -k:]
                scores[:, ci] = part.mean(axis=1)
        return scores

    # --- تخصیص لیبل ---
    def assign_labels(self, scores: np.ndarray,
                      threshold: float = config.LABEL_THRESHOLD,
                      multi_label: bool = config.MULTI_LABEL
                      ) -> list[dict]:
        results: list[dict] = []
        order = np.argsort(scores, axis=1)[:, ::-1]
        n_categories = scores.shape[1]

        for i in range(scores.shape[0]):
            row_order = order[i]
            top_idx = int(row_order[0])
            top_cat = self.cat_keys[top_idx]
            top_score = float(scores[i, top_idx])

            if multi_label:
                above = [self.cat_keys[j] for j in row_order
                         if scores[i, j] >= threshold]
                labels = above if above else [top_cat]
            else:
                labels = [top_cat] if top_score >= threshold else ["NONE"]

            results.append({
                "top_label": top_cat,
                "top_score": round(top_score, 4),
                "labels": labels,
                "score_vector": {
                    self.cat_keys[j]: round(float(scores[i, j]), 4)
                    for j in range(n_categories)
                },
                "confident": top_score >= threshold,
            })
        return results


def label_texts(texts: list[str]) -> tuple[list[dict], np.ndarray]:
    """راحت‌ترین API: متن‌ها → (لیست نتایج، ماتریس امتیازها)."""
    labeler = SemanticLabeler()
    embs = labeler.embed_samples(texts)
    scores = labeler.score_samples(embs)
    results = labeler.assign_labels(scores)
    return results, scores


if __name__ == "__main__":
    demo = [
        "My child flaps his hands and gets very upset when the lights are too bright.",
        "He has a deep passion for trains and can talk about them for hours.",
        "We just got the autism diagnosis after the developmental assessment.",
        "I feel anxious when my daily routine changes unexpectedly.",
    ]
    res, _ = label_texts(demo)
    for txt, r in zip(demo, res):
        print("-" * 60)
        print(txt)
        print(f"  -> top: {r['top_label']} ({r['top_score']}) | "
              f"labels: {r['labels']} | scores: {r['score_vector']}")
