"""بارگذاری دیتاست MentalChat16K از Hugging Face و ساختن utterance.

دیتاست: ShenLab/MentalChat16K (QAهای سلامت روان، انگلیسی).
فیلدهای instruction/input/output به‌صورت یک متن ادغام می‌شوند تا معنای کامل
یک نمونه برای لیبل‌گذاری معنایی در دسترس باشد.
"""
from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset

import config


@dataclass
class Sample:
    idx: int
    text: str
    meta: dict


def _merge_fields(row: dict, fields: tuple[str, ...]) -> str:
    """ادغام فیلدهای متنی دیتاست در یک رشته‌ی تمیز."""
    parts: list[str] = []
    for f in fields:
        val = row.get(f)
        if isinstance(val, str) and val.strip():
            parts.append(f"{f}: {val.strip()}")
    return "\n".join(parts) if parts else ""


def load_samples(limit: int | None = None) -> list[Sample]:
    """بارگذاری دیتاست و تبدیل به لیست Sample.

    Args:
        limit: حداکثر تعداد نمونه. None یا 0 یعنی کل دیتاست.
    """
    if limit is not None and limit <= 0:
        limit = None

    print(f"[data_loader] Loading '{config.DATASET_NAME}' (split={config.DATASET_SPLIT}) ...")
    ds = load_dataset(
        config.DATASET_NAME,
        split=config.DATASET_SPLIT,
        cache_dir=str(config.CACHE_DIR),
    )

    # یک snapshot از ساختار برای دیباگ / inspect schema for debugging
    cols = list(ds.features.keys())
    print(f"[data_loader] Columns: {cols} | total rows: {len(ds)}")

    # فقط فیلدهایی که واقعاً وجود دارند را استفاده کن / use only existing fields
    fields = tuple(f for f in config.DATASET_TEXT_FIELDS if f in cols)
    if not fields:
        # fallback: همه‌ی فیلدهای متنی / all string fields
        fields = tuple(c for c in cols if ds.features[c].dtype == "string")
    print(f"[data_loader] Using text fields: {fields}")

    n = len(ds) if limit is None else min(limit, len(ds))
    samples: list[Sample] = []
    for i in range(n):
        row = ds[i]
        text = _merge_fields(row, fields)
        if not text:
            continue
        samples.append(Sample(idx=i, text=text, meta={k: row[k] for k in fields}))

    print(f"[data_loader] Prepared {len(samples)} samples for labeling.")
    return samples


if __name__ == "__main__":
    s = load_samples(limit=5)
    for x in s:
        print("-" * 60)
        print(x.text[:300])
