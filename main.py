"""Orchestrator کل pipeline:

    دانلود دیتاست → امبدینگ prototypeها و نمونه‌ها → لیبل‌گذاری معنایی
    → ذخیره‌ی خروجی CSV + امتیازها → gap analysis + گزارش/نمودار

اجرا:
    python main.py              # با SAMPLE_LIMIT از config.py (پیش‌فرض 500)
    python main.py --limit 0    # کل دیتاست
    python main.py --limit 100  # ۱۰۰ نمونه
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

import config
from data_loader import load_samples
from labeler import SemanticLabeler
from gap_analysis import run as run_gap_analysis


def _reconfigure_stdio_utf8() -> None:
    # روی ویندوز، وقتی stdout یک pipe (non-TTY) باشد، پایتون از codec پیش‌فرض
    # سیستم (cp1252) استفاده می‌کند و نمی‌تواند متن فارسی را چاپ کند.
    # با UTF-8 دوباره‌پیکربندی می‌کنیم تا printهای فارسی crash نکنند.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_reconfigure_stdio_utf8()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Semantic labeling + gap analysis.")
    p.add_argument("--limit", type=int, default=None,
                   help="تعداد نمونه‌ها. 0 یعنی کل دیتاست. "
                        "default = config.SAMPLE_LIMIT")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    limit = args.limit
    if limit is None:
        limit = config.SAMPLE_LIMIT
    if limit is not None and limit <= 0:
        limit = None

    # 1) بارگذاری داده
    samples = load_samples(limit=limit)
    if not samples:
        print("[main] No samples loaded. Exiting.")
        return 1

    texts = [s.text for s in samples]

    # 2) لیبل‌گذاری معنایی
    labeler = SemanticLabeler()
    embs = labeler.embed_samples(texts)
    scores = labeler.score_samples(embs)
    results = labeler.assign_labels(scores)

    # 3) ذخیره‌ی خروجی
    rows = []
    for s, r in zip(samples, results):
        rows.append({
            "idx": s.idx,
            "text": s.text,
            "top_label": r["top_label"],
            "top_score": r["top_score"],
            "labels": "|".join(r["labels"]),
            "confident": r["confident"],
            **{f"score_{k}": r["score_vector"][k] for k in r["score_vector"]},
        })
    df = pd.DataFrame(rows)
    df.to_csv(config.LABELED_PATH, index=False, encoding="utf-8-sig")
    np.save(config.SCORES_PATH, scores)
    print(f"[main] Labeled data saved -> {config.LABELED_PATH}")
    print(f"[main] Score matrix saved -> {config.SCORES_PATH} "
          f"(shape={scores.shape})")

    # 4) Gap analysis
    report = run_gap_analysis(df, scores)

    # 5) چاپ خلاصه در ترمینال
    print("\n" + "=" * 60)
    print("SUMMARY / خلاصه")
    print("=" * 60)
    print(f"Samples: {report['n_samples']} | "
          f"Confident: {report['n_confident']} ({report['confident_rate']:.1%})")
    print("\nCategory distribution (top-1):")
    for c in config_facts_keys():
        d = report["per_category"][c]
        print(f"  {c}. {d['title']:32s} count={d['count']:5d} "
              f"share={d['share']:.1%} mean={d['mean_score']}")
    print(f"\nWeak  → Strong: {' > '.join(report['ranking_weak_to_strong'])}")
    print(f"\nReports: {config.GAP_REPORT_MD}")
    print(f"         {config.GAP_REPORT_JSON}")
    print(f"         {config.GAP_CHART_PATH}")
    return 0


def config_facts_keys():
    # ایمپورت محلی برای جلوگیری از circular در بالا
    from facts import CATEGORY_KEYS
    return CATEGORY_KEYS


if __name__ == "__main__":
    sys.exit(main())
