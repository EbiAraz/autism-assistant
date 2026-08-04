"""تحلیل شکاف (Gap Analysis): کدام دسته‌های اوتیسم پوشش قوی/ضعیف دارند؟

خروجی:
- توزیع تعداد و درصد هر دسته (top-1).
- میانگین/میانه/انحراف معیار امتیاز top-1 برای هر دسته.
- نسبت نمونه‌های «بااعتماد» (top_score >= threshold).
- رتبه‌بندی قوی/ضعیف و توصیه برای فاز بعدی (داده‌ی مکمل).
- نمودار میله‌ای توزیع + نمودار میانگین امتیاز.
- گزارش Markdown و JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # بدون GUI / headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from facts import CATEGORIES, CATEGORY_KEYS


def _category_label(cat: str) -> str:
    return f"{cat}. {CATEGORIES[cat]['title_en']}"


def build_report(df: pd.DataFrame, scores: np.ndarray,
                 threshold: float = config.LABEL_THRESHOLD) -> dict:
    """ساختن دیکشنری گزارش از DataFrame لیبل‌شده."""
    n = len(df)
    cat_keys = CATEGORY_KEYS

    # امتیاز top-1 هر نمونه و دسته‌ی top
    top_scores = df["top_score"].to_numpy(dtype=float)
    top_cats = df["top_label"].to_numpy()

    per_cat: dict[str, dict] = {}
    for c in cat_keys:
        mask = top_cats == c
        cnt = int(mask.sum())
        sc = top_scores[mask] if cnt else np.array([])
        per_cat[c] = {
            "title": CATEGORIES[c]["title_en"],
            "title_fa": CATEGORIES[c]["title_fa"],
            "count": cnt,
            "share": round(cnt / n, 4) if n else 0.0,
            "mean_score": round(float(sc.mean()), 4) if cnt else 0.0,
            "median_score": round(float(np.median(sc)), 4) if cnt else 0.0,
            "std_score": round(float(sc.std()), 4) if cnt else 0.0,
            "confident_share": (
                round(float((sc >= threshold).mean()), 4) if cnt else 0.0
            ),
        }

    # میانگین امتیاز هر دسته روی کل نمونه‌ها (بدون شرط top-1) — نشان‌دهنده‌ی
    # میزان هم‌خوانی کلی داده‌ها با آن دسته.
    global_cat_score = {
        cat: round(float(mean), 4)
        for cat, mean in zip(cat_keys, np.mean(scores, axis=0))
    }

    # رتبه‌بندی ضعیف→قوی بر اساس count و mean_score ترکیبی
    ranking = sorted(
        cat_keys,
        key=lambda c: (per_cat[c]["count"], per_cat[c]["mean_score"]),
    )

    n_confident = int((top_scores >= threshold).sum())
    n_none = int((df["top_label"] == "NONE").sum()) if "NONE" in set(top_cats) else 0

    report = {
        "n_samples": n,
        "threshold": threshold,
        "n_confident": n_confident,
        "confident_rate": round(n_confident / n, 4) if n else 0.0,
        "n_unlabeled_NONE": n_none,
        "per_category": per_cat,
        "global_mean_score_per_category": global_cat_score,
        "ranking_weak_to_strong": ranking,
        "weakest": ranking[:2],
        "strongest": ranking[-2:],
    }
    return report


def _recommendations(report: dict) -> list[str]:
    recs: list[str] = []
    for c in report["weakest"]:
        pc = report["per_category"][c]
        recs.append(
            f"- **{c}. {pc['title']}**: پوشش ضعیف (count={pc['count']}, "
            f"share={pc['share']:.1%}, mean_score={pc['mean_score']}). "
            f"برای فاز after-tuning/RAG داده‌ی مکمل از دیتاست‌های پزشکی/ASD بعدی "
            f"برای این دسته اضافه شود."
        )
    return recs


def render_markdown(report: dict) -> str:
    pc = report["per_category"]
    lines = [
        "# Gap Analysis — MentalChat16K × Autism Facts (A–F)",
        "",
        f"- تعداد نمونه‌ها / samples: **{report['n_samples']}**",
        f"- آستانه‌ی اعتماد / threshold: **{report['threshold']}**",
        f"- نمونه‌های بااعتماد / confident: "
        f"**{report['n_confident']} ({report['confident_rate']:.1%})**",
        "",
        "## Distribution per category (top-1)",
        "",
        "| Cat | Title (EN) | Count | Share | Mean top-score | Median | Std | Confident% |",
        "|-----|------------|-------|-------|----------------|--------|-----|------------|",
    ]
    for c in CATEGORY_KEYS:
        d = pc[c]
        lines.append(
            f"| {c} | {d['title']} | {d['count']} | {d['share']:.1%} | "
            f"{d['mean_score']} | {d['median_score']} | {d['std_score']} | "
            f"{d['confident_share']:.1%} |"
        )
    lines += [
        "",
        "## Global mean similarity per category (over all samples)",
        "",
    ]
    for c in CATEGORY_KEYS:
        lines.append(
            f"- {c}. {pc[c]['title']}: {report['global_mean_score_per_category'][c]}"
        )
    lines += [
        "",
        "## Ranking (weak → strong)",
        "",
        " → ".join(report["ranking_weak_to_strong"]),
        "",
        "## Weakest categories & recommendations",
        "",
    ]
    lines += _recommendations(report)
    lines += [
        "",
        "## Strongest categories",
        "",
        f"- **{report['strongest'][-1]}**: {pc[report['strongest'][-1]]['title']}",
        f"- **{report['strongest'][-2]}**: {pc[report['strongest'][-2]]['title']}",
        "",
    ]
    return "\n".join(lines)


def plot_distribution(report: dict, out_path: Path) -> None:
    cats = CATEGORY_KEYS
    counts = [report["per_category"][c]["count"] for c in cats]
    means = [report["per_category"][c]["mean_score"] for c in cats]
    labels = [_category_label(c) for c in cats]

    fig, ax1 = plt.subplots(figsize=(11, 6))
    x = np.arange(len(cats))
    bars = ax1.bar(x, counts, color="#4C72B0", alpha=0.85, label="Count (top-1)")
    ax1.set_ylabel("Count of samples (top-1)", color="#4C72B0")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    for b, cnt in zip(bars, counts):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 str(cnt), ha="center", va="bottom", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(x, means, "-o", color="#C44E52", label="Mean top-score")
    ax2.set_ylabel("Mean top-1 cosine similarity", color="#C44E52")
    ax2.set_ylim(0, max(0.6, max(means) + 0.05))

    plt.title("MentalChat16K — autism category distribution & confidence")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[gap_analysis] Chart saved -> {out_path}")


def run(df: pd.DataFrame, scores: np.ndarray) -> dict:
    report = build_report(df, scores)
    md = render_markdown(report)
    config.GAP_REPORT_MD.write_text(md, encoding="utf-8")
    config.GAP_REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_distribution(report, config.GAP_CHART_PATH)
    print(f"[gap_analysis] Reports saved -> {config.GAP_REPORT_MD}, "
          f"{config.GAP_REPORT_JSON}")
    return report
