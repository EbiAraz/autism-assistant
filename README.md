# Autism Assistant — Semantic Labeling & Gap Analysis on MentalChat16K

[![Python tests](https://github.com/EbiAraz/autism-assistant/actions/workflows/python-tests.yml/badge.svg)](https://github.com/EbiAraz/autism-assistant/actions/workflows/python-tests.yml)

پروژه‌ی توسعه‌ی یک LLM (با RAG یا fine-tuning) برای **بیماران اوتیستیک با تأخیر زبانی (language delay)**.
این مخزن مرحله‌ی آماده‌سازی داده را پیاده‌سازی می‌کند: بارگذاری دیتاست **MentalChat16K** (گروه سلامت روان)،
لیبل‌گذاری معنایی (semantic similarity) نمونه‌ها بر اساس ۷ دسته‌ی فکت اوتیسم (A تا G)، و تحلیل شکاف (gap analysis)
برای شناسایی دسته‌های قوی و ضعیف قبل از آموزش مدل نهایی.

## Pipeline / خط لوله

1. `data_loader.py` — دانلود `ShenLab/MentalChat16K` از Hugging Face و ادغام فیلدها به یک utterance.
2. `facts.py` — ۷ دسته (A–G) هرکدام با چند فکت مرجع به‌عنوان **prototype**‌های معنایی.
3. `labeler.py` — امبدینگ چندزبانه (`paraphrase-multilingual-MiniLM-L12-v2`)، محاسبه‌ی cosine similarity
   بین هر نمونه و prototypeها، و تخصیص لیبل top-1 (و multi-label اختیاری با آستانه).
4. `gap_analysis.py` — توزیع دسته‌ها، میانگین/واریانس امتیاز، شناسایی دسته‌های قوی/ضعیف و نمودار.
5. `main.py` — orchestration کل مراحل با یک دستور.

## Install / نصب

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Run / اجرا

```bash
# اجرای کامل روی ۵۰۰ نمونه اول (پیش‌فرض config.py برای تست سریع روی CPU)
python main.py

# اجرای کل دیتاست
python main.py --limit 0

# اجرای محدود برای smoke test
python main.py --limit 100
```

```bash
# راه‌اندازی رابط کاربری مرورگر
streamlit run ui.py
```

خروجی‌ها در پوشه‌ی `outputs/` نوشته می‌شوند: `labeled.csv`, `gap_analysis.md`,
`gap_analysis.json`, `gap_distribution.png`.

## Tests / تست‌ها

برای اجرای تست‌ها از `pytest` استفاده کنید:

```bash
python -m pytest
```

## Continuous Integration / یکپارچه‌سازی پیوسته

این مخزن یک GitHub Actions workflow دارد که روی هر `push` و `pull_request` به `main`
تست‌ها را اجرا می‌کند.

| Workflow | Description |
|----------|-------------|
| `.github/workflows/python-tests.yml` | Run `pytest` on Ubuntu with Python 3.12 |

## Categories / دسته‌ها (A–G)

| Key | Title | #facts |
|-----|-------|--------|
| A | Social Communication | 6 |
| B | Sensory Processing | 3 |
| C | Emotional Regulation & Stress | 4 |
| D | Routine & Predictability | 3 |
| E | Special Interests & Strengths | 7 |
| F | Diagnosis & Support | 6 |
| G | Autism Knowledge & Awareness | 14 |

(فکت‌های کامل باینکاری در `facts.py`.)

## Notes / یادداشت‌ها

- مدل امبدینگ **multilingual** است تا فکت‌های فارسی/انگلیسی و متون انگلیسی دیتاست را در یک فضای
  معنایی مشترک مقایسه کند — نه تطبیق کلمه‌ای، بلکه تطبیق معنا.
- استراتژی **multi-prototype**: امتیاز هر دسته = حداکثر شباهت روی prototypeهای آن دسته.
- خروجی gap analysis مشخص می‌کند کدام دسته‌های اوتیسم پوشش داده‌ی ضعیفی دارند و در فاز after-tuning/RAG
  باید داده‌ی مکمل (دیتاست‌های پزشکی/ASD بعدی) برای آن‌ها اضافه شود.
