"""تنظیمات مرکزی پروژه / Central configuration for the pipeline.

این فایل همه‌ی پارامترهای قابل تنظیم را در یک جا نگه می‌دارد تا با اجرای
`python main.py` رفتار pipeline را بدون دستکاری بقیه‌ی ماژول‌ها کنترل کنید.
"""
from __future__ import annotations

from pathlib import Path

# --- مسیرها / Paths ---
ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT / "data"
OUTPUT_DIR: Path = ROOT / "outputs"
CACHE_DIR: Path = ROOT / ".cache"

for _p in (DATA_DIR, OUTPUT_DIR, CACHE_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# --- دیتاست / Dataset ---
DATASET_NAME: str = "ShenLab/MentalChat16K"
DATASET_SPLIT: str = "train"
# برای تست سریع روی CPU می‌توانید مقدار زیر را کوچک نگه دارید (مثلاً 200).
# None یعنی کل دیتاست. / None means the whole dataset.
SAMPLE_LIMIT: int | None = 500
# فیلدهای متنی دیتاست که با هم به‌عنوان «utterance» برای لیبل‌گذاری استفاده می‌شوند.
# Fields merged to form one utterance for semantic labeling.
DATASET_TEXT_FIELDS: tuple[str, ...] = ("instruction", "input", "output")

# --- مدل امبدینگ / Embedding model ---
# مدل چندزبانه‌ی سبک که روی CPU هم کار می‌کند و فارسی/انگلیسی را پشتیبانی می‌کند.
EMBED_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_BATCH_SIZE: int = 64
# نرمال‌سازی امبدینگ‌ها برای محاسبه‌ی cosine similarity.
NORMALIZE_EMBEDDINGS: bool = True

# --- استراتژی لیبل‌گذاری / Labeling strategy ---
# هر دسته چند «prototype» (فکت) دارد؛ امتیاز دسته = max cosine sim روی prototypeها.
# Each category has several prototype facts; category score = max cosine sim over prototypes.
TOP_K_PROTOTYPES_PER_CAT: int = 1  # 1 یعنی max; مقادیر بالاتر میانگین top-k می‌گیرد.
# آستانه‌ی اعتماد برای پذیرش لیبل (cosine similarity). / Confidence threshold.
LABEL_THRESHOLD: float = 0.35
# اگر بیش از یک دسته بالای آستانه باشند، خروجی multi-label می‌شود.
MULTI_LABEL: bool = True

# --- خروجی‌ها / Outputs ---
LABELED_PATH = OUTPUT_DIR / "labeled.csv"
SCORES_PATH = OUTPUT_DIR / "scores.npy"
GAP_REPORT_MD = OUTPUT_DIR / "gap_analysis.md"
GAP_REPORT_JSON = OUTPUT_DIR / "gap_analysis.json"
GAP_CHART_PATH = OUTPUT_DIR / "gap_distribution.png"
