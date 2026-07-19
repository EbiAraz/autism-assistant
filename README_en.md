# Autism Assistant — Semantic Labeling & Gap Analysis on MentalChat16K

A project to develop an LLM (using RAG or fine-tuning) for **autistic patients with language delay**.
This repository implements the data preparation stage: loading the **MentalChat16K** dataset (mental health group),
semantically labeling samples (via semantic similarity) based on 7 autism fact categories (A through G), and performing a **gap analysis**
to identify strong and weak categories before training the final model.

## Pipeline

1. `data_loader.py` — Downloads `ShenLab/MentalChat16K` from Hugging Face and merges fields into a single utterance.
2. `facts.py` — 7 categories (A–G), each with several reference facts serving as semantic **prototypes**.
3. `labeler.py` — Multilingual embeddings (`paraphrase-multilingual-MiniLM-L12-v2`), computes cosine similarity
   between each sample and the prototypes, and assigns a top-1 label (with optional multi-label using a threshold).
4. `gap_analysis.py` — Category distribution, mean/variance of scores, identification of strong/weak categories, and a chart.
5. `main.py` — Orchestrates all stages with a single command.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Run

```bash
# Full run on the first 500 samples (default config.py for quick CPU testing)
python main.py

# Run on the entire dataset
python main.py --limit 0

# Limited run for smoke testing
python main.py --limit 100
```

Outputs are written to the `outputs/` folder: `labeled.csv`, `gap_analysis.md`,
`gap_analysis.json`, `gap_distribution.png`.

## Categories (A–G)

| Key | Title | #facts |
|-----|-------|--------|
| A | Social Communication | 6 |
| B | Sensory Processing | 3 |
| C | Emotional Regulation & Stress | 4 |
| D | Routine & Predictability | 3 |
| E | Special Interests & Strengths | 7 |
| F | Diagnosis & Support | 6 |
| G | Autism Knowledge & Awareness | 14 |

(Full facts are defined in `facts.py`.)

## Notes

- The embedding model is **multilingual** so that Persian/English facts and the English dataset texts are compared in a shared semantic space — matching by meaning, not by word.
- **Multi-prototype** strategy: the score for each category = maximum similarity across that category's prototypes.
- The gap analysis output identifies which autism categories have weak coverage and need supplementary data (medical/ASD datasets) added in the after-tuning/RAG phase.
