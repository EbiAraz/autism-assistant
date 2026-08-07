"""Streamlit UI for Autism Assistant.

Launch:
    streamlit run ui.py
"""
from __future__ import annotations

import streamlit as st
from data_loader import load_samples
from labeler import SemanticLabeler
from facts import CATEGORIES
import config


@st.cache_resource
def get_labeler() -> SemanticLabeler:
    return SemanticLabeler()


@st.cache_data
def get_samples(limit: int = 50) -> list:
    return load_samples(limit=limit)


def format_category_scores(score_vector: dict[str, float]) -> list[tuple[str, float]]:
    return [(cat, float(score_vector[cat])) for cat in score_vector]


def render_label_result(result: dict) -> None:
    st.subheader("Labeling result")
    st.metric("Top category", f"{result['top_label']} ({result['top_score']:.4f})")
    st.markdown("**Assigned categories:**")
    st.write(", ".join(result["labels"]))
    st.markdown("**Confidence:**  ")
    st.write("✅ Confident" if result["confident"] else "⚠️ Not confident")

    chart_data = {category: [score] for category, score in result["score_vector"].items()}
    st.bar_chart(chart_data)

    st.write("### Category scores")
    st.table([
        {"Category": cat, "Score": f"{score:.4f}"}
        for cat, score in result["score_vector"].items()
    ])


def main() -> None:
    st.set_page_config(page_title="Autism Assistant UI", layout="wide")
    st.title("Autism Assistant — Semantic Labeling UI")
    st.write(
        "Interactive browser UI for semantic labeling with Autism Assistant. "
        "Enter free text or select a sample from the MentalChat16K dataset."
    )

    labeler = get_labeler()

    with st.sidebar:
        st.header("Input options")
        sample_limit = st.slider(
            "Load dataset examples", min_value=10, max_value=200,
            value=50, step=10,
        )
        use_dataset = st.checkbox("Enable sample selection", value=True)

        st.markdown("---")
        st.header("Project info")
        st.write("Model:", labeler.model.__class__.__name__)
        st.write("Embedding model:", config.EMBED_MODEL_NAME)
        st.write("Category count:", len(labeler.cat_keys))

    default_text = ""
    samples = []
    if use_dataset:
        with st.spinner("Loading sample dataset..."):
            samples = get_samples(limit=sample_limit)
        sample_options = [f"#{s.idx}: {s.text[:90].replace('\n', ' ')}..." for s in samples]
        selected = st.selectbox("Choose a dataset sample", options=sample_options)
        sample_index = sample_options.index(selected)
        default_text = samples[sample_index].text

    text = st.text_area("Text to label", value=default_text, height=220)
    if not text.strip():
        st.info("Type text or select a sample to analyze.")
        return

    if st.button("Analyze text"):
        with st.spinner("Embedding and scoring text..."):
            results = labeler.assign_labels(labeler.score_samples(labeler.embed_samples([text])))
            render_label_result(results[0])

    st.markdown("---")
    st.subheader("Category facts")
    category_table = {cat: f"{CATEGORIES[cat]['title_en']} / {CATEGORIES[cat]['title_fa']}" for cat in labeler.cat_keys}
    st.table(category_table)


if __name__ == "__main__":
    main()
