"""Streamlit UI for Autism Assistant.

Launch:
    streamlit run ui.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from data_loader import load_samples
from labeler import SemanticLabeler
from facts import CATEGORIES, category_titles_bilingual
import config


@st.cache_resource
def get_labeler() -> SemanticLabeler:
    return SemanticLabeler()


@st.cache_data
def get_samples(limit: int = 50) -> list:
    return load_samples(limit=limit)


def build_score_table(score_vector: dict[str, float]) -> pd.DataFrame:
    df = pd.DataFrame(
        sorted(score_vector.items(), key=lambda item: item[1], reverse=True),
        columns=["Category", "Score"],
    )
    df["Score"] = df["Score"].round(4)
    return df


def render_label_result(result: dict) -> None:
    st.subheader("Labeling result")
    top_label = result["top_label"]
    top_score = result["top_score"]
    confident = result["confident"]

    left, right = st.columns([2, 3])
    with left:
        st.metric("Top category", f"{top_label}", delta=f"score={top_score:.4f}")
        st.markdown("**Assigned categories:**")
        st.write(", ".join(result["labels"]))
        st.markdown("**Confidence:**")
        st.write("✅ Confident" if confident else "⚠️ Not confident")

    with right:
        score_df = build_score_table(result["score_vector"])
        st.bar_chart(score_df.set_index("Category"))

    st.markdown("### Category scores")
    st.dataframe(score_df, use_container_width=True)


def render_category_legend() -> None:
    titles = category_titles_bilingual()
    legend = pd.DataFrame(
        [(key, value) for key, value in titles.items()],
        columns=["Category", "Title (EN / FA)"],
    )
    st.table(legend)


def main() -> None:
    st.set_page_config(
        page_title="Autism Assistant UI",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Autism Assistant — Semantic Labeling UI")
    st.write(
        "Interactive browser UI for semantic labeling with Autism Assistant. "
        "Use the sample loader or paste your own text to evaluate category similarity."
    )

    labeler = get_labeler()

    with st.sidebar:
        st.header("Input options")
        sample_limit = st.slider(
            "Load dataset examples",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
        )
        use_dataset = st.checkbox("Enable sample selection", value=True)
        st.markdown("---")
        st.header("Project info")
        st.write("Embedding model:", config.EMBED_MODEL_NAME)
        st.write("Category count:", len(labeler.cat_keys))
        st.write("Confidence threshold:", config.LABEL_THRESHOLD)
        st.markdown("---")
        st.header("Category legend")
        st.write("Browse the category titles used for scoring.")
        render_category_legend()

    sample_text = ""
    sample_meta = None
    samples = []
    if use_dataset:
        with st.spinner("Loading sample dataset..."):
            samples = get_samples(limit=sample_limit)

        if samples:
            sample_options = [
                f"#{sample.idx}: {sample.text[:90].replace('\n', ' ')}..."
                for sample in samples
            ]
            selected = st.selectbox("Choose a dataset sample", options=sample_options)
            selected_index = sample_options.index(selected)
            sample_text = samples[selected_index].text
            sample_meta = samples[selected_index].meta
        else:
            st.warning("No samples were loaded. Try increasing the dataset limit.")

    text = st.text_area("Text to label", value=sample_text, height=260)
    analyze_button = st.button("Analyze text")

    if analyze_button:
        if not text.strip():
            st.warning("Please type some text or select a sample before analyzing.")
        else:
            with st.spinner("Embedding and scoring text..."):
                results = labeler.assign_labels(
                    labeler.score_samples(labeler.embed_samples([text]))
                )
            render_label_result(results[0])

            if sample_meta is not None:
                with st.expander("Sample metadata"):
                    st.json(sample_meta)

    st.markdown("---")
    st.subheader("Tips")
    st.write(
        "Use the sidebar to load dataset examples and compare how the model "
        "assigns categories. Paste your own text for quick semantic labeling."
    )


if __name__ == "__main__":
    main()
