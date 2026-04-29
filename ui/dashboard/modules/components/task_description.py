import streamlit as st
from modules.dataset_metadata import DATASET_METADATA


def render(dataset_key):
    """Render dataset-specific task description in clean professional format."""
    meta = DATASET_METADATA.get(dataset_key)
    if not meta:
        st.info("Dataset documentation has not been added yet.")
        return

    # Title
    st.markdown(
        f"### :material/assignment: Task Overview — **{meta['title']}**",
        unsafe_allow_html=True,
    )

    # Goal
    st.markdown("#### :material/flag_circle: Objective")
    st.markdown(meta["goal"])

    # Dataset Description
    st.markdown("#### :material/info: Dataset Summary")
    st.markdown(meta["description"])

    # Labels
    st.markdown("#### :material/label: Label Definition")
    st.markdown(meta["label_definition"])

    # Features
    st.markdown("#### :material/dataset: Feature Definitions")
    for feature, content in meta["feature_definitions"].items():
        definition = content.get("definition", "")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**{feature}**")
            st.caption(definition)

        if "categories" in content:
            with col2:
                st.markdown("*Categories:*")
                for category in content["categories"]:
                    st.markdown(f"- {category}")

        st.divider()
