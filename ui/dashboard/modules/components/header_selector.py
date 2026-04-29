# dashboard/modules/components/header_selector.py
import streamlit as st
from modules.components import case_loader


def on_dataset_change(dataset_names, background_size: int = 1000):
    selected_choice = st.session_state["dataset_selector"]
    dataset_id = dataset_names.get(selected_choice)
    case_loader.load_reference_data(dataset_id, size=background_size)


def render(dataset_names, background_size: int = 1000):
    """Renders the 'Case Review' title and the dataset selector."""
    st.markdown(
        """
        <style>
            .case-header {
                display: flex;
                align-items: center;
                font-size: 1.8rem;
                font-weight: 600;
                margin-bottom: 1rem;
            }
            .case-header .separator {
                margin: 0 0.8rem;
                font-weight: 300;
                font-size: 1.4rem;
                opacity: 0.6;
            }
        </style>
    """,
        unsafe_allow_html=True,
    )

    col_title, _, col_data = st.columns(
        [3, 1, 3], width="stretch", vertical_alignment="center"
    )

    with col_title:
        st.markdown(
            """
            <div class="case-header">
                <span>Case Review</span>
                <span class="separator">›</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_data:
        dataset_choice = st.selectbox(
            "Dataset",
            list(dataset_names.keys()),
            label_visibility="collapsed",
            key="dataset_selector",
            on_change=on_dataset_change,
            args=(
                dataset_names,
                background_size,
            ),
        )

    return dataset_choice
