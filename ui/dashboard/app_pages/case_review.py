# pages/case_review.py
import streamlit as st
from modules.api_client import fetch_datasets
from modules.components import (
    analyst_profile,
    case_loader,
    header_selector,
    task_description,
)
from modules.widgets import (
    case_id,
    categorical_feature_summary,
    decision_box,
    feature_vector,
    numerical_feature_explorer,
    reason_codes,
    risk_score,
    score_distribution,
    shapley_waterfall,
)


def render_case_data(case):
    if case:
        # --- TOP ROW: SUMMARY BOXES (Case ID, Score, Decision) ---
        col_id, col_score, col_decision = st.columns([1, 1, 2])

        # --- Box 1: Case ID ---
        with col_id:
            with st.container(border=True, height="stretch"):
                case_id.render(case)

        # --- Box 2: Risk Score ---
        with col_score:
            with st.container(border=True, height="stretch"):
                risk_score.render(case)

        # --- Box 3: Decision & Confidence (Multi-step workflow) ---
        with col_decision:
            with st.container(border=True, height="stretch"):
                decision_box.render()


def render_model_explanation(case):
    if case:
        with st.expander("Risk Insights", expanded=True):
            col1, col2, col3 = st.columns([1, 1, 1])

            # --- Custom title markdown for smaller text (using h5 or h4 tags) ---
            title_style = '<h5 style="margin-top: 0px;">{}</h5>'

            with col1:
                with st.container(border=True, height="stretch"):
                    st.markdown(
                        title_style.format("Score Distribution"), unsafe_allow_html=True
                    )
                    score_distribution.render(case)

            with col2:
                with st.container(border=True, height="stretch"):
                    st.markdown(
                        title_style.format("Shapley Values"), unsafe_allow_html=True
                    )
                    if case["explainer_id"] is None:
                        st.info("Chart is suppressed for this case.")
                    else:
                        shapley_waterfall.render(case)

            with col3:
                with st.container(border=True, height="stretch"):
                    st.markdown(
                        title_style.format("Reason Codes"), unsafe_allow_html=True
                    )
                    if case["explainer_id"] is None:
                        st.info("Reason codes suppressed for this case.")
                    else:
                        reason_codes.render(case)


def render_data_explorer(case):
    if case:
        with st.expander("Data Explorer", expanded=True):
            # --- Custom title markdown for smaller text (using h5 or h4 tags) ---
            title_style = '<h5 style="margin-top: 0px;">{}</h5>'

            st.markdown(title_style.format("Case Variables"), unsafe_allow_html=True)
            feature_vector.render(case)

            col1, col2 = st.columns([2, 3])

            with col1:
                with st.container(border=True, height="stretch"):
                    st.markdown(
                        title_style.format("Numerical Analysis"), unsafe_allow_html=True
                    )
                    numerical_feature_explorer.render(case)

            with col2:
                with st.container(border=True, height="stretch"):
                    st.markdown(
                        title_style.format("Categorical Risk Factors"),
                        unsafe_allow_html=True,
                    )
                    categorical_feature_summary.render(case)


# --- MAIN PAGE FUNCTION ---
def render_case_review_page():
    dataset_names = fetch_datasets()
    if not dataset_names:
        st.error(
            "Could not load dataset names. Please check the API server connection."
        )
        return

    # 1. Header and Dataset Selector
    dataset_choice = header_selector.render(dataset_names, background_size=1000)
    dataset_id = dataset_names.get(dataset_choice)
    if dataset_id:
        case_loader.ensure_data_loaded(dataset_id, size=1000)

    # 2. Analyst Profile
    analyst = analyst_profile.render()  # noqa: F841

    if analyst == {}:
        task_description.render(dataset_id)
        st.stop()

    # 3. Fetch Case: Runs if render analyst_profile didn't st.stop() execution.
    case = None
    if dataset_id:
        case = case_loader.fetch_next(dataset_id)

    # 4. Render View Components: If analyst_profile didn't st.stop() execution.
    st.html(
        """
        <style>
        details > summary {
            background-color: #0E61EE !important;
            color: white !important;
        }
        </style>
        """
    )
    render_case_data(case)
    render_model_explanation(case)
    render_data_explorer(case)

    # 5. Compact Task Summary Below Widgets
    with st.expander("Task Summary", expanded=False):
        task_description.render(dataset_id)
