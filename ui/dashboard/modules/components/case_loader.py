# dashboard/modules/components/case_loader.py
from datetime import datetime

import random
from copy import deepcopy

import pandas as pd
import streamlit as st
from modules.api_client import fetch_background_cases, fetch_random_case

EXPLANATION_SUPPRESSION_PROB = 0.20   # 20% of cases have explanations disabled


def fetch_next(dataset_id: str):
    # Check if a new case should be fetched
    if st.button("Fetch Next Case ➜"):
        case = fetch_random_case(dataset_id)
        if case:
            # --- Make a mutable copy ---
            patched = deepcopy(case)

            # --- Random suppression of Explanations ---
            suppress = random.random() < EXPLANATION_SUPPRESSION_PROB
            model_id = case['model_id']

            if suppress:
                patched['explainer_id'] = None
                patched['numerical_feature_attributions'] = None
                patched['categorical_feature_attributions'] = None
                patched['baseline'] = None

            # --- Persist in session state ---
            st.session_state["current_case"] = patched
            st.session_state["case_counter"] += 1
            st.session_state["review_state"] = "decision_pending"
            st.session_state["analyst_decision"] = None
            st.session_state["analyst_confidence"] = None
            st.session_state["timestamp_start"] = datetime.now().isoformat()
            st.session_state["timestamp_end"] = None
            st.session_state["model_id"] = model_id
        else:
            st.error("Failed to fetch case data. Check the API connection.")
            st.session_state["current_case"] = None

    # Check and Render Dashboard Content
    case = st.session_state["current_case"]

    return case


def load_reference_data(dataset_id, size: int = 1000):
    # Check if reference data for this dataset is already loaded
    if (
        "ref_data_id" in st.session_state
        and st.session_state["ref_data_id"] == dataset_id
    ):
        return

    # Fetch "size" cases in a single API call
    with st.spinner("Fetching reference data..."):
        reference_cases = fetch_background_cases(dataset_id, size=size) or []

    # Process data, save it and reset review state
    if reference_cases:
        st.session_state["ref_cases"] = reference_cases
        st.session_state["ref_data_id"] = dataset_id

        # Extract and Flatten Data into required Tables
        scores_data = []
        cat_data = []
        num_data = []

        for case in reference_cases:
            instance_id = case.get("instance_id")

            # Table 1: Scores and Labels
            scores_data.append(
                {
                    "instance_id": instance_id,
                    "scores": case.get("scores"),
                    "label": case.get("true_outcome"),
                }
            )

            # Table 2: Categorical Features
            if case.get("categorical_features"):
                cat_row = {"instance_id": instance_id}
                cat_row.update(case["categorical_features"])
                cat_data.append(cat_row)

            # Table 3: Numerical Features
            if case.get("numerical_features"):
                num_row = {"instance_id": instance_id}
                num_row.update(case["numerical_features"])
                num_data.append(num_row)

        # Create DataFrames and Set Index
        df_scores = pd.DataFrame(scores_data)
        if not df_scores.empty:
            df_scores.set_index("instance_id", inplace=True)
            st.session_state["ref_df_scores"] = df_scores

        df_cat = pd.DataFrame(cat_data)
        if not df_cat.empty:
            df_cat.set_index("instance_id", inplace=True)
            st.session_state["ref_df_cat"] = df_cat

        df_num = pd.DataFrame(num_data)
        if not df_num.empty:
            df_num.set_index("instance_id", inplace=True)
            st.session_state["ref_df_num"] = df_num

        # Reset the current review state
        st.session_state["current_case"] = None
        st.session_state["case_counter"] = 0

    else:
        st.error("Failed to load reference cases.")


def ensure_data_loaded(dataset_id: str, size: int = 1000):
    load_reference_data(dataset_id, size=size)
