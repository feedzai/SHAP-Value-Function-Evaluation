import altair as alt
import pandas as pd
import streamlit as st


def render(case: dict):
    # Initialization and Data Check
    ref_df_num = st.session_state.get("ref_df_num")
    ref_df_cat = st.session_state.get("ref_df_cat")
    ref_df_scores = st.session_state.get("ref_df_scores")

    if ref_df_num is None or ref_df_num.empty or ref_df_cat is None or ref_df_cat.empty:
        st.warning("Numerical and categorical data are required for the explorer.")
        return

    if ref_df_scores is None or ref_df_scores.empty:
        st.warning("Reference scores/labels data is required for the explorer.")
        return

    current_num_features = case.get("numerical_features", {})
    current_cat_features = case.get("categorical_features", {})

    if not current_num_features:
        st.info("No numerical features available for this case.")
        return

    # Combine reference dataframes (assuming they share the index/case ID structure)
    try:
        df_ref = pd.merge(ref_df_num, ref_df_cat, on="instance_id", how="inner")
        # Merge with scores to get labels
        df_labels = ref_df_scores[["label"]]
        df_ref = pd.merge(df_ref, df_labels, on="instance_id", how="inner")
    except Exception as e:
        st.error(f"Error merging reference dataframes. Details: {e}")
        return

    # --- Dropdown Setup ---

    # Numerical features must exist as columns in the combined reference data
    num_feature_keys = [k for k in current_num_features.keys() if k in df_ref.columns]

    # Categorical features must exist as columns in the combined reference data
    cat_feature_keys = ["All"] + [
        k for k in current_cat_features.keys() if k in df_ref.columns
    ]

    if not num_feature_keys:
        st.info("No numerical features found in the reference data for plotting.")
        return

    # Create the selector columns
    col_select_num, col_select_cat = st.columns(2)

    with col_select_num:
        selected_num_feature = st.selectbox(
            "Variable", options=num_feature_keys, key="numerical_feature_select"
        )

    with col_select_cat:
        selected_cat_filter = st.selectbox(
            "Filter", options=cat_feature_keys, key="categorical_filter_select"
        )

    # --- Filtering Logic ---
    df_plot = df_ref.copy()

    filter_description = ""

    if selected_cat_filter != "All" and selected_cat_filter in current_cat_features:
        filter_value = current_cat_features[selected_cat_filter]
        df_plot = df_plot[df_plot[selected_cat_filter] == filter_value]

        filter_description = f" where {selected_cat_filter} is {filter_value}"

    # Check for empty dataframe after filtering
    if df_plot.empty:
        st.info("No reference observations match the selected filter.")
        return

    # Ensure the numerical feature column is ready and get current case value
    df_plot = df_plot[[selected_num_feature, "label"]].dropna()
    current_case_value = current_num_features.get(selected_num_feature)

    # --- Plot Generation ---

    # A. KDE Density Plot (separated by label)
    density_chart = (
        alt.Chart(df_plot.reset_index())
        .transform_density(
            density=selected_num_feature,
            as_=[selected_num_feature, "density"],
            groupby=["label"],
            steps=200,
        )
        .mark_area(opacity=0.5)
        .encode(
            x=alt.X(
                f"{selected_num_feature}:Q",
                title=f"Distribution of {selected_num_feature}{filter_description}",
                axis=alt.Axis(titleFontWeight="bold"),
            ),
            y=alt.Y("density:Q", title="Density", axis=None),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(domain=[0, 1], range=["green", "red"]),
                legend=None,
            ),
            tooltip=[
                f"{selected_num_feature}:Q",
                alt.Tooltip("density:Q", format=".3f"),
                "label:N",
            ],
        )
        .properties(height=200)
    )

    # B. Current Case Marker (Vertical Rule)
    current_value_df = pd.DataFrame({selected_num_feature: [current_case_value]})

    current_case_rule = (
        alt.Chart(current_value_df)
        .mark_rule(color="red", strokeWidth=3, strokeDash=[5, 5])
        .encode(
            x=alt.X(selected_num_feature),
            tooltip=[
                alt.Tooltip(
                    selected_num_feature, title="Current Case Value", format=".4f"
                )
            ],
        )
    )

    # C. Combine and Render
    final_chart = (density_chart + current_case_rule).properties()

    st.altair_chart(final_chart, width="stretch")
