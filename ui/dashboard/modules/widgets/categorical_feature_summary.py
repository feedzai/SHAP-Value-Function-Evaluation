import pandas as pd
import streamlit as st


def render(case: dict):
    """
    Renders a summary table for categorical features, prioritized by average risk.
    """

    ref_df_cat = st.session_state.get("ref_df_cat")
    ref_scores = st.session_state.get("ref_df_scores")

    if ref_df_cat is None or ref_df_cat.empty:
        st.warning("Reference categorical data is required for this analysis.")
        return

    current_cat_features = case.get("categorical_features", {})
    if not current_cat_features:
        st.info("No categorical features available for this case.")
        return

    total_ref_cases = len(ref_df_cat)
    OBSERVATION_THRESHOLD = 10
    TOP_N_ROWS = 5

    # --- Metrics for Current Case's Categorical Values ---
    data_rows = []
    overall_avg_risk_percent = ref_scores["label"].mean() * 100

    for feature_name, current_value in current_cat_features.items():
        subset = ref_df_cat[ref_df_cat[feature_name] == current_value]
        subset_scores = ref_scores.loc[subset.index]
        count = len(subset)

        # Observation threshold filter
        if count >= OBSERVATION_THRESHOLD:
            prevalence = (count / total_ref_cases) * 100
            avg_risk = subset_scores["label"].mean() * 100
            scores = (
                subset_scores["scores"]
                .apply(lambda x: x[st.session_state["model_id"]])
                .values
            )
            avg_score = scores.mean()
            risk_ratio = (
                avg_risk / overall_avg_risk_percent
                if overall_avg_risk_percent > 0
                else 1.0
            )

            # Store data for the current feature/value
            data_rows.append(
                {
                    "Variable": feature_name,
                    "Value": current_value,
                    "Prevalence": prevalence,
                    "Historical risk": avg_risk,
                    "Risk Ratio": risk_ratio,
                    "Average score": avg_score,
                    "__count": count,  # Internal count for 'Other' aggregation
                }
            )

    if not data_rows:
        st.info(
            f"No categorical variables above observation threshold of "
            f"{OBSERVATION_THRESHOLD}."
        )
        return

    df_summary = pd.DataFrame(data_rows)

    # 3. Sort by Historical Risk (descending)
    df_summary = df_summary.sort_values(by="Risk Ratio", ascending=False)

    # 4. Truncate to top 5 and create "Other"
    if len(df_summary) > TOP_N_ROWS:
        df_top = df_summary.head(TOP_N_ROWS).copy()
        df_other = df_summary.iloc[TOP_N_ROWS:].copy()

        # Calculate summary for "Other" category: Sum Prevalence, Average Risk/Score
        other_row = pd.Series(
            {
                "Variable": "Other",
                "Value": f"({len(df_other)} others)",
                "Prevalence": df_other["Prevalence"].mean(),
                "Historical risk": df_other["Historical risk"].mean(),
                "Risk Ratio": df_other["Risk Ratio"].mean(),
                "Average score": df_other["Average score"].mean(),
                "__count": df_other["__count"].sum(),
            }
        )

        df_final = pd.concat([df_top, other_row.to_frame().T], ignore_index=True)
    else:
        df_final = df_summary.copy()

    # Drop the internal count column
    df_final = df_final.drop(columns=["__count"])

    # Formatting for display
    df_final["Prevalence"] = df_final["Prevalence"].map("{:.1f}%".format)
    df_final["Historical risk"] = df_final["Historical risk"].map("{:.1f}%".format)
    df_final["Risk Ratio"] = df_final["Risk Ratio"].map("{:.2f}x".format)
    df_final["Average score"] = df_final["Average score"].map("{:.3f}".format)

    # Set 'Variable' as index to meet index bolding requirement
    df_final = df_final.set_index("Variable")

    # Render the dataframe
    st.dataframe(df_final, width="stretch")
