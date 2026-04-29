import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


def render(case):
    df_scores = st.session_state.get("ref_df_scores")
    current_score = case.get("model_score")

    if df_scores is None or df_scores.empty:
        st.warning("Reference scores data not loaded or is empty.")
        return

    # Create a copy and reset index for plotting
    df_plot = df_scores.reset_index().copy()

    model_id = st.session_state["model_id"]
    df_plot["scores"] = df_plot["scores"].apply(lambda x: x[model_id]).values
    # Calculate beeswarm-style jitter based on score density
    df_plot["score_bin"] = pd.cut(
        df_plot["scores"], bins=30, labels=False, include_lowest=True
    )
    df_plot["rank_in_bin"] = df_plot.groupby("score_bin").cumcount()

    max_density = df_plot["rank_in_bin"].max()

    def center_and_scale(series):
        return np.where(series % 2 == 0, series // 2, -(series + 1) // 2)

    df_plot["density_jitter"] = df_plot.groupby("score_bin")["rank_in_bin"].transform(
        center_and_scale
    )
    df_plot["density_jitter"] *= 1.0 / max_density

    # Define score ranges for consistent scale
    y_max = df_plot["density_jitter"].abs().max()

    score_min, score_max = df_plot["scores"].min(), df_plot["scores"].max()
    score_range = score_max - score_min

    if score_range == 0:
        st.info("All reference scores are the same.")
        return

    # Base Chart: Scatter plot of reference cases
    base_chart = (
        alt.Chart(df_plot)
        .mark_circle(size=60, opacity=0.8)
        .encode(
            x=alt.X(
                "scores", title=None, scale=alt.Scale(domain=[score_min, score_max])
            ),
            y=alt.Y(
                "density_jitter",
                title="",
                axis=None,
                scale=alt.Scale(domain=[-y_max, y_max]),
            ),
            color=alt.Color(
                "label:N",
                legend=None,
                scale=alt.Scale(domain=[0, 1], range=["green", "red"]),
            ),
            tooltip=[alt.Tooltip("scores", format=".4f"), "label"],
        )
    )

    # Current Case: Large Black Dot
    current_case_df = pd.DataFrame({"score": [current_score], "jitter": [0.0]})

    current_case_layer = (
        alt.Chart(current_case_df)
        .mark_circle(size=300, color="black", stroke="red", strokeWidth=2)
        .encode(
            x="score",
            y="jitter",
            tooltip=[alt.Tooltip("score", format=".4f", title="Case Score")],
        )
    )

    # Combine the layers and render
    final_chart = (
        (base_chart + current_case_layer)
        .properties(
            height=250,
        )
        .interactive()
    )

    st.altair_chart(final_chart, width="stretch")

    # --- PERCENTILE CALCULATION AND SUMMARY ---

    # Determine the Local Risk Bucket
    BUCKET_LOOKBACK_M = 1
    idx_nearest = np.argmin(np.abs(df_plot["scores"].values - current_score))
    current_score_bin_label = df_plot.loc[idx_nearest, "score_bin"]

    # Define the range of bins to include: [current_bin - m, current_bin + m]
    min_bin = max(0, current_score_bin_label - BUCKET_LOOKBACK_M)
    max_bin = min(
        max(df_plot["score_bin"]) - 1, current_score_bin_label + BUCKET_LOOKBACK_M
    )

    # Filter data
    df_local = df_plot[
        (df_plot["score_bin"] >= min_bin) & (df_plot["score_bin"] <= max_bin)
    ]

    total_local = len(df_local)
    avg_risk_local = 0.0
    rank_color = "red"

    # Calculate Average Local Risk
    if total_local > 0:
        avg_risk_local = df_local["label"].mean() * 100

        if avg_risk_local <= 25:
            rank_color = "green"
        elif avg_risk_local <= 50:
            rank_color = "orange"

    summary_content = f"""
    <div style="text-align: center; margin-bottom: 15px; font-size: 1.1em;">
        Avg. Risk for Similar Scores:
        <span style="color:{rank_color}; font-weight:bold;">{avg_risk_local:.1f}%</span> 
    </div>
    """
    st.markdown(summary_content, unsafe_allow_html=True)
