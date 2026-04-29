# dashboard/modules/widgets/risk_score.py
import numpy as np
import streamlit as st


def render(case: dict):
    st.subheader(":material/gpp_maybe: &nbsp; Risk Score")

    score_value = round(case["model_score"], 3)
    df_scores = st.session_state.get("ref_df_scores")

    percentile_rank = 0.5
    rank_color = "black"

    # Calculate percentile rank only if reference data is available
    if df_scores is not None and not df_scores.empty:
        all_scores = (
            df_scores["scores"].apply(lambda x: x[st.session_state["model_id"]]).values
        )
        total_scores = len(all_scores)

        if total_scores > 0:
            scores_below = np.sum(all_scores < score_value)
            scores_equal = np.sum(all_scores == score_value)

            # Calculate percentile rank
            percentile_rank = (scores_below + 0.5 * scores_equal) / total_scores * 100

            if percentile_rank >= 95:
                rank_color = "#ff4b4b"
            elif percentile_rank >= 90:
                rank_color = "#ffaa00"
            elif percentile_rank <= 25:
                rank_color = "green"

    # Create two columns for side-by-side display
    col_score, col_percentile = st.columns(2)

    with col_score:
        st.metric("Model Score", score_value)

    with col_percentile:
        colored_percentile = (
            f"<span style='color:{rank_color};'>{percentile_rank:.1f}%</span>"
        )

        tooltip_text = (
            f"Percentage of historic cases that have a score lower than {score_value}."
        )

        st.markdown(
            f"""
            <div style='
                display: flex;
                align-items: center;
                gap: 5px;
                font-size: 14px;
                margin-top: 1px;
            '>
                <div>Percentile</div>
                <div title='{tooltip_text}'>&#x24D8;</div> 
            </div>
            <div style='font-size: 2.2rem; margin-top: -4px;'>{colored_percentile}</div>
            """,
            unsafe_allow_html=True,
        )
