import altair as alt
import pandas as pd
import streamlit as st


def render(case, feature_n=4):
    # Load Reference Data for Baseline Score and Y-Axis Scale
    df_scores = st.session_state.get("ref_df_scores")
    if df_scores is None or df_scores.empty:
        st.info("Reference scores data not loaded. Cannot calculate baseline or scale.")
    else:
        baseline_score = case.get("baseline", 0.0)
        scores = (
            df_scores["scores"].apply(lambda x: x[st.session_state["model_id"]]).values
        )
        score_min = scores.min()
        score_max = scores.max()
        buffer = (score_max - score_min) * 0.05
        Y_AXIS_DOMAIN = [score_min - buffer, score_max + buffer]

    numerical = case.get("numerical_feature_attributions", {})
    categorical = case.get("categorical_feature_attributions", {})

    all_attributions = []

    # Add features
    for k, v in numerical.items():
        all_attributions.append({"feature": k, "contribution": v})

    for k, v in categorical.items():
        all_attributions.append({"feature": k, "contribution": v})

    if not all_attributions:
        st.info("No feature attributions available for this case.")
        return

    # Sort and select top N features by absolute contribution
    df_attrs = pd.DataFrame(all_attributions)

    # --- SCALING LOGIC ---

    expected_contribution = case.get("model_score", 0.0) - baseline_score

    # The current sum of raw SHAP contributions
    sum_of_raw_attributions = df_attrs["contribution"].sum()

    scaling_factor = 1.0
    if abs(sum_of_raw_attributions) > 1e-6:
        scaling_factor = expected_contribution / sum_of_raw_attributions

    # Apply the scaling factor to all contributions
    df_attrs["contribution"] = df_attrs["contribution"] * scaling_factor

    # Recalculate absolute contribution using the scaled values
    df_attrs["abs_contribution"] = df_attrs["contribution"].abs()

    # --- SCALING LOGIC ENDS ---

    # Get the top N features based on absolute contribution
    df_top = df_attrs.nlargest(feature_n, "abs_contribution").copy()
    top_features = set(df_top["feature"].tolist())

    # Calculate "Other" contribution (sum of all unselected features)
    df_other = df_attrs[~df_attrs["feature"].isin(top_features)].copy()
    other_contribution = df_other["contribution"].sum()

    # Create the "Other" row
    other_row = {
        "feature": "other",
        "contribution": other_contribution,
        "abs_contribution": abs(other_contribution),
    }

    # Append the "Other" row to the working DataFrame
    df_top = pd.concat([df_top, pd.Series(other_row).to_frame().T], ignore_index=True)

    if df_top.empty:
        st.info("No substantial feature attributions found.")
        return

    # Calculate Waterfall positions (Cumulative Impact)
    df_top["start_value"] = 0.0
    df_top["end_value"] = 0.0

    # Calculate cumulative sum
    cumulative_sum = baseline_score
    max_cumulative_value = baseline_score
    min_cumulative_value = baseline_score

    for index, row in df_top.iterrows():
        df_top.loc[index, "start_value"] = cumulative_sum
        cumulative_sum += row["contribution"]
        df_top.loc[index, "end_value"] = cumulative_sum

        # Update overall min/max cumulative values
        max_cumulative_value = max(
            max_cumulative_value,
            df_top.loc[index, "start_value"],
            df_top.loc[index, "end_value"],
        )
        min_cumulative_value = min(
            min_cumulative_value,
            df_top.loc[index, "start_value"],
            df_top.loc[index, "end_value"],
        )

    # Y-AXIS DOMAIN ADJUSTMENT
    if min_cumulative_value < Y_AXIS_DOMAIN[0]:
        Y_AXIS_DOMAIN[0] = min_cumulative_value - buffer
    if max_cumulative_value > Y_AXIS_DOMAIN[1]:
        Y_AXIS_DOMAIN[1] = max_cumulative_value + buffer

    # Calculate color direction (Red=Risk/Positive, Green=No Risk/Negative)
    df_top["direction"] = df_top["contribution"].apply(
        lambda x: "Positive" if x >= 0 else "Negative"
    )

    # Convert feature names to strings for Altair axis
    df_top["feature"] = df_top["feature"].astype(str)

    # Altair needs this ordered list for consistent categorical axis rendering
    feature_order = df_top["feature"].tolist()

    # Create the Altair Waterfall Chart
    color_scale = alt.Scale(domain=["Positive", "Negative"], range=["red", "green"])

    y_title_text = "Cummulative Risk Effect"

    y_encoding = alt.Y(
        "start_value:Q",
        title=y_title_text,
        scale=alt.Scale(domain=Y_AXIS_DOMAIN),
        axis=alt.Axis(labels=False, titleFontWeight="bold", titleAnchor="middle"),
    )

    # X-Encoding: Defines the categorical order using the explicit list
    x_encoding = alt.X(
        "feature:N",
        sort=feature_order,
        title=None,
        axis=alt.Axis(labels=False, title=None, labelAngle=-90, labelLimit=0),
    )

    color_encoding = alt.Color("direction:N", scale=color_scale, legend=None)

    # Bars representing the step change
    waterfall_bars = (
        alt.Chart(df_top)
        .mark_bar(cornerRadius=5)
        .encode(
            x=x_encoding,
            y=y_encoding,
            y2="end_value:Q",
            color=color_encoding,
            tooltip=[
                alt.Tooltip("feature", title="Feature"),
                alt.Tooltip("contribution", title="Impact", format="+.4f"),
            ],
        )
    )

    # Baseline Marker (Dotted line across the chart at the average score)
    baseline_df = pd.DataFrame({"baseline": [baseline_score]})
    baseline_line = (
        alt.Chart(baseline_df)
        .mark_rule(strokeDash=[5, 5], color="gray", strokeWidth=2)
        .encode(
            y=alt.Y("baseline:Q", scale=alt.Scale(domain=Y_AXIS_DOMAIN)),
            tooltip=[alt.Tooltip("baseline", title="Baseline Score", format=".4f")],
        )
    )

    final_chart = (
        (baseline_line + waterfall_bars)
        .properties(
            height=250,
        )
        .interactive()
    )

    st.altair_chart(final_chart, width="stretch")

    # COLORED FEATURE SUMMARY

    # Start the string with the Baseline value for context
    summary_parts = ['<span style="font-weight:bold;">Baseline</span>']
    summary_parts.append("+")

    num_features = len(df_top)

    # Iterate through features and assign color based on direction
    for i, (_, row) in enumerate(df_top.iterrows()):
        feature = row["feature"]
        direction = row["direction"]

        # Use HTML color tags for Streamlit Markdown color rendering
        if direction == "Positive":
            color_tag = f'<span style="color:red; font-weight:bold;">{feature}</span>'
        else:
            color_tag = f'<span style="color:green; font-weight:bold;">{feature}</span>'

        summary_parts.append(color_tag)

        # Add "+" after every item EXCEPT the last one
        if i < num_features - 1:
            summary_parts.append("+")

    summary_parts.append(
        f'<span style="font-weight:bold;">= {case.get("model_score", 0.0):.3f}</span>'
    )

    # Join the summary parts with a space
    summary_content = " ".join(summary_parts)

    # Wrap in a centered div with an upward vertical offset (negative margin)
    summary_markdown = f"""
    <div style="text-align: center; margin-top: -20px; margin-bottom: 15px; font-size: 1.1em;">
        {summary_content}
    </div>
    """

    # Use st.markdown to render the colored text
    st.markdown(summary_markdown, unsafe_allow_html=True)
