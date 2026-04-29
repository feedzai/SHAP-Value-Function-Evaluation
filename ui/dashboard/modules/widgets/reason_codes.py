# dashboard/modules/widgets/reason_codes.py
import pandas as pd
import streamlit as st

RISK_THRESHOLDS = {
    "red": 0.25,  # High Risk
    "orange": 0.10,  # Medium Risk
    "yellow": 0.05,  # Low Risk
}

CAT_RISK_MULTIPLIERS = {
    10.0: "an extreme risk factor",
    5.0: "a high risk factor",
    2.0: "a risk factor",
}

BASE_TEMPLATE = """
    <div style="text-align: center; padding: 20px; margin-bottom: 10px; color: {text_color}; border: 2px solid {border_color}; border-radius: 8px;">
        <span style="font-weight:bold;">{code}</span>
    </div>
    """

COLOR_STYLES = {
    "red": {"text_color": "red", "border_color": "red"},
    "orange": {"text_color": "orange", "border_color": "orange"},
    "yellow": {"text_color": "black", "border_color": "black"},
}

NO_RISK_MESSAGE = """
    <div style="text-align: center; padding: 20px; color: green; border: 2px solid green; border-radius: 8px;">
        <span style="font-weight:bold;">No Major Risk Factors Identified.</span>
    </div>
    """

# --- UTILITY FUNCTIONS ---


def _get_numerical_percentiles():
    """
    Calculates the 5th, 10th, 20th, 80th, 90th, and 95th percentiles
    for all numerical features in the reference dataset.
    """
    df_num = st.session_state.get("ref_df_num")
    if df_num is None or df_num.empty:
        return {}

    # Define the required percentiles (quantiles)
    quantiles = [0.05, 0.1, 0.2, 0.8, 0.9, 0.95]

    # Calculate quantiles for all numerical columns
    percentiles = df_num.quantile(quantiles).to_dict()

    return percentiles


def _get_categorical_risk_ratios(case: dict):
    """
    Calculates the Historical Risk Ratio ONLY for observed categorical feature values
    in the reference data against the overall dataset risk.
    """
    ref_df_cat = st.session_state.get("ref_df_cat")
    ref_scores = st.session_state.get("ref_df_scores")
    current_cat_features = case.get("categorical_features", {})

    if (
        ref_df_cat is None
        or ref_df_cat.empty
        or ref_scores is None
        or ref_scores.empty
        or not current_cat_features
    ):
        return {}

    overall_avg_risk_percent = ref_scores["label"].mean() * 100
    if overall_avg_risk_percent <= 1e-6:
        return {}

    risk_ratios = {}

    # NEW: Iterate ONLY over the features present in the current case
    for feature_name, current_value in current_cat_features.items():
        # Find all reference cases with this specific feature value
        subset = ref_df_cat[ref_df_cat[feature_name] == current_value]

        # Check if the feature value exists in the reference data
        if not subset.empty:
            subset_indices = subset.index
            subset_scores = ref_scores.loc[subset_indices]

            # Calculate average risk for this specific value
            avg_risk = subset_scores["label"].mean() * 100

            # Calculate the Risk Ratio
            risk_ratio = avg_risk / overall_avg_risk_percent
            risk_ratios[(feature_name, current_value)] = risk_ratio

    return risk_ratios


def _get_custom_reason_code(feature, value, feature_type, percentiles, cat_ratios):
    """
    Customizes the reason code text based on the feature's value relative to
    the numerical percentiles. Only runs logic for numerical features.
    """
    display_value = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
    default_code = f"{feature} has a value of {display_value}."

    if feature_type == "categorical":
        ratio = cat_ratios.get((feature, value), 0.0)

        for multiplier, description in sorted(
            CAT_RISK_MULTIPLIERS.items(), reverse=True
        ):
            if ratio >= multiplier:
                return f"{feature} with {value} is {description}."

        return default_code

    feature_percentiles = percentiles.get(feature)
    if not feature_percentiles:
        return default_code

    # RULE 1: Extreme Outlier (> 95% or < 5%)
    if value > feature_percentiles.get(
        0.95, float("inf")
    ) or value < feature_percentiles.get(0.05, float("-inf")):
        return f"{feature} value of {display_value} is an outlier."

    # RULE 2: Significant Deviation (> 90% or < 10%)
    if value > feature_percentiles.get(0.90, float("inf")):
        return f"{feature} value of {display_value} is abnormally high."
    elif value < feature_percentiles.get(0.10, float("-inf")):
        return f"{feature} value of {display_value} is abnormally low."

    # RULE 3: Notable Tendency (> 80% or < 20%)
    if value > feature_percentiles.get(0.80, float("inf")):
        return f"{feature} value of {display_value} is high."
    elif value < feature_percentiles.get(0.20, float("-inf")):
        return f"{feature} value of {display_value} is low."

    # If no rule is met, return the default code
    return default_code


def render(case: dict):
    # Load Reference Data for Baseline Score and Max Score
    df_scores = st.session_state.get("ref_df_scores")
    if df_scores is None or df_scores.empty:
        st.info(
            "Reference scores data not loaded. Cannot compute risk-based reason codes."
        )
        return

    baseline_score = case.get("baseline", 0.0)
    scores = df_scores["scores"].apply(lambda x: x[st.session_state["model_id"]]).values
    score_max = scores.max()
    score_min = scores.min()
    max_risk_distance = score_max - score_min

    if max_risk_distance <= 1e-6:
        st.info(
            "No risk margin identified (Max Score is at or below Baseline). Cannot compute risk-based reason codes."
        )
        return

    # Calculate Numerical Percentiles and Categorical stats
    numerical_percentiles = _get_numerical_percentiles()
    categorical_ratios = _get_categorical_risk_ratios(case)

    # Extract all feature attributions and values, including feature type
    numerical_attrs = case.get("numerical_feature_attributions", {})
    categorical_attrs = case.get("categorical_feature_attributions", {})
    numerical_values = case.get("numerical_features", {})
    categorical_values = case.get("categorical_features", {})

    all_attributions = []

    # Combine numerical features
    for k, v in numerical_attrs.items():
        all_attributions.append(
            {
                "feature": k,
                "contribution": v,
                "value": numerical_values.get(k),
                "type": "numerical",
            }
        )

    # Combine categorical features
    for k, v in categorical_attrs.items():
        all_attributions.append(
            {
                "feature": k,
                "contribution": v,
                "value": categorical_values.get(k),
                "type": "categorical",
            }
        )

    if not all_attributions:
        st.info("No feature attributions available for this case.")
        return

    df_attrs = pd.DataFrame(all_attributions)

    # --- Scaling Logic ---
    expected_contribution = case.get("model_score", 0.0) - baseline_score
    sum_of_raw_attributions = df_attrs["contribution"].sum()

    scaling_factor = 1.0
    if abs(sum_of_raw_attributions) > 1e-6:
        scaling_factor = expected_contribution / sum_of_raw_attributions

    df_attrs["scaled_contribution"] = df_attrs["contribution"] * scaling_factor
    # --- Scaling Logic Ends ---

    # Check for positive contributions that drive risk only.
    df_positive_attributions = df_attrs[df_attrs["scaled_contribution"] > 0.0].copy()

    if df_positive_attributions.empty:
        st.markdown(NO_RISK_MESSAGE, unsafe_allow_html=True)
        return

    # Compute the Risk Ratio
    df_positive_attributions["risk_ratio"] = (
        df_positive_attributions["scaled_contribution"] / max_risk_distance
    )

    # Restrict to top-4 absolute SHAP contributions across ALL features
    df_abs_sorted = df_attrs.reindex(
        df_attrs["scaled_contribution"].abs().sort_values(ascending=False).index
    )
    top4_abs_features = set(df_abs_sorted.head(4)["feature"])

    reason_codes_list = []

    # Check the ratio against thresholds and activate codes
    for _, row in df_positive_attributions.sort_values(
        by="risk_ratio", ascending=False
    ).iterrows():
        feature = row["feature"]
        value = row["value"]

        if feature not in top4_abs_features:
            continue

        # Get the custom reason code message
        custom_code = _get_custom_reason_code(
            feature, value, row["type"], numerical_percentiles, categorical_ratios
        )

        # Determine the color/risk level based on contribution ratio
        color = None
        if row["risk_ratio"] >= RISK_THRESHOLDS["red"]:
            color = "red"
        elif row["risk_ratio"] >= RISK_THRESHOLDS["orange"]:
            color = "orange"
        elif row["risk_ratio"] >= RISK_THRESHOLDS["yellow"]:
            color = "yellow"

        # If a threshold is met, generate the reason code
        if color:
            reason_codes_list.append(
                {
                    "code": custom_code,  # Use the custom message here
                    "color": color,
                    "ratio": row["risk_ratio"],
                }
            )

    # Render the reason codes
    if not reason_codes_list:
        st.markdown(NO_RISK_MESSAGE, unsafe_allow_html=True)
    else:
        for item in reason_codes_list[:4]:
            color_key = item["color"]
            styles = COLOR_STYLES.get(color_key, {})

            if styles:
                st.markdown(
                    BASE_TEMPLATE.format(
                        code=item["code"],
                        text_color=styles["text_color"],
                        border_color=styles["border_color"],
                    ),
                    unsafe_allow_html=True,
                )
