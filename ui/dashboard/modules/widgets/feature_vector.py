import numpy as np
import pandas as pd
import streamlit as st


def render(case: dict):
    all_features = {}
    all_features.update(case.get("numerical_features", {}))
    all_features.update(case.get("categorical_features", {}))

    # Filter out empty features and sort keys
    sorted_features = {k: v for k, v in sorted(all_features.items()) if v is not None}

    if not sorted_features:
        st.info("No feature data available for this case.")
        return

    # --- Prepare Data for DataFrame ---
    data = []

    for feature, value in sorted_features.items():
        if isinstance(value, (int, float, np.floating, np.integer)):
            if value % 1 == 0:
                display_value = str(int(value))
            else:
                display_value = f"{value:.2f}"
        else:
            display_value = str(value)

        data.append((feature, display_value))

    df = pd.DataFrame(data, columns=["Feature", "Value"])
    df = df.set_index("Feature")

    # Render the transposed DataFrame for a compact, horizontal key-value display
    st.dataframe(df.T, width="stretch", hide_index=True)
