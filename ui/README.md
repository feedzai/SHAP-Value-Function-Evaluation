# Risk Analysis Dashboard

A Streamlit dashboard used as the front-end for an A/B testing user study. Analysts review risk cases one at a time, make binary decisions (Risk / No Risk), and provide feedback on their confidence and the clarity of the Shapley explanations shown. Responses are submitted to a backend API for collection.

## Overview

The dashboard presents analysts with individual risk cases from a selected dataset. Each case displays the model's risk score, feature values, and (when not suppressed) Shapley value explanations. The analyst then records a decision, confidence level, and explanation clarity rating. 20% of cases randomly suppress explanations to serve as a control group.

This frontend assumes a running backend API at `http://localhost:8000` that serves:
  - `GET /datasets` — list of available datasets
  - `GET /case/{dataset_id}` — random case with features, scores, and attributions
  - `GET /background/{dataset_id}?size=N` — reference cases for score distributions
  - `POST /case-response` — submit analyst responses

## Pages

### Home
A landing page with a welcome message and link to help resources.

### Case Review
The main workflow page. The analyst flow is:

1. Select a dataset from the header dropdown
2. Fill in the analyst profile (alias, domain knowledge, ML knowledge, Shapley understanding)
3. Click "Fetch Next Case" to load a case
4. Review the case data and explanations
5. Make a decision (Risk / No Risk)
6. Rate confidence (Weak / Moderate / Strong)
7. Rate explanation clarity (Clear / Confusing)
8. Response is auto-submitted to the API, then fetch the next case

## Widgets

| Widget | Description |
|--------|-------------|
| Case ID | Displays instance ID and session case counter |
| Risk Score | Model score with percentile rank against reference data |
| Decision Box | Multi-step workflow: decision → confidence → clarity → submit |
| Score Distribution | Beeswarm plot of reference scores colored by label, with current case highlighted |
| Shapley Waterfall | Waterfall chart of top feature contributions from baseline to model score |
| Reason Codes | Color-coded risk factor alerts derived from Shapley attributions and feature percentiles |
| Feature Vector | Horizontal table of all feature values for the current case |
| Numerical Feature Explorer | KDE density plot of a selected numerical feature split by label, with categorical filtering |
| Categorical Feature Summary | Table of categorical feature values ranked by historical risk ratio |

## Supported Datasets

The dashboard includes task descriptions and feature definitions for:
- GermanCredit (UCI German Credit Risk)
- MaternalRisk (Maternal Health Risk Assessment)
- HELOC (FICO HELOC Creditworthiness)
- Adult (UCI Census Income)

Dataset metadata is defined in `modules/dataset_metadata.py`.

## Project Structure

```
ui/
└── dashboard/
    ├── app.py                              # Streamlit entry point and sidebar navigation
    ├── .streamlit/
    │   └── config.toml                     # Theme configuration
    ├── app_pages/
    │   ├── home.py                         # Landing page
    │   └── case_review.py                  # Main case review page
    ├── modules/
    │   ├── api_client.py                   # Backend API client (fetch cases, submit responses)
    │   ├── dataset_metadata.py             # Per-dataset task descriptions and feature definitions
    │   ├── components/
    │   │   ├── analyst_profile.py          # Analyst profile form (alias, expertise)
    │   │   ├── case_loader.py              # Case fetching, reference data loading, explanation suppression
    │   │   ├── header_selector.py          # Dataset selector header
    │   │   └── task_description.py         # Dataset-specific task overview
    │   └── widgets/
    │       ├── case_id.py                  # Case ID and counter display
    │       ├── categorical_feature_summary.py  # Categorical risk ratio table
    │       ├── decision_box.py             # Decision → confidence → clarity workflow
    │       ├── feature_vector.py           # Full feature value table
    │       ├── numerical_feature_explorer.py   # KDE density plot with filtering
    │       ├── reason_codes.py             # Risk factor alerts from attributions
    │       ├── risk_score.py               # Score and percentile display
    │       ├── score_distribution.py       # Beeswarm reference score plot
    │       └── shapley_waterfall.py        # Waterfall attribution chart
    └── assets/
        ├── logo.png                        # Sidebar logo
        └── icon.png                        # Browser tab icon
```

## Configuration

The Streamlit theme is configured in `.streamlit/config.toml` with a blue primary color (`#0e61ee`), white background, and sans-serif font.

The API base URL is set in `modules/api_client.py` (`http://localhost:8000`).

Explanation suppression probability (20%) is configured in `modules/components/case_loader.py`.
