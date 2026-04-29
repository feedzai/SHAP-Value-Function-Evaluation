# dashboard/modules/widgets/decision_box.py
from datetime import datetime

import streamlit as st
from modules.api_client import submit_case_response


def _submit_case_response():
    """Build and submit the case response to the API."""
    analyst = st.session_state.get("analyst", {})
    decision_str = st.session_state.get("analyst_decision", "No Risk")
    decision = 1 if decision_str == "Risk" else 0

    # Build full analyst profile payload
    analyst_profile = {
        "alias": analyst.get("alias", "unknown"),
        "domain_knowledge": analyst.get("domain_knowledge", "No"),
        "ml_understanding": analyst.get("ml_understanding", "Low"),
        "shapley_understanding": analyst.get("shapley_understanding", "No"),
        "uuid": analyst.get("uuid", ""),
    }

    case_response = {
        "user": analyst.get("alias", "unknown"),
        "analyst_profile": analyst_profile,
        "decision": decision,
        "confidence": st.session_state.get("analyst_confidence", ""),
        "clarity": st.session_state.get("explanation_clarity", ""),
        "timestamp_start": st.session_state.get("timestamp_start"),
        "timestamp_end": st.session_state.get("timestamp_end"),
        "case_counter": st.session_state.get("case_counter", 0),
        "case": st.session_state.get("current_case"),
    }

    submit_case_response(case_response)


def render_decision_pending_state():
    st.subheader(":material/search_check: &nbsp; Decision")
    col_risk, col_no_risk = st.columns(2)

    st.html(
        """
        <style>
            .st-key-risk_btn, .st-key-no_risk_btn {
                white-space: pre-wrap !important;
                text-align: center;
            }
        </style>
    """
    )

    with col_risk:
        if st.button("🔴\nRisk", width="stretch", key="risk_btn"):
            st.session_state["analyst_decision"] = "Risk"
            st.session_state["timestamp_end"] = datetime.now().isoformat()
            st.session_state["review_state"] = "confidence_pending"
            st.rerun()

    with col_no_risk:
        if st.button("🟢\nNo Risk", width="stretch", key="no_risk_btn"):
            st.session_state["analyst_decision"] = "No Risk"
            st.session_state["timestamp_end"] = datetime.now().isoformat()
            st.session_state["review_state"] = "confidence_pending"
            st.rerun()


def render_confidence_pending_state():
    st.subheader(":material/mystery: &nbsp; Confidence")
    col_not, col_mod, col_cert = st.columns(3)

    st.html(
        """
        <style>
            .st-key-weak_btn, 
            .st-key-moderate_btn, 
            .st-key-strong_btn {
                white-space: pre-wrap !important; 
                text-align: center;
            }
        </style>
    """
    )

    if col_not.button(
        ":material/network_wifi_1_bar:\nWeak", width="stretch", key="weak_btn"
    ):
        st.session_state["analyst_confidence"] = "Weak"
        st.session_state["review_state"] = "explanation_feedback"
        st.rerun()

    if col_mod.button(
        ":material/network_wifi_2_bar:\nModerate", width="stretch", key="moderate_btn"
    ):
        st.session_state["analyst_confidence"] = "Moderate"
        st.session_state["review_state"] = "explanation_feedback"
        st.rerun()

    if col_cert.button(
        ":material/network_wifi:\nStrong", width="stretch", key="strong_btn"
    ):
        st.session_state["analyst_confidence"] = "Strong"
        st.session_state["review_state"] = "explanation_feedback"
        st.rerun()


def render_explanation_feedback_state():
    st.subheader(":material/lightbulb: &nbsp; Explanation Clarity")
    col_yes, col_no = st.columns(2)

    st.html(
        """
        <style>
            .st-key-exp_yes_btn, 
            .st-key-exp_no_btn {
                white-space: pre-wrap !important; 
                text-align: center;
            }
        </style>
    """
    )

    with col_yes:
        if st.button("👍\nClear", width="stretch", key="exp_yes_btn"):
            st.session_state["explanation_clarity"] = "Yes"
            st.session_state["review_state"] = "reviewed"
            _submit_case_response()
            st.rerun()

    with col_no:
        if st.button("👎\nConfusing", width="stretch", key="exp_no_btn"):
            st.session_state["explanation_clarity"] = "No"
            st.session_state["review_state"] = "reviewed"
            _submit_case_response()
            st.rerun()


def render_reviewed_state():
    st.subheader(":material/check_circle: &nbsp; Case Reviewed")

    decision = st.session_state.get("analyst_decision", "N/A")
    confidence = st.session_state.get("analyst_confidence", "N/A")
    feedback = st.session_state.get("explanation_clarity", "N/A")

    st.info(
        f"**Decision:** {decision} | "
        f"**Confidence:** {confidence} | "
        f"**Clarity:** {feedback}"
    )
    st.write("**Fetch Next Case** to continue.")


# --- Main rendering function for the box (called from case_summary) ---


def render():
    """Renders the appropriate state of the Decision/Confidence box."""

    if "review_state" not in st.session_state:
        st.session_state["review_state"] = "decision_pending"

    current_state = st.session_state["review_state"]

    if current_state == "decision_pending":
        render_decision_pending_state()
    elif current_state == "confidence_pending":
        render_confidence_pending_state()
    elif current_state == "explanation_feedback":
        render_explanation_feedback_state()
    elif current_state == "reviewed":
        render_reviewed_state()
