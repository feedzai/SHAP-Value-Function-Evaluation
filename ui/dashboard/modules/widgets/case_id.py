# dashboard/modules/widgets/case_id.py
import streamlit as st


def render(case: dict):
    st.subheader(":material/folder_check_2: &nbsp; Case Details")
    col_case_id, col_case_n = st.columns(2)

    with col_case_id:
        st.metric(label="Instance ID", value=case.get("instance_id", "N/A"))

    with col_case_n:
        st.metric(label="Case Number", value=st.session_state["case_counter"])
