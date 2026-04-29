# dashboard/modules/components/analyst_profile.py
import streamlit as st

import uuid


def render():
    """Renders the Analyst Profile input box and saves to session_state."""
    if "analyst" not in st.session_state:
        st.session_state["analyst"] = {}

    st.html(
        """
        <style>
            .st-key-my_blue_container { background-color: rgba(240, 249, 255, 1); }
        </style>
    """
    )

    # Profile rendering logic
    with st.container(border=True, key="my_blue_container"):
        st.markdown("#### :material/person_shield: Analyst Profile")

        # ----- ROW 1: Alias + Experience -----
        col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])

        with col1:
            alias = st.text_input(
                "Alias",
                value=st.session_state["analyst"].get("alias", ""),
            )

        with col2:
            domain_knowledge = st.selectbox(
                "Domain Knowledge",
                ["Yes", "No"],
                index=["Yes", "No"].index(
                    st.session_state["analyst"].get("domain_knowledge", "No")
                ),
            )

        with col3:
            ml_understanding = st.selectbox(
                "ML Knowledge",
                ["Low", "Moderate", "Strong"],
                index=["Low", "Moderate", "Strong"].index(
                    st.session_state["analyst"].get("ml_understanding", "Low")
                ),
            )

        with col4:
            shapley_understanding = st.selectbox(
                "Understands Shapley",
                ["Yes", "No"],
                index=["Yes", "No"].index(
                    st.session_state["analyst"].get("shapley_understanding", "No")
                ),
            )

        with col5:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            save_analyst_button = st.button("Get Started")

        if save_analyst_button:
            if not alias.strip():
                st.error("Alias cannot be empty.")
            else:
                st.session_state["analyst"] = {
                    "alias": alias.strip(),
                    "domain_knowledge": domain_knowledge,
                    "ml_understanding": ml_understanding,
                    "shapley_understanding": shapley_understanding,
                    "uuid": str(uuid.uuid4()),
                }
                st.success("Profile saved!")

    return st.session_state["analyst"]
