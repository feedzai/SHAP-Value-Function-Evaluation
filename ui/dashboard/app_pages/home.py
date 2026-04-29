# pages/home.py
import streamlit as st


def render_home_page():
    st.markdown(
        """
        <style>
            /* CSS for Centering */
            .centered-title-container {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 70vh;
                text-align: center;
                flex-direction: column;
            }
            .centered-title {
                font-size: 3em;
                font-weight: bold;
                margin-bottom: 0.2em;
            }
            .subtitle-text {
                font-size: 1.5em !important;
                color: #4A4A4A;
                max-width: 800px;
            }
        </style>

        <div class="centered-title-container">
            <div class="centered-title">
                Welcome to the XAI Research Tools and Dashboards Suite 👋
            </div>
            <p class="subtitle-text">
                Here, you'll find stuff we build to test concepts. Use the sidebar to look around.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Need help getting started? There's a 'Get Help' link in the top-right menu."
    )
