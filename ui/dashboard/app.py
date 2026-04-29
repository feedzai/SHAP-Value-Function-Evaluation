import base64

import streamlit as st

# Import the page functions and helper modules
from app_pages import case_review, home
from streamlit_option_menu import option_menu

# --- GLOBAL PAGE CONFIG and SETUP ---
st.set_page_config(
    page_title="XAI Research",
    page_icon="dashboard/assets/icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.dummy.com/dummy",
    },
)
st.set_option("client.toolbarMode", "minimal")

# --- GLOBAL UI CLEANUP ---
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 5rem !important;
        }
        # div[data-testid="stSidebarHeader"] {
        #     display: none !important;
        # }
    </style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------

with st.sidebar:
    try:
        with open("dashboard/assets/logo.png", "rb") as f:
            svg_bytes = f.read()

        b64_encoded = base64.b64encode(svg_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{b64_encoded}"

        st.markdown(
            f"""
            <style>
                @keyframes sidebar-pulse {{
                  0% {{ transform: scale(1); }}
                  50% {{ transform: scale(1.02); }}
                  100% {{ transform: scale(1); }}
                }}
                                
                .sidebar-logo-img {{
                    display: block;
                    margin-left: auto;
                    margin-right: auto;
                    animation: sidebar-pulse 3s infinite ease-in-out;
                    width: 190px; 
                    height: auto;
                }}
            </style>
            <img class="sidebar-logo-img" src="{data_url}" alt="Logo">
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

    except Exception:
        st.subheader("XAI Research")
        st.markdown("---")

    selected = option_menu(
        menu_title=None,
        options=["Home", "Case Review"],
        icons=["house", "clipboard-data"],
        menu_icon="none",
        default_index=0,
        styles={
            "container": {"background-color": "transparent"},
            "icon": {"font-size": "40px"},
        },
    )

# -------------------------------------------------
# MAIN APPLICATION CONTENT - ROUTING
# -------------------------------------------------

if selected == "Home":
    home.render_home_page()

elif selected == "Case Review":
    case_review.render_case_review_page()
