"""
app.py
--------
Home page of the Smart Traffic Accident Analytics dashboard.
Run from project root:
    streamlit run dashboard/app.py
"""

import os
import sys
import streamlit as st
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Road-Safe Ranchi",
    page_icon="dashboard/assets/logo.png",
    layout="wide",
)

# Force sidebar navigation labels (app, about, analytics, etc.) to uppercase
st.markdown("""
<style>
[data-testid="stSidebarNav"] span {
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

DATA_PATH = os.path.join("data", "eda_ready.csv")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["accident_date"] = pd.to_datetime(df["accident_date"])
    return df


import base64
import streamlit.components.v1 as components

def get_base64_image(filename):
    # Try relative to project root
    path1 = os.path.join("dashboard", "assets", filename)
    if os.path.exists(path1):
        with open(path1, "rb") as f:
            return base64.b64encode(f.read()).decode()
    # Try relative to this file
    path2 = os.path.join(os.path.dirname(__file__), "assets", filename)
    if os.path.exists(path2):
        with open(path2, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

logo_light = get_base64_image("logo.png")
logo_dark = get_base64_image("logo1.png")

# Set initial theme based on st.context.theme if available
initial_class = "light-mode"
try:
    if hasattr(st.context, "theme") and st.context.theme:
        if st.context.theme.get("type") == "dark":
            initial_class = "dark-mode"
except Exception:
    pass

logo_col, title_col = st.columns([1, 4])
with logo_col:
    if logo_light and logo_dark:
        st.markdown(
            f"""
            <div class="logo-container {initial_class}">
                <img src="data:image/png;base64,{logo_light}" class="logo-light" width="220">
                <img src="data:image/png;base64,{logo_dark}" class="logo-dark" width="220">
            </div>
            <style>
                .logo-container.light-mode .logo-light {{
                    display: block !important;
                }}
                .logo-container.light-mode .logo-dark {{
                    display: none !important;
                }}
                .logo-container.dark-mode .logo-light {{
                    display: none !important;
                }}
                .logo-container.dark-mode .logo-dark {{
                    display: block !important;
                }}
            </style>
            """,
            unsafe_allow_html=True
        )
        
        components.html(
            """
            <script>
                const doc = window.parent.document;
                const stApp = doc.querySelector('.stApp');
                if (stApp) {
                    const updateLogo = () => {
                        const bg = window.getComputedStyle(stApp).backgroundColor;
                        const rgb = bg.match(/\\d+/g);
                        if (rgb) {
                            const r = parseInt(rgb[0]), g = parseInt(rgb[1]), b = parseInt(rgb[2]);
                            const isDark = (r * 0.299 + g * 0.587 + b * 0.114) < 128;
                            doc.querySelectorAll('.logo-container').forEach(container => {
                                if (isDark) {
                                    container.classList.add('dark-mode');
                                    container.classList.remove('light-mode');
                                } else {
                                    container.classList.add('light-mode');
                                    container.classList.remove('dark-mode');
                                }
                            });
                        }
                    };
                    updateLogo();
                    const observer = new MutationObserver(updateLogo);
                    observer.observe(stApp, { attributes: true, attributeFilter: ['style', 'class'] });
                }
            </script>
            """,
            height=0,
            width=0
        )
    else:
        st.image("dashboard/assets/logo.png", width=220)
with title_col:
    st.title("Road-Safe Ranchi - Smart Traffic Accident Analytics and Safety recommendation System")
    st.markdown("### Ranchi, Jharkhand — Data Analytics & Machine Learning Project")

st.markdown("""
This dashboard presents a complete analytics pipeline built on **44 major locations
in Ranchi**, combining real geographic data (OpenStreetMap) with a Monte Carlo–simulated
synthetic accident dataset, processed and modeled end-to-end using Python.
""")

try:
    df = load_data()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Accidents", f"{len(df):,}")
    col2.metric("Locations Covered", df["location_name"].nunique())
    col3.metric("Fatal Accidents", f"{(df['accident_severity'] == 'Fatal').sum():,}")
    col4.metric("Total Casualties", f"{df['num_casualties'].sum():,}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Severity Distribution")
        severity_counts = df["accident_severity"].value_counts()
        st.bar_chart(severity_counts)

    with col2:
        st.subheader("Top 10 Hotspot Locations")
        top_locations = df["location_name"].value_counts().head(10)
        st.bar_chart(top_locations)

except FileNotFoundError:
    st.error(
        f"Could not find {DATA_PATH}. Make sure you've completed Phases 3-6 "
        "(location geocoding, accident generation, and preprocessing) before running the dashboard."
    )

st.markdown("---")
st.markdown("""
**Navigate using the sidebar** to explore:
- 🗺️ **Hotspot Map** — geographic visualization of accident locations
- 📊 **Analytics** — detailed charts and trends
- 🔮 **Risk Prediction** — predict severity for custom conditions
- 📁 **Dataset Explorer** — browse and filter the raw data
- ℹ️ **About** — project methodology and limitations
""")