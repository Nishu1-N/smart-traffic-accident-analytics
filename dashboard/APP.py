"""
app.py
--------
Home page of the Smart Traffic Accident Analytics dashboard.
Run from project root:
    streamlit run dashboard/app.py
"""

import os
import sys

# Auto-correct page filenames to uppercase with lowercase .py extension
try:
    _db_dir = os.path.dirname(__file__)
    _pg_dir = os.path.join(_db_dir, "pages")
    if os.path.exists(_pg_dir):
        for _f in os.listdir(_pg_dir):
            if _f.lower().endswith(".py"):
                _base = os.path.splitext(_f)[0]
                _expected = _base.upper() + ".py"
                if _f != _expected:
                    _src = os.path.join(_pg_dir, _f)
                    _tmp = os.path.join(_pg_dir, _f + ".tmp")
                    _dest = os.path.join(_pg_dir, _expected)
                    os.rename(_src, _tmp)
                    os.rename(_tmp, _dest)
except Exception:
    pass

import streamlit as st
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


st.set_page_config(
    page_title="Smart Traffic Accident Analytics - Ranchi",
    page_icon="🚦",
    layout="wide",
)

# Force sidebar navigation labels (app, about, analytics, etc.) to uppercase persistently to prevent loading flickers
st.markdown("""
<img src="x" style="display:none" onerror="
    const doc = (window.parent.document || document);
    if (!doc.getElementById(&quot;global-sidebar-style&quot;)) {
        const style = doc.createElement(&quot;style&quot;);
        style.id = &quot;global-sidebar-style&quot;;
        style.textContent = &quot;[data-testid=stSidebarNav] span { text-transform: uppercase !important; letter-spacing: 0.5px !important; }&quot;;
        doc.head.appendChild(style);
    }
">
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


st.title("🚦 Smart Traffic Accident Analytics and Risk Prediction System")
st.markdown("### Ranchi, Jharkhand — Data Analytics & Machine Learning Minor Project")

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