"""
Safety Recommendations page - analyzes the accident dataset and
generates specific, data-driven recommendations for traffic
authorities. This is what demonstrates real problem-solving value:
turning raw analytics into concrete suggested interventions.
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = BASE_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"
DATA_PATH = BASE_DIR.parent / "data" / "eda_ready.csv"

# Logo
logo = Image.open(LOGO_PATH)

st.set_page_config(
    page_title="Safety Recommendations",
    page_icon=logo,
    layout="wide"
)

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
    return df


def recommend_intervention(location_stats):
    """
    Given a location's accident stats, generate a short list of
    concrete intervention suggestions - the kind a traffic
    department could actually act on.
    """
    suggestions = []
    fatal_rate = location_stats["fatal_rate"]
    top_road_type = location_stats["top_road_type"]
    top_weather = location_stats["top_weather"]
    top_vehicle = location_stats["top_vehicle"]
    rush_hour_share = location_stats["rush_hour_share"]

    if fatal_rate > 0.15:
        suggestions.append("Deploy speed enforcement (radar/cameras) — fatal rate is significantly above average.")
    if top_road_type == "Chowk/Intersection":
        suggestions.append("Install or upgrade traffic signals and pedestrian crossings at this intersection.")
    if top_road_type == "Highway":
        suggestions.append("Add median barriers and speed-limit signage; consider rumble strips before the location.")
    if top_weather in ("Rainy", "Foggy"):
        suggestions.append("Install weather-responsive warning signage and improve road surface drainage.")
    if top_vehicle == "Two-Wheeler":
        suggestions.append("Add dedicated two-wheeler lanes or improve lane markings to reduce weaving.")
    if top_vehicle in ("Bus", "Truck"):
        suggestions.append("Enforce heavy-vehicle speed limits and restrict heavy-vehicle hours if feasible.")
    if rush_hour_share > 0.5:
        suggestions.append("Deploy traffic police during rush hours (8-10 AM, 5-8 PM) to manage congestion-related risk.")

    if not suggestions:
        suggestions.append("No urgent intervention flagged — continue routine monitoring.")

    return suggestions


st.title("🛡️ Data-Driven Safety Recommendations")
st.markdown("""
This page analyzes accident patterns across all locations and generates **specific,
actionable recommendations** for traffic authorities — turning raw analytics into
practical safety interventions.
""")

try:
    df = load_data()

    # Build per-location statistics
    location_groups = df.groupby("location_name")

    location_stats_list = []
    for name, group in location_groups:
        total = len(group)
        fatal = (group["accident_severity"] == "Fatal").sum()
        fatal_rate = fatal / total if total > 0 else 0
        top_road_type = group["road_type"].mode()[0] if not group["road_type"].mode().empty else "Unknown"
        top_weather = group["weather_condition"].mode()[0] if not group["weather_condition"].mode().empty else "Unknown"
        top_vehicle = group["vehicle_type"].mode()[0] if not group["vehicle_type"].mode().empty else "Unknown"
        rush_hour_share = group["is_rush_hour"].mean() if "is_rush_hour" in group.columns else 0

        location_stats_list.append({
            "location_name": name,
            "total_accidents": total,
            "fatal_count": fatal,
            "fatal_rate": fatal_rate,
            "top_road_type": top_road_type,
            "top_weather": top_weather,
            "top_vehicle": top_vehicle,
            "rush_hour_share": rush_hour_share,
        })

    location_stats_df = pd.DataFrame(location_stats_list).sort_values("fatal_rate", ascending=False)

    # --- Overall dataset-level recommendations ---
    st.subheader("📋 City-Wide Priority Actions")
    overall_top_vehicle = df["vehicle_type"].value_counts().idxmax()
    overall_top_weather_fatal = (
        df[df["accident_severity"] == "Fatal"]["weather_condition"].value_counts().idxmax()
        if (df["accident_severity"] == "Fatal").sum() > 0 else "N/A"
    )
    peak_hour = df["hour"].value_counts().idxmax()

    st.markdown(f"""
    - **Most frequently involved vehicle type:** {overall_top_vehicle} — consider targeted awareness campaigns and lane infrastructure for this vehicle category.
    - **Weather condition most associated with fatal accidents:** {overall_top_weather_fatal} — prioritize weather-responsive signage and road maintenance ahead of this condition.
    - **Peak accident hour:** {peak_hour}:00 — recommend increased traffic police presence during this period.
    """)

    st.markdown("---")

    # --- Top 10 highest-risk locations with specific recommendations ---
    st.subheader("🚨 Top 10 Highest-Risk Locations — Specific Recommendations")
    st.caption("Ranked by fatal accident rate. Each location includes tailored intervention suggestions based on its dominant conditions.")

    top_10 = location_stats_df.head(10)

    for _, row in top_10.iterrows():
        with st.expander(f"📍 {row['location_name']} — Fatal rate: {row['fatal_rate']*100:.1f}% ({row['total_accidents']} total accidents)"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Total accidents:** {row['total_accidents']}")
                st.markdown(f"**Fatal accidents:** {row['fatal_count']}")
                st.markdown(f"**Dominant road type:** {row['top_road_type']}")
            with col2:
                st.markdown(f"**Dominant weather:** {row['top_weather']}")
                st.markdown(f"**Most common vehicle:** {row['top_vehicle']}")
                st.markdown(f"**Rush-hour share:** {row['rush_hour_share']*100:.0f}%")

            st.markdown("**Recommended interventions:**")
            recs = recommend_intervention(row)
            for r in recs:
                st.markdown(f"- {r}")

    st.markdown("---")
    st.info(
        "These recommendations are generated from patterns in the synthetic dataset "
        "and follow standard road-safety engineering practices. If real accident data "
        "were available, this same analysis engine would generate recommendations "
        "based on actual historical patterns for Ranchi."
    )

except FileNotFoundError:
    st.error(f"Could not find {DATA_PATH}. Run Phases 3-6 first.")