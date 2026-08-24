"""
Hotspot Map page - shows accident locations on an interactive map,
color-coded by each location's TRUE overall fatal accident rate
(calculated from the full dataset), with a heatmap overlay showing
density. The severity filter controls which locations are SHOWN,
but does not change the risk color - this avoids the bug where
filtering to a single severity (e.g. "Serious" only) would make
every location appear artificially low-risk.
"""

import os
import pandas as pd
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

st.set_page_config(page_title="Hotspot Map", page_icon="dashboard/assets/logo.png", layout="wide")

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


def compute_location_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each location's TRUE fatal accident rate using the FULL
    (unfiltered) dataset. This should always be called on the complete
    data, never on a severity-filtered subset, otherwise filtering to
    a single severity would artificially zero out the fatal rate.
    """
    return (
        df.groupby(["location_name", "latitude", "longitude"])
        .agg(
            total_accidents_all=("accident_id", "count"),
            fatal_count_all=("accident_severity", lambda x: (x == "Fatal").sum()),
        )
        .reset_index()
        .assign(fatal_rate=lambda d: d["fatal_count_all"] / d["total_accidents_all"])
    )


st.title("🗺️ Accident Hotspot Map — Ranchi")
st.markdown("Explore accident locations across Ranchi. Toggle between individual markers and a density heatmap.")

try:
    df = load_data()

    # Risk color is always computed from the FULL dataset (all severities)
    location_risk = compute_location_risk(df)

    col1, col2 = st.columns([1, 3])
    with col1:
        view_mode = st.radio("View mode", ["Heatmap (density)", "Markers (by severity)"])
        severity_filter = st.multiselect(
            "Filter by severity",
            options=["Minor", "Serious", "Fatal"],
            default=["Minor", "Serious", "Fatal"],
        )

        st.markdown("---")
        st.markdown("**Marker Color Legend**")
        st.markdown(
            "Marker color reflects each location's **overall fatal accident rate** "
            "(calculated across ALL accidents at that location, regardless of the "
            "severity filter above). The filter only controls which locations are "
            "shown, not the risk color."
        )
        st.markdown("🟢 **Green** — Low risk (fatal rate under 8%)")
        st.markdown("🟠 **Orange** — Moderate risk (fatal rate 8-15%)")
        st.markdown("🔴 **Red** — High risk (fatal rate above 15%)")
        st.caption("Marker size scales with the number of accidents matching the selected severities.")

    # Filter which locations to show based on selected severities,
    # but risk color always comes from the pre-computed location_risk table
    filtered_df = df[df["accident_severity"].isin(severity_filter)]
    
    # Calculate filtered counts to update marker sizes dynamically
    filtered_counts = filtered_df.groupby("location_name").size().to_dict()
    
    locations_to_show = [loc for loc, count in filtered_counts.items() if count > 0]
    display_data = location_risk[location_risk["location_name"].isin(locations_to_show)]

    if display_data.empty:
        center_lat, center_lng = 23.3441, 85.3090 # Ranchi center defaults
    else:
        center_lat = display_data["latitude"].mean()
        center_lng = display_data["longitude"].mean()

    m = folium.Map(location=[center_lat, center_lng], zoom_start=12, tiles="OpenStreetMap")

    if view_mode == "Heatmap (density)":
        heat_data = filtered_df[["latitude", "longitude"]].values.tolist()
        HeatMap(heat_data, radius=12, blur=15).add_to(m)
    else:
        for _, row in display_data.iterrows():
            fatal_rate = row["fatal_rate"]
            color = "red" if fatal_rate > 0.15 else ("orange" if fatal_rate > 0.08 else "green")
            filtered_count = filtered_counts.get(row["location_name"], 0)

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=6 + (filtered_count ** 0.5),
                popup=(
                    f"<b>{row['location_name']}</b><br>"
                    f"Filtered accidents (selected severities): {filtered_count}<br>"
                    f"Total accidents (all severities): {row['total_accidents_all']}<br>"
                    f"Fatal: {row['fatal_count_all']} ({fatal_rate*100:.1f}%)"
                ),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
            ).add_to(m)

    with col2:
        st_folium(m, width=900, height=550)

    st.markdown("---")
    st.subheader("Location Summary Table")
    summary = (
        filtered_df.groupby("location_name")
        .agg(
            total_accidents=("accident_id", "count"),
            fatal=("accident_severity", lambda x: (x == "Fatal").sum()),
            serious=("accident_severity", lambda x: (x == "Serious").sum()),
            minor=("accident_severity", lambda x: (x == "Minor").sum()),
        )
        .sort_values("total_accidents", ascending=False)
        .reset_index()
    )
    st.dataframe(summary, use_container_width=True)

except FileNotFoundError:
    st.error(f"Could not find {DATA_PATH}. Run Phases 3-6 first.")