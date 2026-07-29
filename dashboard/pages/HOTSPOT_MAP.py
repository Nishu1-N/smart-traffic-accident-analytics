"""
Hotspot Map page — shows accident locations on an interactive map,
color-coded by severity, with a heatmap overlay showing density.
"""

import os
import pandas as pd
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
st.set_page_config(page_title="Hotspot Map", page_icon="🗺️", layout="wide")

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

SEVERITY_COLORS = {"Minor": "green", "Serious": "orange", "Fatal": "red"}


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


st.title("🗺️ Accident Hotspot Map — Ranchi")
st.markdown("Explore accident locations across Ranchi. Toggle between individual markers and a density heatmap.")

try:
    df = load_data()

    col1, col2 = st.columns([1, 3])
    with col1:
        view_mode = st.radio("View mode", ["Heatmap (density)", "Markers (by severity)"])
        severity_filter = st.multiselect(
            "Filter by severity",
            options=["Minor", "Serious", "Fatal"],
            default=["Minor", "Serious", "Fatal"],
        )

    filtered_df = df[df["accident_severity"].isin(severity_filter)]

    # Center map on average lat/lng of all locations
    center_lat = filtered_df["latitude"].mean()
    center_lng = filtered_df["longitude"].mean()

    m = folium.Map(location=[center_lat, center_lng], zoom_start=12, tiles="OpenStreetMap")

    if view_mode == "Heatmap (density)":
        heat_data = filtered_df[["latitude", "longitude"]].values.tolist()
        HeatMap(heat_data, radius=12, blur=15).add_to(m)
    else:
        # Aggregate per location so we don't drop thousands of overlapping markers
        location_summary = (
            filtered_df.groupby(["location_name", "latitude", "longitude"])
            .agg(
                total_accidents=("accident_id", "count"),
                fatal_count=("accident_severity", lambda x: (x == "Fatal").sum()),
            )
            .reset_index()
        )

        for _, row in location_summary.iterrows():
            # Marker size reflects accident volume, color reflects fatal risk
            fatal_ratio = row["fatal_count"] / row["total_accidents"] if row["total_accidents"] > 0 else 0
            color = "red" if fatal_ratio > 0.15 else ("orange" if fatal_ratio > 0.08 else "green")

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=6 + (row["total_accidents"] ** 0.5),
                popup=(
                    f"<b>{row['location_name']}</b><br>"
                    f"Total accidents: {row['total_accidents']}<br>"
                    f"Fatal: {row['fatal_count']} ({fatal_ratio*100:.1f}%)"
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