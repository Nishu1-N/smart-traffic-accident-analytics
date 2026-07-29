"""
Dataset Explorer page — browse, filter, and download the underlying
accident dataset.
"""

import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dataset Explorer", page_icon="📁", layout="wide")

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


st.title("📁 Dataset Explorer")
st.markdown("Browse and filter the full accident dataset. Use the filters below, then download the filtered results as CSV.")

try:
    df = load_data()

    with st.expander("Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            locations = st.multiselect("Location", options=sorted(df["location_name"].unique()))
            severity = st.multiselect("Severity", options=["Minor", "Serious", "Fatal"])
        with col2:
            weather = st.multiselect("Weather", options=sorted(df["weather_condition"].unique()))
            road_type = st.multiselect("Road Type", options=sorted(df["road_type"].unique()))
        with col3:
            vehicle = st.multiselect("Vehicle Type", options=sorted(df["vehicle_type"].unique()))

    filtered = df.copy()
    if locations:
        filtered = filtered[filtered["location_name"].isin(locations)]
    if severity:
        filtered = filtered[filtered["accident_severity"].isin(severity)]
    if weather:
        filtered = filtered[filtered["weather_condition"].isin(weather)]
    if road_type:
        filtered = filtered[filtered["road_type"].isin(road_type)]
    if vehicle:
        filtered = filtered[filtered["vehicle_type"].isin(vehicle)]

    st.markdown(f"**{len(filtered):,} rows** match your filters (out of {len(df):,} total).")
    st.dataframe(filtered, use_container_width=True, height=500)

    st.markdown("---")
    st.subheader("Save Filtered Data")
    st.markdown(
        "Browser downloads can sometimes save the wrong file. Instead, click below "
        "to save the CSV directly to your **Desktop** using Python."
    )

    if st.button("💾 Save filtered data to Desktop"):
        try:
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            os.makedirs(desktop_path, exist_ok=True)
            save_path = os.path.join(desktop_path, "filtered_accidents.csv")
            filtered.to_csv(save_path, index=False)
            st.success(f"✅ Saved successfully to: {save_path}")
        except Exception as e:
            st.error(f"Could not save file: {e}")

    # Keep the browser download button too, as a fallback option
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Or download via browser",
        data=csv,
        file_name="filtered_accidents.csv",
        mime="text/csv",
    )

except FileNotFoundError:
    st.error(f"Could not find {DATA_PATH}. Run Phases 3-6 first.")