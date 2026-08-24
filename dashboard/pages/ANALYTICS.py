"""
Analytics page — interactive Plotly charts exploring accident
patterns by time, weather, vehicle type, road type, and trends.
"""

import os
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")

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


st.title("📊 Analytics Dashboard")

try:
    df = load_data()

    # --- Sidebar filters ---
    st.sidebar.header("Filters")
    year_filter = st.sidebar.multiselect(
        "Year", options=sorted(df["year"].unique()), default=sorted(df["year"].unique())
    )
    road_filter = st.sidebar.multiselect(
        "Road Type", options=df["road_type"].unique(), default=df["road_type"].unique()
    )

    filtered_df = df[df["year"].isin(year_filter) & df["road_type"].isin(road_filter)]

    st.markdown(f"**Showing {len(filtered_df):,} accidents** based on current filters.")
    st.markdown("---")

    # --- Row 1: Hour distribution + Severity pie ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Accidents by Hour of Day")
        hourly = filtered_df["hour"].value_counts().sort_index()

        def hour_to_ampm(h):
            period = "AM" if h < 12 else "PM"
            display_hour = h % 12
            if display_hour == 0:
                display_hour = 12
            return f"{display_hour} {period}"

        hour_labels = [hour_to_ampm(h) for h in hourly.index]
        fig = px.bar(x=hour_labels, y=hourly.values, labels={"x": "Hour", "y": "Accidents"})
        fig.update_xaxes(categoryorder="array", categoryarray=hour_labels)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**📝 Description:**")
        st.caption("Shows accident frequency across the 24-hour day. Peaks typically appear during morning and evening rush hours.")

    with col2:
        st.subheader("Severity Distribution")
        severity_counts = filtered_df["accident_severity"].value_counts()
        fig = px.pie(
            values=severity_counts.values, names=severity_counts.index,
            color=severity_counts.index,
            color_discrete_map={"Minor": "#4CAF50", "Serious": "#FF9800", "Fatal": "#F44336"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**📝 Description:**")
        st.caption("Proportion of accidents by severity level — Minor, Serious, or Fatal — across the filtered dataset.")

    # --- Row 2: Weather + Vehicle type ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Accidents by Weather")
        weather_counts = filtered_df["weather_condition"].value_counts()
        fig = px.bar(x=weather_counts.index, y=weather_counts.values, labels={"x": "Weather", "y": "Accidents"})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**📝 Description:**")
        st.caption("Accident counts under each weather condition. Higher counts under Clear weather mostly reflect that clear days are far more common overall.")

    with col2:
        st.subheader("Accidents by Vehicle Type")
        vehicle_counts = filtered_df["vehicle_type"].value_counts()
        fig = px.bar(x=vehicle_counts.index, y=vehicle_counts.values, labels={"x": "Vehicle", "y": "Accidents"})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**📝 Description:**")
        st.caption("Number of accidents involving each vehicle type. Two-wheelers appear most often, consistent with their high share of Indian road traffic.")

    # --- Row 3: Monthly trend ---
    st.subheader("Monthly Accident Trend")
    monthly = filtered_df.groupby(filtered_df["accident_date"].dt.to_period("M")).size()
    monthly.index = monthly.index.astype(str)
    fig = px.line(x=monthly.index, y=monthly.values, labels={"x": "Month", "y": "Accidents"})
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**📝 Description:**")
    st.caption("Accident counts over time (2021-2025), showing month-by-month trends across the full dataset period.")

    # --- Row 4: Severity by road type (stacked) ---
    st.subheader("Severity Breakdown by Road Type")
    cross = pd.crosstab(filtered_df["road_type"], filtered_df["accident_severity"], normalize="index") * 100
    cross = cross[["Minor", "Serious", "Fatal"]].reset_index()
    cross_melted = cross.melt(id_vars="road_type", var_name="Severity", value_name="Percentage")
    fig = px.bar(
        cross_melted, x="road_type", y="Percentage", color="Severity",
        color_discrete_map={"Minor": "#4CAF50", "Serious": "#FF9800", "Fatal": "#F44336"},
        barmode="stack",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**📝 Description:**")
    st.caption("Percentage breakdown of severity within each road type. Highways tend to show a higher share of Fatal accidents due to higher speeds.")

except FileNotFoundError:
    st.error(f"Could not find {DATA_PATH}. Run Phases 3-6 first.")