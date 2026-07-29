"""
Risk Prediction page — lets the user pick accident conditions and
get a live severity prediction from the trained ML model
(models/model.pkl, trained in Phase 8).
"""

import os
import sys
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
st.set_page_config(page_title="Risk Prediction", page_icon="🔮", layout="wide")

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



st.title("🔮 Accident Risk Prediction")
st.markdown("""
Select conditions below to predict the likely severity of an accident under those
circumstances, using the machine learning model trained.
""")

try:
    from models.predict import (
        SeverityPredictor, WEATHER_OPTIONS, ROAD_TYPE_OPTIONS,
        TRAFFIC_DENSITY_OPTIONS, VEHICLE_TYPE_OPTIONS, DAY_OPTIONS, TIME_OF_DAY_OPTIONS
    )

    @st.cache_resource
    def get_predictor():
        return SeverityPredictor()

    predictor = get_predictor()

    col1, col2, col3 = st.columns(3)
    with col1:
        weather = st.selectbox("Weather Condition", WEATHER_OPTIONS)
        road_type = st.selectbox("Road Type", ROAD_TYPE_OPTIONS)
        traffic_density = st.selectbox("Traffic Density", TRAFFIC_DENSITY_OPTIONS)

    with col2:
        vehicle_type = st.selectbox("Vehicle Type", VEHICLE_TYPE_OPTIONS)
        day_of_week = st.selectbox("Day of Week", DAY_OPTIONS)

    with col3:
        # Build 12-hour AM/PM labels for all 24 hours (e.g. "12 AM", "1 AM", ..., "11 PM")
        hour_labels = []
        for h in range(24):
            period = "AM" if h < 12 else "PM"
            display_hour = h % 12
            if display_hour == 0:
                display_hour = 12
            hour_labels.append(f"{display_hour} {period}")

        selected_label = st.select_slider("Hour of Day", options=hour_labels, value="6 PM")
        hour = hour_labels.index(selected_label)  # convert back to 0-23 for the model
        time_of_day = st.selectbox("Time of Day", TIME_OF_DAY_OPTIONS)

    st.markdown("---")

    if st.button("🔮 Predict Severity", type="primary"):
        label, probs = predictor.predict(
            weather=weather,
            road_type=road_type,
            traffic_density=traffic_density,
            vehicle_type=vehicle_type,
            day_of_week=day_of_week,
            hour=hour,
            time_of_day=time_of_day,
        )

        severity_colors = {"Minor": "🟢", "Serious": "🟠", "Fatal": "🔴"}
        st.markdown(f"## Predicted Severity: {severity_colors.get(label, '')} **{label}**")

        if probs:
            st.subheader("Prediction Confidence")
            col1, col2, col3 = st.columns(3)
            col1.metric("Minor", f"{probs.get('Minor', 0) * 100:.1f}%")
            col2.metric("Serious", f"{probs.get('Serious', 0) * 100:.1f}%")
            col3.metric("Fatal", f"{probs.get('Fatal', 0) * 100:.1f}%")
            st.bar_chart(probs)

        st.info(
            "Note: This prediction is based on a model trained on synthetic (simulated) "
            "accident data, intended to demonstrate the full analytics pipeline. It is not "
            "based on real historical accident records for Ranchi."
        )

except FileNotFoundError:
    st.error(
        "Could not find models/model.pkl. Please run Phase 8 "
        "(python models/train_model.py) before using this page."
    )
except Exception as e:
    st.error(f"An error occurred: {e}")