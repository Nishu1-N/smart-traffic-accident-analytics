"""
Risk Prediction page - lets the user pick accident conditions, get a
live severity prediction from the trained ML model, AND now shows
specific, actionable safety precautions based on those conditions.
This is what turns the tool from "just a prediction" into practical
guidance a user could actually act on.
"""

import os
import sys
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Risk Prediction", page_icon="dashboard/assets/logo.png", layout="wide")

st.markdown("""
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
circumstances, using the machine learning model trained  — along with
specific safety precautions for those conditions.
""")

try:
    from models.predict import (
        SeverityPredictor, generate_precautions, WEATHER_OPTIONS, ROAD_TYPE_OPTIONS,
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
        hour_labels = []
        for h in range(24):
            period = "AM" if h < 12 else "PM"
            display_hour = h % 12
            if display_hour == 0:
                display_hour = 12
            hour_labels.append(f"{display_hour} {period}")
        selected_label = st.select_slider("Hour of Day", options=hour_labels, value="6 PM")
        hour = hour_labels.index(selected_label)
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

        # --- NEW: Actionable safety precautions ---
        st.markdown("---")
        st.subheader("🛡️ Recommended Safety Precautions")
        precautions = generate_precautions(
            weather=weather, road_type=road_type, traffic_density=traffic_density,
            vehicle_type=vehicle_type, hour=hour, day_of_week=day_of_week,
            predicted_severity=label,
        )
        for tip in precautions:
            st.markdown(f"- {tip}")

        st.info(
            "Note: This prediction and these precautions are based on a model trained on "
            "synthetic (simulated) accident data, intended to demonstrate the full analytics "
            "pipeline. They are general road-safety guidance, not based on real historical "
            "records for Ranchi."
        )

except FileNotFoundError:
    st.error(
        "Could not find models/model.pkl. Please run Phase 8 "
        "(python models/train_model.py) before using this page."
    )
except Exception as e:
    st.error(f"An error occurred: {e}")