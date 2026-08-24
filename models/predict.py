"""
predict.py
------------
Loads the trained model (models/model.pkl) and provides a simple
predict_severity() function that takes human-readable inputs
and returns a predicted severity label with class probabilities.
"""

import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join("models", "model.pkl")
METADATA_PATH = os.path.join("models", "model_metadata.pkl")

WEATHER_OPTIONS = ["Clear", "Foggy", "Rainy", "Stormy"]
ROAD_TYPE_OPTIONS = ["Chowk/Intersection", "Highway", "Main Road", "Residential Road"]
TRAFFIC_DENSITY_OPTIONS = ["High", "Low", "Medium"]
VEHICLE_TYPE_OPTIONS = ["Auto-Rickshaw", "Bus", "Car", "Truck", "Two-Wheeler"]
DAY_OPTIONS = ["Friday", "Monday", "Saturday", "Sunday", "Thursday", "Tuesday", "Wednesday"]
TIME_OF_DAY_OPTIONS = ["Afternoon", "Evening", "Morning", "Night"]


def _label_encode(value, options):
    sorted_opts = sorted(options)
    return sorted_opts.index(value)


class SeverityPredictor:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.metadata = joblib.load(METADATA_PATH)
        self.feature_cols = self.metadata["feature_cols"]
        self.severity_labels = self.metadata["severity_labels"]

    def predict(self, weather, road_type, traffic_density, vehicle_type,
                day_of_week, hour, time_of_day):
        is_weekend = 1 if day_of_week in ("Saturday", "Sunday") else 0
        is_rush_hour = 1 if (8 <= hour <= 10) or (17 <= hour <= 20) else 0
        is_adverse_weather = 1 if weather in ("Rainy", "Foggy", "Stormy") else 0

        row = {
            "weather_condition_encoded": _label_encode(weather, WEATHER_OPTIONS),
            "road_type_encoded": _label_encode(road_type, ROAD_TYPE_OPTIONS),
            "traffic_density_encoded": _label_encode(traffic_density, TRAFFIC_DENSITY_OPTIONS),
            "vehicle_type_encoded": _label_encode(vehicle_type, VEHICLE_TYPE_OPTIONS),
            "day_of_week_encoded": _label_encode(day_of_week, DAY_OPTIONS),
            "time_of_day_encoded": _label_encode(time_of_day, TIME_OF_DAY_OPTIONS),
            "hour": hour,
            "is_weekend": is_weekend,
            "is_rush_hour": is_rush_hour,
            "is_adverse_weather": is_adverse_weather,
        }

        X = pd.DataFrame([row])[self.feature_cols]

        pred_class = self.model.predict(X)[0]
        pred_label = self.severity_labels[pred_class]

        probabilities = {}
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            probabilities = {
                self.severity_labels[i]: round(float(p), 3)
                for i, p in enumerate(proba)
            }

        return pred_label, probabilities


def generate_precautions(weather, road_type, traffic_density, vehicle_type,
                          hour, day_of_week, predicted_severity):
    """
    Rule-based precaution generator. Returns a list of specific,
    actionable safety suggestions based on the selected conditions
    and the model's predicted severity. This is what turns a raw
    prediction into practical, real-world advice.
    """
    tips = []

    # Severity-driven headline advice
    if predicted_severity == "Fatal":
        tips.append("⚠️ High-risk combination detected. Consider avoiding travel under these exact conditions if possible, or exercise extreme caution.")
    elif predicted_severity == "Serious":
        tips.append("⚠️ Elevated risk conditions. Reduce speed and maintain extra following distance.")
    else:
        tips.append("✅ Relatively lower-risk conditions, but standard road safety practices still apply.")

    # Weather-specific
    if weather == "Rainy":
        tips.append("🌧️ Wet roads reduce tire grip significantly — reduce speed by at least 20-25% and avoid sudden braking.")
    elif weather == "Foggy":
        tips.append("🌫️ Low visibility conditions — use fog lights, avoid overtaking, and increase following distance.")
    elif weather == "Stormy":
        tips.append("⛈️ Severe weather — consider delaying travel until conditions improve if the journey isn't urgent.")

    # Road type specific
    if road_type == "Highway":
        tips.append("🛣️ Highway driving — maintain lane discipline and avoid sudden lane changes, especially at higher speeds.")
    elif road_type == "Chowk/Intersection":
        tips.append("🚦 Intersection risk is usually from sudden crossing traffic — approach at reduced speed and watch for pedestrians/two-wheelers.")

    # Vehicle specific
    if vehicle_type == "Two-Wheeler":
        tips.append("🏍️ Two-wheeler riders: always wear a helmet, avoid weaving between lanes, and stay visible to larger vehicles.")
    elif vehicle_type in ("Bus", "Truck"):
        tips.append("🚛 Heavy vehicle: allow longer stopping distances and be extra cautious around blind spots.")

    # Time-based
    if (8 <= hour <= 10) or (17 <= hour <= 20):
        tips.append("⏰ This falls in rush-hour timing — expect heavier traffic and unpredictable lane changes from other vehicles.")
    if hour >= 22 or hour <= 4:
        tips.append("🌙 Late-night/early-morning travel — visibility is lower and fatigue-related risk increases; stay alert.")

    # Traffic density
    if traffic_density == "Low" and road_type == "Highway":
        tips.append("🏎️ Low traffic on a highway often leads to higher speeds — resist the temptation to speed even when roads are empty.")

    return tips