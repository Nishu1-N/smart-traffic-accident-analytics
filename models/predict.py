"""
predict.py
------------
Loads the trained model (models/model.pkl) and provides a simple
predict_severity() function that takes human-readable inputs
(e.g. "Rainy", "Highway", "High", "Truck") and returns a predicted
severity label with class probabilities. Used by the Streamlit
dashboard's Risk Prediction page (Phase 9).
"""

import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join("models", "model.pkl")
METADATA_PATH = os.path.join("models", "model_metadata.pkl")

# These must match the LabelEncoder mappings created in
# preprocessing/feature_engineering.py. If you re-run feature
# engineering and the category order changes, re-check these against
# the printed "Category encodings used" output from that script.
WEATHER_OPTIONS = ["Clear", "Foggy", "Rainy", "Stormy"]
ROAD_TYPE_OPTIONS = ["Chowk/Intersection", "Highway", "Main Road", "Residential Road"]
TRAFFIC_DENSITY_OPTIONS = ["High", "Low", "Medium"]
VEHICLE_TYPE_OPTIONS = ["Auto-Rickshaw", "Bus", "Car", "Truck", "Two-Wheeler"]
DAY_OPTIONS = ["Friday", "Monday", "Saturday", "Sunday", "Thursday", "Tuesday", "Wednesday"]
TIME_OF_DAY_OPTIONS = ["Afternoon", "Evening", "Morning", "Night"]


def _label_encode(value, options_sorted_alphabetically):
    """
    Mimic sklearn's LabelEncoder, which assigns integer codes in
    alphabetically sorted order of the unique categories.
    """
    sorted_opts = sorted(options_sorted_alphabetically)
    return sorted_opts.index(value)


class SeverityPredictor:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.metadata = joblib.load(METADATA_PATH)
        self.feature_cols = self.metadata["feature_cols"]
        self.severity_labels = self.metadata["severity_labels"]

    def predict(self, weather, road_type, traffic_density, vehicle_type,
                day_of_week, hour, time_of_day):
        """
        Predict accident severity from human-readable inputs.
        Returns (predicted_label, probability_dict).
        """
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


if __name__ == "__main__":
    # quick manual test
    predictor = SeverityPredictor()
    label, probs = predictor.predict(
        weather="Rainy",
        road_type="Highway",
        traffic_density="Low",
        vehicle_type="Truck",
        day_of_week="Friday",
        hour=19,
        time_of_day="Evening",
    )
    print(f"Predicted severity: {label}")
    print(f"Probabilities: {probs}")