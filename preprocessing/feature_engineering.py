"""
feature_engineering.py
-------------------------
Reads data/processed_data.csv (output of clean_data.py), adds derived
features useful for EDA and ML, encodes categorical columns, and
saves two outputs:
  - data/eda_ready.csv       (human-readable, for charts/EDA)
  - data/ml_ready.csv        (numerically encoded, for model training)

Run from project root:
    python preprocessing/feature_engineering.py
"""

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

INPUT_PATH = os.path.join("data", "processed_data.csv")
EDA_OUTPUT_PATH = os.path.join("data", "eda_ready.csv")
ML_OUTPUT_PATH = os.path.join("data", "ml_ready.csv")


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable derived columns useful for EDA and modeling."""
    df["accident_date"] = pd.to_datetime(df["accident_date"])

    # accident_time may be saved as "HH:MM:SS" or "0 days HH:MM:SS"
    # depending on the source - pd.to_timedelta handles both safely.
    time_deltas = pd.to_timedelta(df["accident_time"].astype(str))
    total_seconds = time_deltas.dt.total_seconds().astype(int)
    df["hour"] = (total_seconds // 3600) % 24

    # Is it a weekend?
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"])

    # Time-of-day bucket
    def time_bucket(hour):
        if 5 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 21:
            return "Evening"
        else:
            return "Night"

    df["time_of_day"] = df["hour"].apply(time_bucket)

    # Is it rush hour? (8-10 AM or 5-8 PM)
    df["is_rush_hour"] = df["hour"].apply(lambda h: (8 <= h <= 10) or (17 <= h <= 20))

    # Month and year for trend analysis
    df["month"] = df["accident_date"].dt.month
    df["year"] = df["accident_date"].dt.year

    # Is the weather adverse?
    df["is_adverse_weather"] = df["weather_condition"].isin(["Rainy", "Foggy", "Stormy"])

    return df


def encode_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a numerically-encoded copy of the dataset for machine learning.
    Uses Label Encoding for categorical columns (simple, works well with
    tree-based models like Random Forest / XGBoost which we'll use in Phase 8).
    """
    ml_df = df.copy()

    categorical_cols = [
        "weather_condition", "road_type", "traffic_density",
        "vehicle_type", "day_of_week", "time_of_day"
    ]

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        ml_df[col + "_encoded"] = le.fit_transform(ml_df[col])
        encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    # Target variable: accident_severity -> numeric encoding
    severity_order = {"Minor": 0, "Serious": 1, "Fatal": 2}
    ml_df["severity_encoded"] = ml_df["accident_severity"].map(severity_order)

    # Convert booleans to 0/1
    for col in ["is_weekend", "is_rush_hour", "is_adverse_weather"]:
        ml_df[col] = ml_df[col].astype(int)

    print("\nCategory encodings used:")
    for col, mapping in encoders.items():
        print(f"  {col}: {mapping}")
    print(f"  accident_severity: {severity_order}")

    return ml_df


def main():
    print(f"Loading cleaned data from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)

    print("Adding derived features...")
    df = add_derived_features(df)
    df.to_csv(EDA_OUTPUT_PATH, index=False)
    print(f"Saved EDA-ready dataset to {EDA_OUTPUT_PATH} — shape: {df.shape}")

    print("\nEncoding categorical features for ML...")
    ml_df = encode_for_ml(df)
    ml_df.to_csv(ML_OUTPUT_PATH, index=False)
    print(f"\nSaved ML-ready dataset to {ML_OUTPUT_PATH} — shape: {ml_df.shape}")

    print("\nPreview of EDA-ready data:")
    print(df[["location_name", "hour", "time_of_day", "is_weekend", "is_rush_hour", "accident_severity"]].head())


if __name__ == "__main__":
    main()