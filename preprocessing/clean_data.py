"""
clean_data.py
---------------
Pulls the joined locations + accidents data from MySQL, checks for
and handles missing values / duplicates, and saves a clean CSV to
data/processed_data.csv for use in EDA and feature engineering.

Run from project root:
    python preprocessing/clean_data.py
"""

import os
import sys
import pandas as pd
from datetime import time as dt_time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_connection import get_db_connection

OUTPUT_PATH = os.path.join("data", "processed_data.csv")


def load_data_from_mysql() -> pd.DataFrame:
    """Join accidents with locations and load into a DataFrame."""
    conn = get_db_connection()

    query = """
        SELECT
            a.accident_id,
            a.location_id,
            l.location_name,
            l.latitude,
            l.longitude,
            a.accident_date,
            a.accident_time,
            a.day_of_week,
            a.weather_condition,
            a.road_type,
            a.traffic_density,
            a.vehicle_type,
            a.accident_severity,
            a.num_casualties
        FROM accidents a
        JOIN locations l ON a.location_id = l.location_id
    """

    df = pd.read_sql(query, conn)
    conn.close()
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, duplicates, and basic type fixes."""
    print(f"Raw rows loaded: {len(df)}")

    # 1. Check missing values
    missing_counts = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing_counts[missing_counts > 0] if missing_counts.sum() > 0 else "  None found.")

    # Drop rows missing critical fields (defensive - shouldn't trigger on our synthetic data)
    critical_cols = ["location_id", "accident_date", "accident_severity"]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    if before != len(df):
        print(f"Dropped {before - len(df)} rows with missing critical fields.")

    # 2. Remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    if before != len(df):
        print(f"Dropped {before - len(df)} duplicate rows.")

    # 3. Fix data types
    df["accident_date"] = pd.to_datetime(df["accident_date"])

    # MySQL TIME columns can come back via the connector as either
    # plain "HH:MM:SS" strings or timedelta-style "0 days HH:MM:SS"
    # strings, depending on driver/version. pd.to_timedelta() handles
    # both formats safely, so we convert through that instead of
    # assuming a fixed string format.
    time_deltas = pd.to_timedelta(df["accident_time"].astype(str))
    total_seconds = time_deltas.dt.total_seconds().astype(int)
    df["accident_time"] = total_seconds.apply(
        lambda s: dt_time(hour=(s // 3600) % 24, minute=(s % 3600) // 60, second=s % 60)
    )

    # 4. Standardize text columns (strip whitespace, consistent casing)
    text_cols = ["weather_condition", "road_type", "traffic_density", "vehicle_type", "accident_severity"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    print(f"\nClean rows remaining: {len(df)}")
    return df


def main():
    print("Loading data from MySQL...")
    df = load_data_from_mysql()

    print("\nCleaning data...")
    df_clean = clean_data(df)

    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved cleaned dataset to {OUTPUT_PATH}")
    print(f"Shape: {df_clean.shape}")
    print("\nPreview:")
    print(df_clean.head())


if __name__ == "__main__":
    main()