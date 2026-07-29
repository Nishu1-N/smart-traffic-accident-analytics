"""
analysis.py
-------------
Computes key statistical insights from data/eda_ready.csv and prints
a report-ready summary. Complements visualization.py (which produces
the charts) with the numbers behind them.

Run from project root:
    python eda/analysis.py
"""

import os
import pandas as pd

INPUT_PATH = os.path.join("data", "eda_ready.csv")


def main():
    df = pd.read_csv(INPUT_PATH)
    df["accident_date"] = pd.to_datetime(df["accident_date"])

    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS - KEY INSIGHTS")
    print("=" * 60)

    # 1. Overall stats
    print(f"\nTotal accident records: {len(df)}")
    print(f"Date range: {df['accident_date'].min().date()} to {df['accident_date'].max().date()}")
    print(f"Number of unique locations: {df['location_name'].nunique()}")

    # 2. Severity breakdown
    print("\n--- Severity Breakdown ---")
    severity_pct = (df["accident_severity"].value_counts(normalize=True) * 100).round(1)
    print(severity_pct.to_string())

    # 3. Top hotspots
    print("\n--- Top 10 Hotspot Locations ---")
    print(df["location_name"].value_counts().head(10).to_string())

    # 4. Fatal accidents by road type
    print("\n--- Fatal Accident Rate by Road Type ---")
    fatal_rate = df.groupby("road_type")["accident_severity"].apply(
        lambda x: (x == "Fatal").mean() * 100
    ).round(1).sort_values(ascending=False)
    print(fatal_rate.to_string())

    # 5. Peak accident hour
    print("\n--- Peak Accident Hours (Top 5) ---")
    print(df["hour"].value_counts().head(5).to_string())

    # 6. Weather impact
    print("\n--- Fatal Accident Rate by Weather ---")
    weather_fatal = df.groupby("weather_condition")["accident_severity"].apply(
        lambda x: (x == "Fatal").mean() * 100
    ).round(1).sort_values(ascending=False)
    print(weather_fatal.to_string())

    # 7. Vehicle type risk
    print("\n--- Fatal Accident Rate by Vehicle Type ---")
    vehicle_fatal = df.groupby("vehicle_type")["accident_severity"].apply(
        lambda x: (x == "Fatal").mean() * 100
    ).round(1).sort_values(ascending=False)
    print(vehicle_fatal.to_string())

    # 8. Weekend vs weekday
    print("\n--- Weekend vs Weekday Accident Counts ---")
    print(df["is_weekend"].value_counts().rename({True: "Weekend", False: "Weekday"}).to_string())

    # 9. Casualties
    print(f"\n--- Casualties ---")
    print(f"Total casualties: {df['num_casualties'].sum()}")
    print(f"Average per accident: {df['num_casualties'].mean():.2f}")
    print(f"Max in a single accident: {df['num_casualties'].max()}")

    print("\n" + "=" * 60)
    print("Use these numbers directly in your report's EDA section.")
    print("=" * 60)


if __name__ == "__main__":
    main()