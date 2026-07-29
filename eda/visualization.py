"""
visualization.py
-------------------
Generates all key exploratory charts from data/eda_ready.csv and
saves them as PNG files in reports/figures/. These charts feed
directly into your final report, PPT, and the dashboard's
Analytics page.

Run from project root:
    python eda/visualization.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_PATH = os.path.join("data", "eda_ready.csv")
OUTPUT_DIR = os.path.join("reports", "figures")

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_fig(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_accidents_by_location(df):
    top_locations = df["location_name"].value_counts().head(15)
    fig, ax = plt.subplots()
    sns.barplot(x=top_locations.values, y=top_locations.index, ax=ax, palette="Reds_r")
    ax.set_title("Top 15 Accident Hotspot Locations")
    ax.set_xlabel("Number of Accidents")
    ax.set_ylabel("Location")
    save_fig(fig, "01_accidents_by_location.png")


def plot_accidents_by_hour(df):
    fig, ax = plt.subplots()
    sns.histplot(df["hour"], bins=24, kde=True, ax=ax, color="steelblue")
    ax.set_title("Accidents by Hour of Day")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Number of Accidents")

    # Show AM/PM labels instead of 24-hour numbers, at every 2 hours to avoid clutter
    tick_hours = list(range(0, 24, 2))
    def hour_to_ampm(h):
        period = "AM" if h < 12 else "PM"
        display_hour = h % 12
        if display_hour == 0:
            display_hour = 12
        return f"{display_hour} {period}"
    ax.set_xticks(tick_hours)
    ax.set_xticklabels([hour_to_ampm(h) for h in tick_hours], rotation=45)

    save_fig(fig, "02_accidents_by_hour.png")


def plot_accidents_by_weather(df):
    fig, ax = plt.subplots()
    order = df["weather_condition"].value_counts().index
    sns.countplot(x="weather_condition", data=df, order=order, ax=ax, palette="Blues_r")
    ax.set_title("Accidents by Weather Condition")
    ax.set_xlabel("Weather Condition")
    ax.set_ylabel("Number of Accidents")
    save_fig(fig, "03_accidents_by_weather.png")


def plot_vehicle_type_analysis(df):
    fig, ax = plt.subplots()
    order = df["vehicle_type"].value_counts().index
    sns.countplot(x="vehicle_type", data=df, order=order, ax=ax, palette="Greens_r")
    ax.set_title("Accidents by Vehicle Type")
    ax.set_xlabel("Vehicle Type")
    ax.set_ylabel("Number of Accidents")
    plt.xticks(rotation=20)
    save_fig(fig, "04_accidents_by_vehicle_type.png")


def plot_road_type_analysis(df):
    fig, ax = plt.subplots()
    order = df["road_type"].value_counts().index
    sns.countplot(x="road_type", data=df, order=order, ax=ax, palette="Purples_r")
    ax.set_title("Accidents by Road Type")
    ax.set_xlabel("Road Type")
    ax.set_ylabel("Number of Accidents")
    plt.xticks(rotation=15)
    save_fig(fig, "05_accidents_by_road_type.png")


def plot_severity_distribution(df):
    fig, ax = plt.subplots()
    order = ["Minor", "Serious", "Fatal"]
    counts = df["accident_severity"].value_counts().reindex(order)
    colors = ["#4CAF50", "#FF9800", "#F44336"]
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", colors=colors, startangle=90)
    ax.set_title("Accident Severity Distribution")
    save_fig(fig, "06_severity_distribution.png")


def plot_monthly_trends(df):
    monthly = df.groupby([df["accident_date"].dt.to_period("M")]).size()
    fig, ax = plt.subplots(figsize=(14, 6))
    monthly.index = monthly.index.astype(str)
    ax.plot(monthly.index, monthly.values, marker="o", linewidth=1.5, color="darkorange")
    ax.set_title("Monthly Accident Trend (2021-2025)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Accidents")
    plt.xticks(rotation=90, fontsize=7)
    save_fig(fig, "07_monthly_trends.png")


def plot_severity_by_road_type(df):
    fig, ax = plt.subplots()
    cross = pd.crosstab(df["road_type"], df["accident_severity"], normalize="index") * 100
    cross = cross[["Minor", "Serious", "Fatal"]]
    cross.plot(kind="bar", stacked=True, ax=ax, color=["#4CAF50", "#FF9800", "#F44336"])
    ax.set_title("Severity Breakdown by Road Type (%)")
    ax.set_xlabel("Road Type")
    ax.set_ylabel("Percentage of Accidents")
    plt.xticks(rotation=15)
    ax.legend(title="Severity")
    save_fig(fig, "08_severity_by_road_type.png")


def plot_correlation_matrix(df):
    numeric_cols = ["hour", "month", "year", "is_weekend", "is_rush_hour", "is_adverse_weather", "num_casualties"]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax, fmt=".2f")
    ax.set_title("Correlation Matrix of Numeric Features")
    save_fig(fig, "09_correlation_matrix.png")


def plot_rush_hour_vs_severity(df):
    fig, ax = plt.subplots()
    cross = pd.crosstab(df["is_rush_hour"], df["accident_severity"], normalize="index") * 100
    cross.index = ["Non-Rush Hour", "Rush Hour"]
    cross = cross[["Minor", "Serious", "Fatal"]]
    cross.plot(kind="bar", stacked=True, ax=ax, color=["#4CAF50", "#FF9800", "#F44336"])
    ax.set_title("Severity: Rush Hour vs Non-Rush Hour (%)")
    ax.set_ylabel("Percentage")
    plt.xticks(rotation=0)
    save_fig(fig, "10_rush_hour_vs_severity.png")


def print_summary_stats(df):
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    print(f"Total accidents: {len(df)}")
    print(f"Date range: {df['accident_date'].min().date()} to {df['accident_date'].max().date()}")
    print(f"\nSeverity breakdown:\n{df['accident_severity'].value_counts()}")
    print(f"\nTop 5 hotspot locations:\n{df['location_name'].value_counts().head()}")
    print(f"\nTotal casualties: {df['num_casualties'].sum()}")
    print(f"Average casualties per accident: {df['num_casualties'].mean():.2f}")
    print("=" * 50)


def main():
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    df["accident_date"] = pd.to_datetime(df["accident_date"])

    ensure_output_dir()
    print(f"\nGenerating charts, saving to {OUTPUT_DIR}/ ...")

    plot_accidents_by_location(df)
    plot_accidents_by_hour(df)
    plot_accidents_by_weather(df)
    plot_vehicle_type_analysis(df)
    plot_road_type_analysis(df)
    plot_severity_distribution(df)
    plot_monthly_trends(df)
    plot_severity_by_road_type(df)
    plot_correlation_matrix(df)
    plot_rush_hour_vs_severity(df)

    print_summary_stats(df)
    print("\nAll charts generated successfully.")


if __name__ == "__main__":
    main()