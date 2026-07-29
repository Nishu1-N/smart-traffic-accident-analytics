"""
accident_generator.py
------------------------
Generates a synthetic accident dataset using Monte Carlo sampling
(see monte_carlo.py) and inserts it into the MySQL `accidents` table.

Each of the 50 locations is assigned a fixed "risk weight" so that
some locations naturally accumulate more accidents than others —
this creates realistic hotspots instead of a uniform spread, which
matters for the hotspot map and ML model in later phases.

Run from project root:
    python simulation/accident_generator.py
"""

import os
import sys
import random
from datetime import date

import numpy as np
from dotenv import load_dotenv

# allow importing sibling modules (database/, simulation/) when run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_connection import get_db_connection
from simulation.monte_carlo import (
    sample_datetime,
    sample_road_type,
    sample_weather,
    sample_traffic_density,
    sample_vehicle_type,
    sample_severity,
    sample_casualties,
)

load_dotenv()

# ---------- Config ----------
TOTAL_RECORDS = 8000          # within your planned 5,000-10,000 range
START_DATE = date(2021, 1, 1)
END_DATE = date(2025, 12, 31)
BATCH_SIZE = 500              # insert in batches for speed
RANDOM_SEED = 42              # reproducibility


def fetch_locations():
    """Get all location_ids from MySQL."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT location_id, location_name FROM locations")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


def assign_location_risk_weights(locations):
    """
    Assign each location a random but fixed 'risk weight' so accident
    counts aren't uniform across locations. Weights follow a lognormal
    distribution to create a realistic long-tail (few very risky
    hotspots, many moderate/low-risk locations).
    """
    np.random.seed(RANDOM_SEED)
    raw_weights = np.random.lognormal(mean=0, sigma=0.6, size=len(locations))
    normalized = raw_weights / raw_weights.sum()
    return {loc["location_id"]: w for loc, w in zip(locations, normalized)}


def generate_records(locations, location_weights, total_records):
    """Generate `total_records` synthetic accident rows as a list of tuples."""
    random.seed(RANDOM_SEED)

    location_ids = [loc["location_id"] for loc in locations]
    weights = [location_weights[lid] for lid in location_ids]

    # Pre-sample which location each accident belongs to, based on risk weights
    chosen_locations = random.choices(location_ids, weights=weights, k=total_records)

    records = []
    for location_id in chosen_locations:
        accident_date, accident_time, day_of_week = sample_datetime(START_DATE, END_DATE)
        road_type = sample_road_type()
        weather = sample_weather()
        traffic_density = sample_traffic_density()
        vehicle_type = sample_vehicle_type()
        severity = sample_severity(road_type, weather, traffic_density, vehicle_type)
        casualties = int(sample_casualties(severity))

        records.append((
            location_id,
            accident_date,
            accident_time,
            day_of_week,
            weather,
            road_type,
            traffic_density,
            vehicle_type,
            severity,
            casualties,
        ))

    return records


def insert_records(records, batch_size=BATCH_SIZE):
    """Insert generated records into MySQL in batches."""
    conn = get_db_connection()
    cursor = conn.cursor()

    insert_query = """
        INSERT INTO accidents (
            location_id, accident_date, accident_time, day_of_week,
            weather_condition, road_type, traffic_density,
            vehicle_type, accident_severity, num_casualties
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    total_inserted = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        cursor.executemany(insert_query, batch)
        conn.commit()
        total_inserted += len(batch)
        print(f"  Inserted {total_inserted}/{len(records)} records...")

    cursor.close()
    conn.close()
    return total_inserted


def main():
    print("Fetching locations from MySQL...")
    locations = fetch_locations()

    if not locations:
        print("No locations found. Run maps/osm_geocode.py first (Phase 3).")
        return

    print(f"Found {len(locations)} locations.")
    print("Assigning risk weights to each location...")
    location_weights = assign_location_risk_weights(locations)

    print(f"Generating {TOTAL_RECORDS} synthetic accident records via Monte Carlo simulation...")
    records = generate_records(locations, location_weights, TOTAL_RECORDS)

    print("Inserting records into MySQL...")
    total = insert_records(records)

    print(f"\nDone. Inserted {total} synthetic accident records into the `accidents` table.")


if __name__ == "__main__":
    main()