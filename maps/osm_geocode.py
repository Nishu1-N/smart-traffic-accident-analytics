"""
osm_geocode.py
----------------
Reads location names from data/ranchi_locations.csv, geocodes each one
using OpenStreetMap's free Nominatim API (via geopy), and stores the
results (name, latitude, longitude, formatted_address) into the MySQL
`locations` table.

No API key required. Nominatim's usage policy requires:
  - max 1 request per second
  - a descriptive User-Agent identifying the app/contact

Run from project root:
    python maps/osm_geocode.py
"""

import os
import time
import csv
import mysql.connector
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# ---------- Setup ----------
load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "traffic_accident_db")

CSV_PATH = os.path.join("data", "ranchi_locations.csv")

# IMPORTANT: Nominatim requires a real, descriptive user_agent.
# Replace the email/app name below with your own before running.
geolocator = Nominatim(user_agent="smart_traffic_analytics_ranchi_minor_project")

# Bounding box around Ranchi city (min_lat, min_lon, max_lat, max_lon).
# Restricting searches to this box prevents Nominatim from matching
# same-named villages/localities elsewhere in Jharkhand/India (e.g. a
# second "Jagannathpur" village in West Singhbhum district).
RANCHI_BOUNDING_BOX = [(23.28, 85.22), (23.45, 85.42)]


def get_db_connection():
    """Create and return a MySQL connection."""
    return mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
    )


def geocode_location(location_name: str, retries: int = 3):
    """
    Call Nominatim for a single location name.
    Returns a dict with name, lat, lng, formatted_address
    or None if geocoding failed after retries.
    """
    for attempt in range(1, retries + 1):
        try:
            result = geolocator.geocode(
                location_name,
                timeout=10,
                viewbox=RANCHI_BOUNDING_BOX,
                bounded=True,  # hard-restrict results to inside the bounding box
            )
            if not result:
                print(f"  [WARN] No result found for: {location_name}")
                return None

            return {
                "location_name": location_name,
                "latitude": result.latitude,
                "longitude": result.longitude,
                "formatted_address": result.address,
            }
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"  [RETRY {attempt}/{retries}] {location_name}: {e}")
            time.sleep(2)
        except Exception as e:
            print(f"  [ERROR] Failed to geocode '{location_name}': {e}")
            return None

    print(f"  [FAILED] Gave up on: {location_name}")
    return None


def save_to_mysql(records):
    """Insert a list of geocoded location dicts into MySQL."""
    conn = get_db_connection()
    cursor = conn.cursor()

    insert_query = """
        INSERT INTO locations (location_name, latitude, longitude, formatted_address)
        VALUES (%s, %s, %s, %s)
    """

    inserted = 0
    for rec in records:
        cursor.execute(
            insert_query,
            (
                rec["location_name"],
                rec["latitude"],
                rec["longitude"],
                rec["formatted_address"],
            ),
        )
        inserted += 1

    conn.commit()
    cursor.close()
    conn.close()
    print(f"\nInserted {inserted} locations into MySQL.")


def main():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        location_names = [row["location_name"].strip() for row in reader if row["location_name"].strip()]

    print(f"Found {len(location_names)} locations to geocode.\n")

    geocoded_records = []
    failed_names = []

    for i, name in enumerate(location_names, start=1):
        print(f"[{i}/{len(location_names)}] Geocoding: {name}")
        result = geocode_location(name)
        if result:
            geocoded_records.append(result)
        else:
            failed_names.append(name)

        # Respect Nominatim's 1 request/second rate limit
        time.sleep(1)

    print(f"\nSuccessfully geocoded {len(geocoded_records)} / {len(location_names)} locations.")

    if failed_names:
        print("\nThe following locations could not be geocoded automatically.")
        print("You may need to search them manually on openstreetmap.org and add coordinates by hand:")
        for name in failed_names:
            print(f"  - {name}")

    if geocoded_records:
        save_to_mysql(geocoded_records)
    else:
        print("No records to save.")


if __name__ == "__main__":
    main()