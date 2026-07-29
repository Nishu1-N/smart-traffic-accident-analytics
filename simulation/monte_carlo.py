"""
monte_carlo.py
----------------
Defines realistic probability distributions for accident attributes
and provides sampling functions used by accident_generator.py.

The distributions here are informed, reasonable assumptions (not
derived from real Ranchi accident data, since that data is what
we're substituting for). They aim to reflect general patterns
documented in Indian road safety studies:
  - two-wheelers are overrepresented in accident counts
  - evening rush hour has the highest accident frequency
  - poor weather (rain/fog) increases severity risk
  - intersections/chowks see more (mostly minor) accidents
  - fatal accidents skew toward highways/main roads with low traffic
    (higher speeds) rather than congested intersections
"""

import random
import numpy as np
from datetime import date, timedelta, time

# ---------- Category weights (probabilities) ----------

ROAD_TYPES = ["Chowk/Intersection", "Main Road", "Highway", "Residential Road"]
ROAD_TYPE_WEIGHTS = [0.35, 0.30, 0.20, 0.15]

WEATHER_CONDITIONS = ["Clear", "Rainy", "Foggy", "Stormy"]
WEATHER_WEIGHTS = [0.65, 0.20, 0.10, 0.05]

TRAFFIC_DENSITY = ["Low", "Medium", "High"]
TRAFFIC_DENSITY_WEIGHTS = [0.25, 0.40, 0.35]

VEHICLE_TYPES = ["Two-Wheeler", "Car", "Auto-Rickshaw", "Bus", "Truck"]
VEHICLE_WEIGHTS = [0.45, 0.25, 0.15, 0.08, 0.07]

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def sample_datetime(start_date: date, end_date: date):
    """
    Sample a random accident date uniformly across the range,
    but sample the TIME using a distribution skewed toward
    morning (8-10 AM) and evening (5-8 PM) rush hours.
    """
    delta_days = (end_date - start_date).days
    random_day_offset = random.randint(0, delta_days)
    accident_date = start_date + timedelta(days=random_day_offset)

    # Hour distribution: weighted toward rush hours using a mixture
    # of two normal distributions (morning + evening peaks)
    if random.random() < 0.5:
        hour = int(np.clip(np.random.normal(loc=9, scale=1.5), 0, 23))   # morning peak
    else:
        hour = int(np.clip(np.random.normal(loc=18.5, scale=1.5), 0, 23))  # evening peak

    minute = random.randint(0, 59)
    accident_time = time(hour=hour, minute=minute)
    day_of_week = DAY_NAMES[accident_date.weekday()]

    return accident_date, accident_time, day_of_week


def sample_road_type():
    return random.choices(ROAD_TYPES, weights=ROAD_TYPE_WEIGHTS, k=1)[0]


def sample_weather():
    return random.choices(WEATHER_CONDITIONS, weights=WEATHER_WEIGHTS, k=1)[0]


def sample_traffic_density():
    return random.choices(TRAFFIC_DENSITY, weights=TRAFFIC_DENSITY_WEIGHTS, k=1)[0]


def sample_vehicle_type():
    return random.choices(VEHICLE_TYPES, weights=VEHICLE_WEIGHTS, k=1)[0]


def sample_severity(road_type: str, weather: str, traffic_density: str, vehicle_type: str):
    """
    Severity is NOT independent — it depends on the other sampled
    attributes. We start from a base distribution and adjust the
    weights based on risk-increasing conditions.
    """
    # Base weights: [Minor, Serious, Fatal]
    weights = np.array([0.60, 0.30, 0.10])

    # Highways + low traffic density => higher speed => more fatal risk
    if road_type == "Highway":
        weights += np.array([-0.10, 0.00, 0.10])
    if traffic_density == "Low":
        weights += np.array([-0.05, 0.00, 0.05])

    # Bad weather increases severity
    if weather in ("Rainy", "Foggy", "Stormy"):
        weights += np.array([-0.05, 0.03, 0.02])

    # Heavy vehicles (bus/truck) more likely to cause serious/fatal outcomes
    if vehicle_type in ("Bus", "Truck"):
        weights += np.array([-0.10, 0.05, 0.05])

    # Chowks/intersections: mostly low-speed, minor collisions
    if road_type == "Chowk/Intersection":
        weights += np.array([0.10, -0.05, -0.05])

    # Clip to valid probability range and renormalize
    weights = np.clip(weights, 0.01, None)
    weights = weights / weights.sum()

    severity = np.random.choice(["Minor", "Serious", "Fatal"], p=weights)
    return severity


def sample_casualties(severity: str):
    """Number of casualties correlated with severity."""
    if severity == "Minor":
        return np.random.choice([0, 1], p=[0.7, 0.3])
    elif severity == "Serious":
        return np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
    else:  # Fatal
        return np.random.choice([1, 2, 3, 4], p=[0.55, 0.25, 0.13, 0.07])