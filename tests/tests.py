"""
run_tests.py
--------------
Phase 10: Testing & Validation for the Smart Traffic Accident
Analytics project. Runs a series of simple, readable checks across
five areas: Database, Data Validation, Model Validation, Functional,
and Performance testing.

Each test prints PASS or FAIL with a short reason. At the end, a
summary shows how many tests passed out of the total - use this
summary directly in your report's "Testing & Validation" section.

Run from project root:
    python tests/run_tests.py
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

results = []  # (test_name, passed: bool, detail: str)


def check(test_name, condition, detail=""):
    """Record and print the result of a single test."""
    status = "PASS" if condition else "FAIL"
    results.append((test_name, condition, detail))
    print(f"[{status}] {test_name}" + (f" - {detail}" if detail else ""))


def section(title):
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


# =========================================================
# 1. DATABASE TESTING
# =========================================================
section("1. DATABASE TESTING")

try:
    from database.db_connection import get_db_connection

    conn = get_db_connection()
    check("MySQL connection established", conn.is_connected())

    cursor = conn.cursor()

    # Check locations table exists and has rows
    cursor.execute("SELECT COUNT(*) FROM locations")
    location_count = cursor.fetchone()[0]
    check(
        "locations table has data",
        location_count > 0,
        f"{location_count} rows found",
    )

    # Check accidents table exists and has rows
    cursor.execute("SELECT COUNT(*) FROM accidents")
    accident_count = cursor.fetchone()[0]
    check(
        "accidents table has data",
        accident_count > 0,
        f"{accident_count} rows found",
    )

    # Check every accident references a valid location (foreign key integrity)
    cursor.execute("""
        SELECT COUNT(*) FROM accidents a
        LEFT JOIN locations l ON a.location_id = l.location_id
        WHERE l.location_id IS NULL
    """)
    orphaned = cursor.fetchone()[0]
    check(
        "No orphaned accident records (all location_id values are valid)",
        orphaned == 0,
        f"{orphaned} orphaned rows" if orphaned else "all references valid",
    )

    cursor.close()
    conn.close()

except Exception as e:
    check("MySQL connection established", False, str(e))
    check("locations table has data", False, "skipped - connection failed")
    check("accidents table has data", False, "skipped - connection failed")
    check("No orphaned accident records", False, "skipped - connection failed")


# =========================================================
# 2. DATA VALIDATION TESTING
# =========================================================
section("2. DATA VALIDATION TESTING")

DATA_PATH = os.path.join("data", "eda_ready.csv")
ML_DATA_PATH = os.path.join("data", "ml_ready.csv")

try:
    df = pd.read_csv(DATA_PATH)
    check(f"{DATA_PATH} loads successfully", True, f"{len(df)} rows")

    # No missing values in critical columns
    critical_cols = ["location_name", "accident_severity", "weather_condition", "road_type"]
    missing = df[critical_cols].isnull().sum().sum()
    check(
        "No missing values in critical columns",
        missing == 0,
        f"{missing} missing values found" if missing else "0 missing values",
    )

    # Severity values are only the expected 3 categories
    valid_severities = {"Minor", "Serious", "Fatal"}
    actual_severities = set(df["accident_severity"].unique())
    check(
        "Severity column only contains Minor/Serious/Fatal",
        actual_severities.issubset(valid_severities),
        f"Found: {actual_severities}",
    )

    # Casualties should never be negative
    check(
        "num_casualties has no negative values",
        (df["num_casualties"] >= 0).all(),
    )

    # Hour should be within 0-23
    check(
        "hour column values are within 0-23",
        df["hour"].between(0, 23).all(),
    )

except Exception as e:
    check(f"{DATA_PATH} loads successfully", False, str(e))

try:
    ml_df = pd.read_csv(ML_DATA_PATH)
    check(f"{ML_DATA_PATH} loads successfully", True, f"{len(ml_df)} rows")

    # Encoded columns should be present
    expected_encoded_cols = [
        "weather_condition_encoded", "road_type_encoded",
        "traffic_density_encoded", "vehicle_type_encoded", "severity_encoded",
    ]
    missing_cols = [c for c in expected_encoded_cols if c not in ml_df.columns]
    check(
        "All expected encoded columns are present",
        len(missing_cols) == 0,
        f"Missing: {missing_cols}" if missing_cols else "all present",
    )

except Exception as e:
    check(f"{ML_DATA_PATH} loads successfully", False, str(e))


# =========================================================
# 3. MODEL VALIDATION TESTING
# =========================================================
section("3. MODEL VALIDATION TESTING")

try:
    from models.predict import SeverityPredictor

    start = time.time()
    predictor = SeverityPredictor()
    load_time = time.time() - start
    check("Model loads successfully", True, f"loaded in {load_time:.3f}s")

    # Run a test prediction and confirm output format
    label, probs = predictor.predict(
        weather="Rainy", road_type="Highway", traffic_density="Low",
        vehicle_type="Truck", day_of_week="Friday", hour=19, time_of_day="Evening",
    )
    check(
        "Prediction returns a valid severity label",
        label in ("Minor", "Serious", "Fatal"),
        f"Predicted: {label}",
    )
    check(
        "Prediction probabilities sum to approximately 1.0",
        abs(sum(probs.values()) - 1.0) < 0.01 if probs else False,
        f"Sum: {sum(probs.values()):.3f}" if probs else "no probabilities returned",
    )

    # Check saved test accuracy meets a reasonable minimum threshold
    import joblib
    metadata = joblib.load(os.path.join("models", "model_metadata.pkl"))
    test_accuracy = metadata.get("test_accuracy", 0)
    check(
        "Model meets minimum accuracy threshold (>= 70%)",
        test_accuracy >= 0.70,
        f"Test accuracy: {test_accuracy*100:.1f}%",
    )

except Exception as e:
    check("Model loads successfully", False, str(e))


# =========================================================
# 4. FUNCTIONAL TESTING
# =========================================================
section("4. FUNCTIONAL TESTING")

required_files = [
    "dashboard/app.py",
    "database/db_connection.py",
    "simulation/monte_carlo.py",
    "simulation/accident_generator.py",
    "preprocessing/clean_data.py",
    "preprocessing/feature_engineering.py",
    "models/train_model.py",
    "models/predict.py",
    "models/model.pkl",
    "models/model_metadata.pkl",
    "data/eda_ready.csv",
    "data/ml_ready.csv",
]

for filepath in required_files:
    exists = os.path.isfile(filepath)
    check(f"Required file exists: {filepath}", exists)

# Confirm all dashboard page files are syntactically valid Python
import py_compile

dashboard_pages_dir = os.path.join("dashboard", "pages")
if os.path.isdir(dashboard_pages_dir):
    for filename in os.listdir(dashboard_pages_dir):
        if filename.endswith(".py"):
            filepath = os.path.join(dashboard_pages_dir, filename)
            try:
                py_compile.compile(filepath, doraise=True)
                check(f"Valid Python syntax: {filename}", True)
            except py_compile.PyCompileError as e:
                check(f"Valid Python syntax: {filename}", False, str(e))


# =========================================================
# 5. PERFORMANCE TESTING
# =========================================================
section("5. PERFORMANCE TESTING")

try:
    start = time.time()
    _ = pd.read_csv(DATA_PATH)
    load_time = time.time() - start
    check(
        "eda_ready.csv loads in under 2 seconds",
        load_time < 2.0,
        f"{load_time:.3f}s",
    )
except Exception as e:
    check("eda_ready.csv loads in under 2 seconds", False, str(e))

try:
    from models.predict import SeverityPredictor
    predictor = SeverityPredictor()
    start = time.time()
    for _ in range(10):
        predictor.predict(
            weather="Clear", road_type="Main Road", traffic_density="Medium",
            vehicle_type="Car", day_of_week="Monday", hour=14, time_of_day="Afternoon",
        )
    total_time = time.time() - start
    avg_time = total_time / 10
    check(
        "Average prediction time is under 0.5 seconds",
        avg_time < 0.5,
        f"{avg_time*1000:.1f}ms average over 10 predictions",
    )
except Exception as e:
    check("Average prediction time is under 0.5 seconds", False, str(e))


# =========================================================
# SUMMARY
# =========================================================
section("TEST SUMMARY")

total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = total - passed

print(f"\nTotal tests run: {total}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print(f"Pass rate: {(passed/total*100):.1f}%\n")

if failed > 0:
    print("Failed tests:")
    for name, ok, detail in results:
        if not ok:
            print(f"  - {name}" + (f" ({detail})" if detail else ""))

print("\n" + "=" * 60)
print("Use this summary in your report's Testing & Validation section.")
print("=" * 60)