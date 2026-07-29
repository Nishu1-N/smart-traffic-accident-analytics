"""
train_model.py
-----------------
Trains and compares three classification models (Decision Tree,
Random Forest, XGBoost) to predict accident severity (Minor / Serious
/ Fatal) from accident conditions. Evaluates each with standard
classification metrics, then saves the best-performing model plus
the label encoders needed to use it later in the dashboard.

Run from project root:
    python models/train_model.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("xgboost not installed - skipping XGBoost model. Run: pip install xgboost")

INPUT_PATH = os.path.join("data", "ml_ready.csv")
MODEL_OUTPUT_PATH = os.path.join("models", "model.pkl")
METADATA_OUTPUT_PATH = os.path.join("models", "model_metadata.pkl")

# Features used to predict severity - all numeric/encoded columns
FEATURE_COLS = [
    "weather_condition_encoded",
    "road_type_encoded",
    "traffic_density_encoded",
    "vehicle_type_encoded",
    "day_of_week_encoded",
    "time_of_day_encoded",
    "hour",
    "is_weekend",
    "is_rush_hour",
    "is_adverse_weather",
]
TARGET_COL = "severity_encoded"
SEVERITY_LABELS = {0: "Minor", 1: "Serious", 2: "Fatal"}


def load_data():
    df = pd.read_csv(INPUT_PATH)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y, df


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n{'=' * 55}")
    print(f"MODEL: {name}")
    print(f"{'=' * 55}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    labels = [SEVERITY_LABELS[i] for i in sorted(SEVERITY_LABELS.keys())]
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df.to_string())

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))

    return {"name": name, "accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "model": model}


def main():
    print(f"Loading {INPUT_PATH}...")
    X, y, df = load_data()

    print(f"Dataset shape: {X.shape}")
    print(f"Features used: {FEATURE_COLS}")
    print(f"Target distribution:\n{y.value_counts().rename(SEVERITY_LABELS).to_string()}")

    # 80/20 train-test split, stratified so severity classes stay balanced in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")

    results = []

    # --- Model 1: Decision Tree ---
    dt = DecisionTreeClassifier(max_depth=8, random_state=42, class_weight="balanced")
    dt.fit(X_train, y_train)
    results.append(evaluate_model("Decision Tree", dt, X_test, y_test))

    # --- Model 2: Random Forest ---
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    results.append(evaluate_model("Random Forest", rf, X_test, y_test))

    # --- Model 3: XGBoost (optional) ---
    if XGBOOST_AVAILABLE:
        xgb = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric="mlogloss"
        )
        xgb.fit(X_train, y_train)
        results.append(evaluate_model("XGBoost", xgb, X_test, y_test))

    # --- Pick best model by F1-score (better than accuracy for imbalanced classes) ---
    best = max(results, key=lambda r: r["f1"])
    print(f"\n{'#' * 55}")
    print(f"BEST MODEL: {best['name']} (F1-Score: {best['f1']:.4f})")
    print(f"{'#' * 55}")

    # --- Feature importance (for tree-based models) ---
    if hasattr(best["model"], "feature_importances_"):
        importances = pd.Series(best["model"].feature_importances_, index=FEATURE_COLS)
        importances = importances.sort_values(ascending=False)
        print("\nFeature Importance:")
        print(importances.to_string())

    # --- Save the best model + metadata needed to use it later ---
    os.makedirs("models", exist_ok=True)
    joblib.dump(best["model"], MODEL_OUTPUT_PATH)

    metadata = {
        "model_name": best["name"],
        "feature_cols": FEATURE_COLS,
        "severity_labels": SEVERITY_LABELS,
        "test_accuracy": best["accuracy"],
        "test_f1": best["f1"],
    }
    joblib.dump(metadata, METADATA_OUTPUT_PATH)

    print(f"\nSaved best model to {MODEL_OUTPUT_PATH}")
    print(f"Saved model metadata to {METADATA_OUTPUT_PATH}")


if __name__ == "__main__":
    main()