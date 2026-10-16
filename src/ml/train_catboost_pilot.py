from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = Path("data/processed/ml_training_data.parquet")
MODEL_PATH = Path("models/catboost_speed_pilot.cbm")
PREDICTIONS_PATH = Path("data/processed/ml_predictions_pilot.parquet")

FEATURES = [
    "edge_length_m",
    "highway",
    "maxspeed_kmh",
    "lanes",
    "oneway",
    "hour",
    "day_of_week",
    "has_maxspeed",
    "has_lanes",
]

CATEGORICAL = ["highway", "oneway"]
MIN_SPEED_KMH = 3.0
RANDOM_STATE = 42

data = pd.read_parquet(DATA_PATH)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=RANDOM_STATE,
)

train_idx, test_idx = next(
    splitter.split(
        data,
        groups=data["trip_id"],
    )
)

train = data.iloc[train_idx].copy()
test = data.iloc[test_idx].copy()

assert set(train["trip_id"]).isdisjoint(
    set(test["trip_id"])
)

X_train = train[FEATURES].copy()
X_test = test[FEATURES].copy()

for col in CATEGORICAL:
    X_train[col] = X_train[col].fillna("unknown").astype(str)
    X_test[col] = X_test[col].fillna("unknown").astype(str)

y_train = train["target_speed_kmh"]
y_test = test["target_speed_kmh"]

baseline_speed = float(y_train.median())

baseline_pred = np.full(
    len(test),
    baseline_speed,
)

model = CatBoostRegressor(
    iterations=500,
    depth=4,
    learning_rate=0.03,
    loss_function="MAE",
    random_seed=RANDOM_STATE,
    verbose=False,
    thread_count=4,
    l2_leaf_reg=5,
)

model.fit(
    X_train,
    y_train,
    cat_features=CATEGORICAL,
)

catboost_pred = model.predict(X_test)

baseline_pred = np.maximum(
    baseline_pred,
    MIN_SPEED_KMH,
)
catboost_pred = np.maximum(
    catboost_pred,
    MIN_SPEED_KMH,
)

def evaluate(name, predicted_speed):
    speed_mae = mean_absolute_error(
        y_test,
        predicted_speed,
    )

    speed_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predicted_speed,
        )
    )

    # Evaluate time over the OBSERVED partial edge length.
    length_m = test["traversed_length_m"].to_numpy()

    actual_time = test["observed_time_seconds"].to_numpy()

    predicted_time = (
        3.6 * length_m / predicted_speed
    )

    time_mae = mean_absolute_error(
        actual_time,
        predicted_time,
    )

    print(
        f"{name:12s} | "
        f"Speed MAE: {speed_mae:7.3f} km/h | "
        f"Speed RMSE: {speed_rmse:7.3f} km/h | "
        f"Time MAE: {time_mae:7.3f} s"
    )

    return predicted_time


print("=== DATA SPLIT ===")
print("Train rows:", len(train))
print("Test rows:", len(test))
print("Train trips:", train["trip_id"].nunique())
print("Test trips:", test["trip_id"].nunique())
print("Baseline median speed:", round(baseline_speed, 3), "km/h")

print("\n=== MODEL COMPARISON ===")
baseline_time = evaluate(
    "Baseline",
    baseline_pred,
)
catboost_time = evaluate(
    "CatBoost",
    catboost_pred,
)

print("\n=== FEATURE IMPORTANCE ===")
importance = pd.Series(
    model.feature_importances_,
    index=FEATURES,
).sort_values(ascending=False)

print(importance.to_string())

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
model.save_model(str(MODEL_PATH))

test["baseline_speed_kmh"] = baseline_pred
test["catboost_speed_kmh"] = catboost_pred
test["baseline_time_seconds"] = baseline_time
test["catboost_time_seconds"] = catboost_time

PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
test.to_parquet(PREDICTIONS_PATH, index=False)

print("\nSaved model:", MODEL_PATH)
print("Saved predictions:", PREDICTIONS_PATH)
