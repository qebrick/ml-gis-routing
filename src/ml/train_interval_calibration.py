from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error

INPUT_PATH = Path(
    "data/processed/ml_interval_features.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/ml_interval_calibration_predictions.parquet"
)

df = pd.read_parquet(INPUT_PATH).copy()

features = [
    "mean_osm_speed_kmh",
    "share_motorway",
    "share_primary",
    "share_secondary",
    "share_tertiary",
    "share_residential",
    "share_living_street",
    "share_other",
]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(
        df,
        groups=df["trip_id"],
    )
)

train = df.iloc[train_idx].copy()
test = df.iloc[test_idx].copy()

X_train = train[features]
X_test = test[features]

y_train = np.log(train["target_time_ratio"])

model = CatBoostRegressor(
    iterations=400,
    depth=3,
    learning_rate=0.03,
    loss_function="RMSE",
    random_seed=42,
    verbose=False,
    thread_count=4,
)

model.fit(X_train, y_train)

# OSM baseline.
test["pred_osm_s"] = test["osm_time_s"]

# Train-only global multiplicative calibration.
global_ratio = np.exp(
    np.median(y_train)
)

test["pred_global_s"] = (
    test["osm_time_s"] * global_ratio
)

# ML multiplicative calibration.
pred_log_ratio = model.predict(X_test)

test["pred_ml_s"] = (
    test["osm_time_s"]
    * np.exp(pred_log_ratio)
)

actual = test["observed_time_s"]

print("=== DATA SPLIT ===")
print("Train intervals:", len(train))
print("Test intervals:", len(test))
print("Train trips:", train["trip_id"].nunique())
print("Test trips:", test["trip_id"].nunique())
print(
    "Overlapping trips:",
    len(
        set(train["trip_id"])
        & set(test["trip_id"])
    ),
)

print("\n=== CALIBRATION ===")
print(
    "Train median ratio:",
    round(global_ratio, 3)
)

print("\n=== TEST RESULTS ===")

for name, column in [
    ("OSM", "pred_osm_s"),
    ("Global calibration", "pred_global_s"),
    ("CatBoost calibration", "pred_ml_s"),
]:
    predictions = test[column]

    mae = mean_absolute_error(
        actual, predictions
    )

    bias = (
        predictions - actual
    ).mean()

    print(
        f"{name:22s} "
        f"MAE={mae:.3f}s "
        f"bias={bias:.3f}s"
    )

print("\n=== FEATURE IMPORTANCE ===")

importance = pd.Series(
    model.feature_importances_,
    index=features,
).sort_values(ascending=False)

print(importance.round(2).to_string())

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

test.to_parquet(
    OUTPUT_PATH,
    index=False,
)

print("\nSaved:", OUTPUT_PATH)
