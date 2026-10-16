from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

INPUT_PATH = Path(
    "data/processed/ml_distance_based_routes.parquet"
)

df = pd.read_parquet(INPUT_PATH)

# One distance threshold only: avoids overlap
# between different threshold datasets.
df = df.loc[
    df["target_distance_m"] == 500
].copy()

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

X_train = train[["osm_time_s"]]
X_test = test[["osm_time_s"]]

y_train = train["observed_time_s"]
y_test = test["observed_time_s"]

# 1. Original OSM baseline.
test["pred_osm_s"] = test["osm_time_s"]

# 2. Multiplicative calibration.
# Ratio of total observed to total OSM time:
# train-only, weighted by OSM duration.
factor = (
    train["observed_time_s"].sum()
    / train["osm_time_s"].sum()
)

test["pred_scaled_s"] = (
    factor * test["osm_time_s"]
)

# 3. Linear calibration with intercept.
linear = LinearRegression()
linear.fit(X_train, y_train)

test["pred_linear_s"] = np.maximum(
    linear.predict(X_test),
    0.0,
)

print("=== DATA SPLIT ===")
print("Train routes:", len(train))
print("Test routes:", len(test))
print("Train trips:", train["trip_id"].nunique())
print("Test trips:", test["trip_id"].nunique())
print(
    "Overlapping trips:",
    len(
        set(train["trip_id"])
        & set(test["trip_id"])
    ),
)

print("\n=== CALIBRATION PARAMETERS ===")
print("Multiplicative factor:", round(factor, 4))
print("Linear intercept:", round(linear.intercept_, 4))
print("Linear slope:", round(linear.coef_[0], 4))

print("\n=== TEST RESULTS ===")

for name, column in [
    ("OSM", "pred_osm_s"),
    ("Scaled OSM", "pred_scaled_s"),
    ("Linear OSM", "pred_linear_s"),
]:
    pred = test[column]

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(
        mean_squared_error(y_test, pred)
    )
    bias = (pred - y_test).mean()

    print(
        f"{name:15s} "
        f"MAE={mae:.3f}s "
        f"RMSE={rmse:.3f}s "
        f"bias={bias:.3f}s"
    )

print("\n=== TEST TARGET DISTRIBUTION ===")
print(
    y_test.describe(
        percentiles=[0.25, 0.5, 0.75, 0.9]
    ).round(2).to_string()
)
