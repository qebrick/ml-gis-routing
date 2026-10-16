from pathlib import Path

import numpy as np
import osmnx as ox
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

GRAPH_PATH = Path(
    "data/graphs/porto_drive_conservative_refit.graphml"
)
INPUT_PATH = Path(
    "data/processed/alt_heuristic_diagnostics.parquet"
)
OUTPUT_PATH = Path(
    "data/processed/ml_alt_gap_predictions.parquet"
)

SEED = 42

G = ox.load_graphml(GRAPH_PATH)
df = pd.read_parquet(INPUT_PATH)

coords = {
    int(node): (
        float(attrs["y"]),
        float(attrs["x"]),
    )
    for node, attrs in G.nodes(data=True)
}

node_lat = df["node"].map(
    lambda n: coords[int(n)][0]
)
node_lon = df["node"].map(
    lambda n: coords[int(n)][1]
)

target_lat = df["target"].map(
    lambda n: coords[int(n)][0]
)
target_lon = df["target"].map(
    lambda n: coords[int(n)][1]
)

df["node_lat"] = node_lat
df["node_lon"] = node_lon
df["target_lat"] = target_lat
df["target_lon"] = target_lon

df["delta_lat"] = (
    df["target_lat"] - df["node_lat"]
)
df["delta_lon"] = (
    df["target_lon"] - df["node_lon"]
)

df["abs_delta_lat"] = df["delta_lat"].abs()
df["abs_delta_lon"] = df["delta_lon"].abs()

features = [
    "alt_cost_s",
    "node_lat",
    "node_lon",
    "target_lat",
    "target_lon",
    "delta_lat",
    "delta_lon",
    "abs_delta_lat",
    "abs_delta_lon",
]

target_column = "gap_s"

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=SEED,
)

train_idx, test_idx = next(
    splitter.split(
        df,
        groups=df["target"],
    )
)

train = df.iloc[train_idx].copy()
test = df.iloc[test_idx].copy()

assert set(train["target"]).isdisjoint(
    set(test["target"])
)

model = CatBoostRegressor(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="MAE",
    random_seed=SEED,
    verbose=False,
    thread_count=-1,
)

model.fit(
    train[features],
    train[target_column],
)

predicted_gap = model.predict(
    test[features]
)

# Gap is nonnegative by definition.
predicted_gap = np.maximum(
    predicted_gap,
    0.0,
)

test["predicted_gap_s"] = predicted_gap

test["ml_predicted_cost_s"] = (
    test["alt_cost_s"]
    + test["predicted_gap_s"]
)

test["ml_gap_error_s"] = (
    test["predicted_gap_s"]
    - test["gap_s"]
)

test["ml_cost_error_s"] = (
    test["ml_predicted_cost_s"]
    - test["true_cost_s"]
)

# A model predicting a positive gap can exceed
# the true shortest-path cost. Such predictions
# are NOT admissible A* heuristics.
test["ml_overestimates"] = (
    test["ml_predicted_cost_s"]
    > test["true_cost_s"] + 1e-6
)

alt_mae = mean_absolute_error(
    test["true_cost_s"],
    test["alt_cost_s"],
)

ml_mae = mean_absolute_error(
    test["true_cost_s"],
    test["ml_predicted_cost_s"],
)

gap_rmse = np.sqrt(
    mean_squared_error(
        test["gap_s"],
        test["predicted_gap_s"],
    )
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

test.to_parquet(
    OUTPUT_PATH,
    index=False,
)

print("=== DATA SPLIT ===")
print("Train pairs:", len(train))
print("Test pairs:", len(test))
print(
    "Train targets:",
    train["target"].nunique(),
)
print(
    "Test targets:",
    test["target"].nunique(),
)

print("\n=== GAP PREDICTION ===")
print(
    "Gap MAE (s):",
    round(
        mean_absolute_error(
            test["gap_s"],
            test["predicted_gap_s"],
        ),
        3,
    ),
)
print(
    "Gap RMSE (s):",
    round(gap_rmse, 3),
)

print("\n=== REMAINING COST PREDICTION ===")
print(
    "ALT MAE (s):",
    round(alt_mae, 3),
)
print(
    "ML MAE (s):",
    round(ml_mae, 3),
)
print(
    "MAE improvement (%):",
    round(
        100 * (alt_mae - ml_mae) / alt_mae,
        3,
    ),
)

print("\n=== ADMISSIBILITY DIAGNOSTIC ===")
print(
    "ML overestimation rate (%):",
    round(
        100 * test["ml_overestimates"].mean(),
        3,
    ),
)
print(
    "Mean overestimation (s), "
    "among violations:",
    round(
        test.loc[
            test["ml_overestimates"],
            "ml_cost_error_s",
        ].mean(),
        3,
    ),
)

print("\n=== FEATURE IMPORTANCE ===")
importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_,
})

print(
    importance.sort_values(
        "importance",
        ascending=False,
    ).round(3).to_string(index=False)
)

print("\nSaved:", OUTPUT_PATH)
