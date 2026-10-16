from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error

INPUT_PATH = Path(
    "data/processed/ml_distance_based_routes.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/repeated_route_baseline_validation.parquet"
)

df = pd.read_parquet(INPUT_PATH)

df = df.loc[
    df["target_distance_m"] == 500
].copy()

splitter = GroupShuffleSplit(
    n_splits=30,
    test_size=0.2,
    random_state=42,
)

records = []

for split_id, (train_idx, test_idx) in enumerate(
    splitter.split(df, groups=df["trip_id"])
):
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]

    y_train = train["observed_time_s"]
    y_test = test["observed_time_s"]

    osm_train = train["osm_time_s"]
    osm_test = test["osm_time_s"]

    # Original OSM.
    pred_osm = osm_test.to_numpy()

    # Train-only multiplicative calibration.
    factor = y_train.sum() / osm_train.sum()
    pred_scaled = factor * pred_osm

    # Train-only linear calibration.
    model = LinearRegression()
    model.fit(
        train[["osm_time_s"]],
        y_train,
    )

    pred_linear = np.maximum(
        model.predict(test[["osm_time_s"]]),
        0.0,
    )

    mae_osm = mean_absolute_error(
        y_test, pred_osm
    )
    mae_scaled = mean_absolute_error(
        y_test, pred_scaled
    )
    mae_linear = mean_absolute_error(
        y_test, pred_linear
    )

    records.append({
        "split_id": split_id,
        "train_routes": len(train),
        "test_routes": len(test),
        "test_trips": test["trip_id"].nunique(),
        "mae_osm_s": mae_osm,
        "mae_scaled_s": mae_scaled,
        "mae_linear_s": mae_linear,
        "linear_improvement_pct": (
            100 * (mae_osm - mae_linear) / mae_osm
        ),
        "linear_intercept": model.intercept_,
        "linear_slope": model.coef_[0],
        "scaled_factor": factor,
    })

result = pd.DataFrame(records)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)
result.to_parquet(OUTPUT_PATH, index=False)

print("=== REPEATED VALIDATION ===")

columns = [
    "mae_osm_s",
    "mae_scaled_s",
    "mae_linear_s",
    "linear_improvement_pct",
    "linear_intercept",
    "linear_slope",
    "scaled_factor",
]

print(
    result[columns]
    .describe(percentiles=[0.1, 0.5, 0.9])
    .round(3)
    .to_string()
)

print("\n=== MODEL WINS ===")

print(
    "Linear better than OSM:",
    int(
        (result["mae_linear_s"] < result["mae_osm_s"])
        .sum()
    ),
    "/",
    len(result),
)

print(
    "Scaled better than OSM:",
    int(
        (result["mae_scaled_s"] < result["mae_osm_s"])
        .sum()
    ),
    "/",
    len(result),
)

print("\n=== WORST LINEAR SPLITS ===")

print(
    result.nsmallest(
        5, "linear_improvement_pct"
    )[
        [
            "split_id",
            "test_routes",
            "mae_osm_s",
            "mae_linear_s",
            "linear_improvement_pct",
        ]
    ]
    .round(2)
    .to_string(index=False)
)

print("\nSaved:", OUTPUT_PATH)
