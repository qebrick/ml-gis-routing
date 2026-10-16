from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

INPUT_PATH = Path(
    "data/processed/ml_road_class_time_features.parquet"
)
OUTPUT_PATH = Path(
    "data/processed/conservative_weight_validation.parquet"
)

LAMBDA = 100.0

classes = [
    "motorway",
    "primary",
    "secondary",
    "tertiary",
    "residential",
]

features = [
    f"osm_time_{name}_s"
    for name in classes
]

df = pd.read_parquet(INPUT_PATH)

X = df[features].to_numpy(dtype=float)
y = df["observed_time_s"].to_numpy(dtype=float)
osm = df["osm_time_s"].to_numpy(dtype=float)
other = df["osm_time_other_s"].to_numpy(dtype=float)

splitter = GroupShuffleSplit(
    n_splits=30,
    test_size=0.2,
    random_state=42,
)

records = []

for split_id, (train_idx, test_idx) in enumerate(
    splitter.split(X, y, groups=df["trip_id"])
):
    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx] - other[train_idx]
    y_test = y[test_idx]

    n_train = len(train_idx)

    A = np.vstack([
        X_train / np.sqrt(n_train),
        np.sqrt(LAMBDA) * np.eye(len(classes)),
    ])

    b = np.concatenate([
        y_train / np.sqrt(n_train),
        np.sqrt(LAMBDA) * np.ones(len(classes)),
    ])

    fit = lsq_linear(
        A,
        b,
        bounds=(0.5, 2.0),
    )

    if not fit.success:
        raise RuntimeError(fit.message)

    class_coef = fit.x.copy()

    conservative_coef = class_coef.copy()
    conservative_coef[
        classes.index("residential")
    ] = 1.0

    predictions = {
        "osm": osm[test_idx],
        "class": (
            X_test @ class_coef
            + other[test_idx]
        ),
        "conservative": (
            X_test @ conservative_coef
            + other[test_idx]
        ),
    }

    record = {
        "split_id": split_id,
        "train_routes": len(train_idx),
        "test_routes": len(test_idx),
        "test_trips": df.iloc[
            test_idx
        ]["trip_id"].nunique(),
    }

    for name, prediction in predictions.items():
        record[f"mae_{name}_s"] = (
            mean_absolute_error(
                y_test, prediction
            )
        )

        record[f"bias_{name}_s"] = float(
            np.mean(prediction - y_test)
        )

    record["class_improvement_pct"] = (
        100
        * (
            record["mae_osm_s"]
            - record["mae_class_s"]
        )
        / record["mae_osm_s"]
    )

    record["conservative_improvement_pct"] = (
        100
        * (
            record["mae_osm_s"]
            - record["mae_conservative_s"]
        )
        / record["mae_osm_s"]
    )

    record["conservative_vs_class_s"] = (
        record["mae_conservative_s"]
        - record["mae_class_s"]
    )

    records.append(record)

result = pd.DataFrame(records)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)
result.to_parquet(OUTPUT_PATH, index=False)

print("=== REPEATED MODEL VALIDATION ===")

columns = [
    "mae_osm_s",
    "mae_class_s",
    "mae_conservative_s",
    "class_improvement_pct",
    "conservative_improvement_pct",
    "conservative_vs_class_s",
]

print(
    result[columns]
    .agg(["mean", "std", "min", "median", "max"])
    .round(3)
    .to_string()
)

print("\n=== MODEL WINS ===")

print(
    "Class better than OSM:",
    int(
        (
            result["mae_class_s"]
            < result["mae_osm_s"]
        ).sum()
    ),
    "/ 30",
)

print(
    "Conservative better than OSM:",
    int(
        (
            result["mae_conservative_s"]
            < result["mae_osm_s"]
        ).sum()
    ),
    "/ 30",
)

print(
    "Conservative better than Class:",
    int(
        (
            result["mae_conservative_s"]
            < result["mae_class_s"]
        ).sum()
    ),
    "/ 30",
)

print("\n=== WORST CONSERVATIVE SPLITS ===")

print(
    result.nsmallest(
        5, "conservative_improvement_pct"
    )[
        [
            "split_id",
            "test_routes",
            "mae_osm_s",
            "mae_class_s",
            "mae_conservative_s",
            "conservative_improvement_pct",
        ]
    ]
    .round(3)
    .to_string(index=False)
)

print("\nSaved:", OUTPUT_PATH)
