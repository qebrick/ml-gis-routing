from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

INPUT = Path(
    "data/processed/ml_road_class_time_features.parquet"
)
OUTPUT = Path(
    "data/processed/conservative_refit_validation.parquet"
)

LAMBDA = 100.0

all_classes = [
    "motorway",
    "primary",
    "secondary",
    "tertiary",
    "residential",
]

refit_classes = [
    "motorway",
    "primary",
    "secondary",
    "tertiary",
]

df = pd.read_parquet(INPUT)

X = df[
    [f"osm_time_{c}_s" for c in all_classes]
].to_numpy(dtype=float)

y = df["observed_time_s"].to_numpy(dtype=float)
osm = df["osm_time_s"].to_numpy(dtype=float)
other = df["osm_time_other_s"].to_numpy(dtype=float)

residential_idx = all_classes.index("residential")


def fit_coefficients(X_train, target, lam):
    n, p = X_train.shape

    A = np.vstack([
        X_train / np.sqrt(n),
        np.sqrt(lam) * np.eye(p),
    ])

    b = np.concatenate([
        target / np.sqrt(n),
        np.sqrt(lam) * np.ones(p),
    ])

    result = lsq_linear(
        A,
        b,
        bounds=(0.5, 2.0),
    )

    if not result.success:
        raise RuntimeError(result.message)

    return result.x


splitter = GroupShuffleSplit(
    n_splits=30,
    test_size=0.2,
    random_state=42,
)

rows = []

for split_id, (train_idx, test_idx) in enumerate(
    splitter.split(X, groups=df["trip_id"])
):
    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # Full class model.
    full_coef = fit_coefficients(
        X_train,
        y_train - other[train_idx],
        LAMBDA,
    )

    full_pred = (
        X_test @ full_coef
        + other[test_idx]
    )

    # Previous post-hoc conservative model.
    posthoc_coef = full_coef.copy()
    posthoc_coef[residential_idx] = 1.0

    posthoc_pred = (
        X_test @ posthoc_coef
        + other[test_idx]
    )

    # Correct conservative model:
    # residential and other are fixed at 1
    # before fitting the remaining coefficients.
    fixed_train = (
        other[train_idx]
        + X_train[:, residential_idx]
    )

    fixed_test = (
        other[test_idx]
        + X_test[:, residential_idx]
    )

    refit_coef = fit_coefficients(
        X_train[:, :4],
        y_train - fixed_train,
        LAMBDA,
    )

    refit_pred = (
        X_test[:, :4] @ refit_coef
        + fixed_test
    )

    predictions = {
        "osm": osm[test_idx],
        "class": full_pred,
        "posthoc": posthoc_pred,
        "refit": refit_pred,
    }

    row = {
        "split_id": split_id,
        "test_routes": len(test_idx),
        "test_trips": df.iloc[
            test_idx
        ]["trip_id"].nunique(),
    }

    for name, pred in predictions.items():
        row[f"mae_{name}_s"] = mean_absolute_error(
            y_test, pred
        )

    row["refit_vs_posthoc_s"] = (
        row["mae_refit_s"]
        - row["mae_posthoc_s"]
    )

    row["refit_improvement_pct"] = (
        100
        * (
            row["mae_osm_s"]
            - row["mae_refit_s"]
        )
        / row["mae_osm_s"]
    )

    for name, coef in zip(refit_classes, refit_coef):
        row[f"coef_{name}"] = float(coef)

    rows.append(row)


result = pd.DataFrame(rows)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)
result.to_parquet(OUTPUT, index=False)

print("=== MODEL COMPARISON ===")

columns = [
    "mae_osm_s",
    "mae_class_s",
    "mae_posthoc_s",
    "mae_refit_s",
    "refit_vs_posthoc_s",
    "refit_improvement_pct",
]

print(
    result[columns]
    .agg(["mean", "std", "min", "median", "max"])
    .round(3)
    .to_string()
)

print("\n=== MODEL WINS ===")

print(
    "Refit better than OSM:",
    int(
        (result["mae_refit_s"] < result["mae_osm_s"]).sum()
    ),
    "/ 30",
)

print(
    "Refit better than post-hoc:",
    int(
        (result["mae_refit_s"] < result["mae_posthoc_s"]).sum()
    ),
    "/ 30",
)

print(
    "Refit better than Class:",
    int(
        (result["mae_refit_s"] < result["mae_class_s"]).sum()
    ),
    "/ 30",
)

print("\n=== REFIT COEFFICIENT STABILITY ===")

coef_cols = [
    f"coef_{name}"
    for name in refit_classes
]

print(
    result[coef_cols]
    .agg(["mean", "std", "min", "max"])
    .round(4)
    .to_string()
)

print("\nSaved:", OUTPUT)
