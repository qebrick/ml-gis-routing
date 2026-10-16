from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import lsq_linear
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error

INPUT_PATH = Path(
    "data/processed/ml_road_class_time_features.parquet"
)
OUTPUT_PATH = Path(
    "data/processed/class_weight_validation.parquet"
)

df = pd.read_parquet(INPUT_PATH)

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

# 'other' remains unmodified.
fixed_column = "osm_time_other_s"

# Preselected regularization strengths.
lambdas = [0.0, 10.0, 100.0, 1000.0]

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

    X_train = train[features].to_numpy(dtype=float)
    X_test = test[features].to_numpy(dtype=float)

    # The other-class contribution is fixed at 1.0.
    y_train = (
        train["observed_time_s"].to_numpy()
        - train[fixed_column].to_numpy()
    )

    y_test = test["observed_time_s"].to_numpy()

    osm_pred = test["osm_time_s"].to_numpy()

    osm_mae = mean_absolute_error(
        y_test, osm_pred
    )

    for lam in lambdas:
        # Normalize data loss by number of train routes.
        n_train = len(train)

        A = X_train / np.sqrt(n_train)
        b = y_train / np.sqrt(n_train)

        # Penalize deviations from c = 1.
        A_reg = np.sqrt(lam) * np.eye(len(classes))
        b_reg = np.sqrt(lam) * np.ones(len(classes))

        A_full = np.vstack([A, A_reg])
        b_full = np.concatenate([b, b_reg])

        fit = lsq_linear(
            A_full,
            b_full,
            bounds=(0.5, 2.0),
        )

        if not fit.success:
            raise RuntimeError(fit.message)

        coefficients = fit.x

        pred = (
            X_test @ coefficients
            + test[fixed_column].to_numpy()
        )

        mae = mean_absolute_error(y_test, pred)
        bias = np.mean(pred - y_test)

        record = {
            "split_id": split_id,
            "lambda": lam,
            "train_routes": len(train),
            "test_routes": len(test),
            "test_trips": test["trip_id"].nunique(),
            "osm_mae_s": osm_mae,
            "class_mae_s": mae,
            "class_bias_s": bias,
            "improvement_pct": (
                100 * (osm_mae - mae) / osm_mae
            ),
        }

        for name, coefficient in zip(
            classes, coefficients
        ):
            record[f"coef_{name}"] = coefficient

        records.append(record)

result = pd.DataFrame(records)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)
result.to_parquet(OUTPUT_PATH, index=False)

print("=== REPEATED CLASS VALIDATION ===")

summary = (
    result.groupby("lambda")
    .agg(
        mean_osm_mae_s=("osm_mae_s", "mean"),
        mean_class_mae_s=("class_mae_s", "mean"),
        median_improvement_pct=(
            "improvement_pct", "median"
        ),
        mean_improvement_pct=(
            "improvement_pct", "mean"
        ),
        worst_improvement_pct=(
            "improvement_pct", "min"
        ),
        wins=("improvement_pct", lambda x: (x > 0).sum()),
    )
)

print(summary.round(3).to_string())

print("\n=== COEFFICIENT STABILITY ===")

coef_columns = [
    f"coef_{name}"
    for name in classes
]

for lam in lambdas:
    subset = result.loc[
        result["lambda"] == lam
    ]

    print(f"\nLambda = {lam:g}")

    print(
        subset[coef_columns]
        .agg(["mean", "std", "min", "max"])
        .round(3)
        .to_string()
    )

print("\nSaved:", OUTPUT_PATH)
