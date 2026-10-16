from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

INPUT = Path(
    "data/processed/ml_road_class_time_features.parquet"
)
OUTPUT = Path(
    "data/processed/nested_eta_validation.parquet"
)

CLASSES = [
    "motorway",
    "primary",
    "secondary",
    "tertiary",
    "residential",
]

LAMBDAS = [0.0, 10.0, 100.0, 1000.0]

df = pd.read_parquet(INPUT)

X = df[
    [f"osm_time_{c}_s" for c in CLASSES]
].to_numpy(dtype=float)

other = df["osm_time_other_s"].to_numpy(dtype=float)
osm = df["osm_time_s"].to_numpy(dtype=float)
y = df["observed_time_s"].to_numpy(dtype=float)
groups = df["trip_id"].astype(str).to_numpy()


def fit_model(X_train, y_train, fixed_train, lam):
    n, p = X_train.shape

    if lam > 0:
        A = np.vstack([
            X_train / np.sqrt(n),
            np.sqrt(lam) * np.eye(p),
        ])
        b = np.concatenate([
            (y_train - fixed_train) / np.sqrt(n),
            np.sqrt(lam) * np.ones(p),
        ])
    else:
        A = X_train / np.sqrt(n)
        b = (y_train - fixed_train) / np.sqrt(n)

    result = lsq_linear(
        A,
        b,
        bounds=(0.5, 2.0),
    )

    if not result.success:
        raise RuntimeError(result.message)

    return result.x


def predict(model, train, test, lam):
    if model == "class":
        train_X = X[train]
        test_X = X[test]
        fixed_train = other[train]
        fixed_test = other[test]

    elif model == "conservative":
        train_X = X[train, :4]
        test_X = X[test, :4]
        fixed_train = other[train] + X[train, 4]
        fixed_test = other[test] + X[test, 4]

    else:
        raise ValueError(model)

    coef = fit_model(
        train_X,
        y[train],
        fixed_train,
        lam,
    )

    return test_X @ coef + fixed_test, coef


outer_cv = GroupKFold(n_splits=5)
records = []

for outer_fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(X, y, groups)
):
    inner_cv = GroupKFold(n_splits=4)

    row = {
        "outer_fold": outer_fold,
        "train_routes": len(train_idx),
        "test_routes": len(test_idx),
        "train_trips": len(set(groups[train_idx])),
        "test_trips": len(set(groups[test_idx])),
        "mae_osm_s": mean_absolute_error(
            y[test_idx], osm[test_idx]
        ),
    }

    for model in ["class", "conservative"]:
        scores = {}

        for lam in LAMBDAS:
            errors = []

            for inner_train, inner_val in inner_cv.split(
                X[train_idx],
                y[train_idx],
                groups[train_idx],
            ):
                actual_train = train_idx[inner_train]
                actual_val = train_idx[inner_val]

                pred, _ = predict(
                    model,
                    actual_train,
                    actual_val,
                    lam,
                )

                errors.extend(
                    np.abs(y[actual_val] - pred).tolist()
                )

            scores[lam] = float(np.mean(errors))

        best_lambda = min(
            LAMBDAS,
            key=lambda lam: (scores[lam], lam),
        )

        pred, coef = predict(
            model,
            train_idx,
            test_idx,
            best_lambda,
        )

        row[f"lambda_{model}"] = best_lambda
        row[f"mae_{model}_s"] = mean_absolute_error(
            y[test_idx], pred
        )

        for name, value in zip(
            CLASSES if model == "class" else CLASSES[:4],
            coef,
        ):
            row[f"coef_{model}_{name}"] = float(value)

    records.append(row)

result = pd.DataFrame(records)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_parquet(OUTPUT, index=False)

print("=== OUTER FOLD RESULTS ===")
print(
    result[
        [
            "outer_fold",
            "test_routes",
            "test_trips",
            "lambda_class",
            "lambda_conservative",
            "mae_osm_s",
            "mae_class_s",
            "mae_conservative_s",
        ]
    ].round(3).to_string(index=False)
)

print("\n=== POOLED OUTER MAE ===")

for model in ["osm", "class", "conservative"]:
    pooled_mae = np.average(
        result[f"mae_{model}_s"],
        weights=result["test_routes"],
    )
    print(f"{model:15s}: {pooled_mae:.3f} s")

print("\n=== LAMBDA SELECTION ===")

for model in ["class", "conservative"]:
    print(f"\n{model.upper()}")
    print(
        result[f"lambda_{model}"]
        .value_counts()
        .sort_index()
        .to_string()
    )

print("\nSaved:", OUTPUT)
