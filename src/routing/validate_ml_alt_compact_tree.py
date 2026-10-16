from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.tree import (
    DecisionTreeRegressor,
    export_text,
)

INPUT = Path(
    "data/processed/ml_alt_budget_groupkfold.parquet"
)
OUTPUT = Path(
    "data/processed/ml_alt_compact_tree_validation.parquet"
)

df = pd.read_parquet(INPUT).sort_values(
    "pair_id"
).reset_index(drop=True)

feature_cols = [
    "air_distance_m",
    "alt_best",
    "alt_second",
    "alt_fourth",
    "alt_eighth",
    "alt_mean",
    "alt_std",
    "alt_gap_1_2",
    "alt_gap_2_4",
    "alt_gap_4_8",
    "alt_positive_count",
    "abs_delta_lat",
    "abs_delta_lon",
]

X = df[feature_cols].to_numpy(dtype=np.float64)
y = df["saved_expansions"].to_numpy(dtype=np.float64)
groups = df["destination"].astype(str).to_numpy()

time2 = df["alt2_time_ms"].to_numpy()
time4 = df["alt4_time_ms"].to_numpy()
time_gate = df["distance_gate_time_ms"].to_numpy()

saved_time = time2 - time4

outer = GroupKFold(n_splits=5)

depths = [2, 3, 4]
thresholds = [
    0, 10, 25, 50, 100, 150, 200, 300, 500
]

records = []

for depth in depths:
    print(f"\n=== DEPTH {depth} ===")

    for fold, (train, test) in enumerate(
        outer.split(X, y, groups),
        start=1,
    ):
        # Inner out-of-fold predictions on outer-train
        # only. Never use outer-test labels.
        inner_groups = groups[train]
        inner_cv = GroupKFold(n_splits=4)

        oof = np.full(len(train), np.nan)

        for inner_train, inner_valid in inner_cv.split(
            X[train],
            y[train],
            inner_groups,
        ):
            model = DecisionTreeRegressor(
                max_depth=depth,
                min_samples_leaf=5,
                random_state=42,
            )

            model.fit(
                X[train[inner_train]],
                y[train[inner_train]],
            )

            oof[inner_valid] = model.predict(
                X[train[inner_valid]]
            )

        assert np.isfinite(oof).all()

        # Threshold selected exclusively using
        # inner out-of-fold predictions.
        best_threshold = None
        best_gain = -np.inf

        for threshold in thresholds:
            use4 = oof >= threshold

            gain = np.where(
                use4,
                saved_time[train],
                0.0,
            ).sum()

            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold

        # Fit final tree on outer-train.
        model = DecisionTreeRegressor(
            max_depth=depth,
            min_samples_leaf=5,
            random_state=42,
        )

        model.fit(X[train], y[train])
        predicted = model.predict(X[test])

        chosen = np.where(
            predicted >= best_threshold,
            4, 2,
        )

        print(
            f"Fold {fold}: "
            f"threshold={best_threshold}, "
            f"k4={int((chosen == 4).sum())}/{len(test)}, "
            f"tree_nodes={model.tree_.node_count}"
        )

        for local_i, pair_i in enumerate(test):
            k = int(chosen[local_i])

            records.append({
                "depth": depth,
                "fold": fold,
                "pair_id": int(df.iloc[pair_i]["pair_id"]),
                "destination": groups[pair_i],
                "predicted_saved_expansions": float(
                    predicted[local_i]
                ),
                "threshold": float(best_threshold),
                "chosen_k": k,
                "ml_time_ms": float(
                    time4[pair_i] if k == 4
                    else time2[pair_i]
                ),
                "alt2_time_ms": float(time2[pair_i]),
                "alt4_time_ms": float(time4[pair_i]),
                "distance_gate_time_ms": float(
                    time_gate[pair_i]
                ),
                "saved_expansions": float(y[pair_i]),
            })

    # For interpretation only: tree trained on all
    # OD. This is not the deployed fold model.
    full_model = DecisionTreeRegressor(
        max_depth=depth,
        min_samples_leaf=5,
        random_state=42,
    )
    full_model.fit(X, y)

    print(f"\nFull-data tree, depth={depth}:")
    print(
        export_text(
            full_model,
            feature_names=feature_cols,
            max_depth=depth,
            decimals=2,
        )
    )

result = pd.DataFrame(records)

print("\n=== NESTED GROUPKFOLD SUMMARY ===")

summary = (
    result.groupby("depth")
    .agg(
        routes=("pair_id", "size"),
        k4_routes=("chosen_k", lambda x: int((x == 4).sum())),
        ml_total_ms=("ml_time_ms", "sum"),
        alt2_total_ms=("alt2_time_ms", "sum"),
        alt4_total_ms=("alt4_time_ms", "sum"),
        gate_total_ms=("distance_gate_time_ms", "sum"),
    )
)

summary["gain_vs_alt2_ms"] = (
    summary["alt2_total_ms"]
    - summary["ml_total_ms"]
)

summary["gain_vs_gate_ms"] = (
    summary["gate_total_ms"]
    - summary["ml_total_ms"]
)

print(summary.round(4).to_string())

print("\n=== PER-FOLD GAIN VS ALT-2 ===")

result["gain_vs_alt2_ms"] = (
    result["alt2_time_ms"]
    - result["ml_time_ms"]
)

print(
    result.groupby(["depth", "fold"])
    ["gain_vs_alt2_ms"]
    .sum()
    .round(4)
    .to_string()
)

OUTPUT.parent.mkdir(
    parents=True, exist_ok=True
)

result.to_parquet(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
