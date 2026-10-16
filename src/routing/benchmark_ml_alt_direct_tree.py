from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeRegressor

DATA = Path(
    "data/processed/ml_alt_budget_groupkfold.parquet"
)
VALIDATION = Path(
    "data/processed/ml_alt_compact_tree_validation.parquet"
)
OUTPUT = Path(
    "data/processed/ml_alt_tree_inference_benchmark.parquet"
)

SEED = 42
REPEATS = 100

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

df = (
    pd.read_parquet(DATA)
    .sort_values("pair_id")
    .reset_index(drop=True)
)

saved = pd.read_parquet(VALIDATION)
saved = saved[saved["depth"] == 2].copy()

X = df[feature_cols].to_numpy(dtype=np.float64)
y = df["saved_expansions"].to_numpy(dtype=np.float64)
groups = df["destination"].astype(str).to_numpy()

assert len(df) == 142
assert saved["pair_id"].is_unique

saved = (
    saved.set_index("pair_id")
    .loc[df["pair_id"]]
    .reset_index()
)

def make_tree():
    return DecisionTreeRegressor(
        max_depth=2,
        min_samples_leaf=5,
        random_state=SEED,
    )

def predict_direct(tree, x):
    """
    Execute the trained sklearn tree using
    raw tree arrays, without sklearn.predict().
    """
    t = tree.tree_
    node = 0

    while t.children_left[node] != -1:
        feature = t.feature[node]
        threshold = t.threshold[node]

        if x[feature] <= threshold:
            node = t.children_left[node]
        else:
            node = t.children_right[node]

    return float(t.value[node, 0, 0])

models = {}
threshold_by_fold = {}

outer = GroupKFold(n_splits=5)

for fold, (train, test) in enumerate(
    outer.split(X, y, groups),
    start=1,
):
    inner = GroupKFold(n_splits=4)
    oof = np.full(len(train), np.nan)

    for inner_train, inner_valid in inner.split(
        X[train],
        y[train],
        groups[train],
    ):
        m = make_tree()
        m.fit(
            X[train[inner_train]],
            y[train[inner_train]],
        )
        oof[inner_valid] = m.predict(
            X[train[inner_valid]]
        )

    assert np.isfinite(oof).all()

    saved_time = (
        df["alt2_time_ms"].to_numpy()
        - df["alt4_time_ms"].to_numpy()
    )

    candidates = [
        0, 10, 25, 50, 100, 150, 200, 300, 500
    ]

    best_threshold = None
    best_gain = -np.inf

    for threshold in candidates:
        gain = np.where(
            oof >= threshold,
            saved_time[train],
            0.0,
        ).sum()

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    model = make_tree()
    model.fit(X[train], y[train])

    expected_k = np.where(
        model.predict(X[test]) >= best_threshold,
        4, 2,
    )

    actual_k = saved.iloc[test]["chosen_k"].to_numpy()

    if not np.array_equal(expected_k, actual_k):
        raise AssertionError(
            f"Fold {fold}: saved decisions mismatch"
        )

    models[fold] = model
    threshold_by_fold[fold] = best_threshold

    print(
        f"Fold {fold}: "
        f"threshold={best_threshold}, "
        f"nodes={model.tree_.node_count}, "
        f"decisions verified={len(test)}"
    )

# -------------------------------------------------
# Warm-up and correctness of direct traversal
# -------------------------------------------------

for i in range(len(df)):
    fold = int(saved.iloc[i]["fold"])
    model = models[fold]

    direct = predict_direct(model, X[i])
    sklearn_value = float(
        model.predict(X[i:i + 1])[0]
    )

    if not np.isclose(
        direct, sklearn_value,
        atol=1e-9,
        rtol=1e-9,
    ):
        raise AssertionError(
            f"Prediction mismatch at pair {i}"
        )

# -------------------------------------------------
# Compare decision overheads
# -------------------------------------------------

rng = np.random.default_rng(SEED)
records = []

# Threshold from the existing distance-gate
# experiment is not needed to benchmark its
# decision overhead. A fixed scalar suffices.
distance_threshold = 2900.0

for repeat in range(REPEATS):
    order = rng.permutation(len(df))

    for i in order:
        fold = int(saved.iloc[i]["fold"])
        model = models[fold]
        threshold = threshold_by_fold[fold]
        x = X[i]

        # Rotate method order to limit systematic
        # measurement-order effects.
        methods = rng.permutation([
            "distance_rule",
            "direct_tree",
            "sklearn_tree",
        ])

        for method in methods:
            start = perf_counter_ns()

            if method == "distance_rule":
                k = (
                    4 if x[0] >= distance_threshold
                    else 2
                )

            elif method == "direct_tree":
                score = predict_direct(model, x)
                k = 4 if score >= threshold else 2

            else:
                score = float(
                    model.predict(x.reshape(1, -1))[0]
                )
                k = 4 if score >= threshold else 2

            elapsed_us = (
                perf_counter_ns() - start
            ) / 1000.0

            records.append({
                "repeat": repeat,
                "pair_id": int(df.iloc[i]["pair_id"]),
                "method": method,
                "elapsed_us": elapsed_us,
                "chosen_k": int(k),
            })

result = pd.DataFrame(records)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)
result.to_parquet(OUTPUT, index=False)

print("\n=== DECISION LATENCY ===")

summary = (
    result.groupby("method")["elapsed_us"]
    .agg(
        median_us="median",
        mean_us="mean",
        p95_us=lambda s: s.quantile(0.95),
    )
)

print(summary.round(4).to_string())

print("\n=== INCREMENTAL TREE COST ===")

distance_median = summary.loc[
    "distance_rule", "median_us"
]
direct_median = summary.loc[
    "direct_tree", "median_us"
]

incremental = direct_median - distance_median

print(
    "Direct tree minus distance rule (us):",
    round(incremental, 4)
)

print(
    "Estimated incremental cost over 142 OD (ms):",
    round(incremental * 142 / 1000, 4)
)

print(
    "Offline ML advantage over distance gate (ms):",
    round(
        (
            df["distance_gate_time_ms"]
            - saved["ml_time_ms"]
        ).sum(),
        4,
    )
)

print("\n=== DECISION CONSISTENCY ===")

direct = result[
    result["method"] == "direct_tree"
].sort_values(["repeat", "pair_id"])

sklearn = result[
    result["method"] == "sklearn_tree"
].sort_values(["repeat", "pair_id"])

assert np.array_equal(
    direct["chosen_k"].to_numpy(),
    sklearn["chosen_k"].to_numpy(),
)

print("Direct tree matches sklearn: True")
print("Saved:", OUTPUT)
