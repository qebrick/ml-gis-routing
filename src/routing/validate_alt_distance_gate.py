from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

BENCHMARK = Path(
    "data/processed/adaptive_alt_precomputed_benchmark.parquet"
)
FEATURES = Path(
    "data/processed/adaptive_alt_presearch_features.parquet"
)
OUTPUT = Path(
    "data/processed/adaptive_alt_distance_gate_validation.parquet"
)

REPEATS = 5
N_SPLITS = 5
DISTANCE_QUANTILE = 0.75

timings = pd.read_parquet(BENCHMARK)
features = pd.read_parquet(FEATURES)

features = features[
    ["pair_id", "destination", "air_distance_m"]
].drop_duplicates("pair_id")

# One row per repeat, OD and k.
timings = timings[
    timings["k"].isin([2, 4])
].copy()

assert not timings.duplicated(
    ["repeat", "pair_id", "k"]
).any()

wide = timings.pivot(
    index=["repeat", "pair_id"],
    columns="k",
    values=["total_ms", "expanded"],
)

wide.columns = [
    f"{metric}_k{k}"
    for metric, k in wide.columns
]

wide = wide.reset_index().merge(
    features,
    on="pair_id",
    validate="many_to_one",
)

assert len(wide) == REPEATS * len(features)
assert wide.notna().all().all()

# Split by destination. Every destination is
# tested exactly once.
unique_targets = np.array(
    sorted(features["destination"].astype(str).unique())
)

folds = GroupKFold(n_splits=N_SPLITS)

assignments = []

for fold, (train_idx, test_idx) in enumerate(
    folds.split(
        unique_targets,
        groups=unique_targets,
    ),
    start=1,
):
    train_targets = set(unique_targets[train_idx])
    test_targets = set(unique_targets[test_idx])

    train_rows = features[
        features["destination"].astype(str).isin(
            train_targets
        )
    ]

    test_rows = features[
        features["destination"].astype(str).isin(
            test_targets
        )
    ].copy()

    threshold = train_rows[
        "air_distance_m"
    ].quantile(DISTANCE_QUANTILE)

    test_rows["fold"] = fold
    test_rows["threshold_m"] = threshold

    test_rows["chosen_k"] = np.where(
        test_rows["air_distance_m"] >= threshold,
        4,
        2,
    )

    assignments.append(
        test_rows[
            [
                "pair_id",
                "fold",
                "threshold_m",
                "chosen_k",
            ]
        ]
    )

assignments = pd.concat(
    assignments,
    ignore_index=True,
)

assert assignments["pair_id"].is_unique

result = wide.merge(
    assignments,
    on="pair_id",
    validate="many_to_one",
)

result["gate_ms"] = np.where(
    result["chosen_k"] == 4,
    result["total_ms_k4"],
    result["total_ms_k2"],
)

result["gate_expanded"] = np.where(
    result["chosen_k"] == 4,
    result["expanded_k4"],
    result["expanded_k2"],
)

# The cost of computing air distance is not
# included here. It should be measured in a
# subsequent end-to-end implementation.
result["delta_gate_vs_k2_ms"] = (
    result["gate_ms"]
    - result["total_ms_k2"]
)

result["delta_gate_vs_k4_ms"] = (
    result["gate_ms"]
    - result["total_ms_k4"]
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result.to_parquet(
    OUTPUT,
    index=False,
)

print("=== COVERAGE ===")
print("Routes:", len(features))
print("Unique targets:", len(unique_targets))
print("Repeated observations:", len(result))
print(
    "Routes assigned k=4:",
    int((assignments["chosen_k"] == 4).sum()),
)

print("\n=== FOLD THRESHOLDS ===")

print(
    assignments.groupby("fold")
    .agg(
        routes=("pair_id", "size"),
        threshold_m=("threshold_m", "first"),
        k4_routes=("chosen_k", lambda s: int((s == 4).sum())),
    )
    .round(3)
    .to_string()
)

print("\n=== PAIRED RUNTIME BY REPEAT ===")

by_repeat = (
    result.groupby("repeat")
    .agg(
        k2_ms=("total_ms_k2", "sum"),
        k4_ms=("total_ms_k4", "sum"),
        gate_ms=("gate_ms", "sum"),
        k2_expanded=("expanded_k2", "sum"),
        k4_expanded=("expanded_k4", "sum"),
        gate_expanded=("gate_expanded", "sum"),
    )
)

print(by_repeat.round(3).to_string())

print("\n=== MEDIAN TOTAL RUNTIME ===")

for col in ["k2_ms", "k4_ms", "gate_ms"]:
    print(
        col,
        round(by_repeat[col].median(), 3),
    )

print("\n=== GATE VS BASELINES ===")

for baseline in ["k2", "k4"]:
    delta = (
        result["gate_ms"]
        - result[f"total_ms_{baseline}"]
    )

    print(
        f"Gate vs {baseline}: "
        f"mean delta per OD={delta.mean():+.5f} ms, "
        f"faster={(delta < 0).sum()}, "
        f"slower={(delta > 0).sum()}, "
        f"equal={(delta == 0).sum()}"
    )

print("\n=== EXPANSIONS ===")

for name in ["k2", "k4", "gate"]:
    print(
        name,
        int(by_repeat[f"{name}_expanded"].iloc[0]),
    )

# Cluster bootstrap by destination, averaging
# repetitions first.
per_od = (
    result.groupby(
        ["pair_id", "destination"],
        as_index=False,
    )
    .agg(
        delta_vs_k2=(
            "delta_gate_vs_k2_ms", "mean"
        ),
        delta_vs_k4=(
            "delta_gate_vs_k4_ms", "mean"
        ),
    )
)

groups = [
    group
    for _, group in per_od.groupby("destination")
]

rng = np.random.default_rng(42)

print("\n=== TARGET-CLUSTER BOOTSTRAP ===")

for column in ["delta_vs_k2", "delta_vs_k4"]:
    samples = np.empty(10000)

    for i in range(len(samples)):
        selected = rng.integers(
            0,
            len(groups),
            size=len(groups),
        )

        values = np.concatenate([
            groups[j][column].to_numpy()
            for j in selected
        ])

        samples[i] = values.mean()

    lo, hi = np.percentile(
        samples,
        [2.5, 97.5],
    )

    print(
        column,
        "mean:",
        round(per_od[column].mean(), 5),
        "95% CI:",
        f"[{lo:.5f}, {hi:.5f}]",
    )

print("\nSaved:", OUTPUT)
