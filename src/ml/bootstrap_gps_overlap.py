from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path(
    "data/processed/gps_traversed_overlap_comparison.parquet"
)

OUTPUT_SUMMARY = Path(
    "data/processed/gps_overlap_bootstrap_summary.csv"
)

OUTPUT_DRAWS = Path(
    "data/processed/gps_overlap_bootstrap_draws.parquet"
)

N_BOOTSTRAP = 10000
SEED = 42
ALPHA = 0.05

df = pd.read_parquet(INPUT_PATH)

required = [
    "trip_id",
    "delta_class",
    "delta_conservative",
    "class_changed",
    "conservative_changed",
]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

df["trip_id"] = df["trip_id"].astype(str)

for model in ["class", "conservative"]:
    delta_col = f"delta_{model}"
    changed_col = f"{model}_changed"

    if df[delta_col].isna().any():
        raise ValueError(
            f"NaN values in {delta_col}"
        )

    df[changed_col] = df[changed_col].astype(bool)

# Preserve all OD fragments belonging to each trip.
trip_groups = [
    group
    for _, group in df.groupby("trip_id", sort=True)
]

n_trips = len(trip_groups)

if n_trips < 2:
    raise ValueError(
        "At least two trips are required"
    )

# Precompute sufficient statistics per trip.
# This is equivalent to resampling entire trip clusters,
# but much faster than concatenating DataFrames.
trip_stats = {}

for model in ["class", "conservative"]:
    delta_col = f"delta_{model}"
    changed_col = f"{model}_changed"

    trip_stats[model] = {
        "sum_all": np.array([
            group[delta_col].sum()
            for group in trip_groups
        ], dtype=float),

        "count_all": np.array([
            len(group)
            for group in trip_groups
        ], dtype=int),

        "sum_changed": np.array([
            group.loc[
                group[changed_col],
                delta_col
            ].sum()
            for group in trip_groups
        ], dtype=float),

        "count_changed": np.array([
            group[changed_col].sum()
            for group in trip_groups
        ], dtype=int),
    }

rng = np.random.default_rng(SEED)

# One shared cluster resample for both models:
# preserves the paired comparison.
sample_indices = rng.integers(
    low=0,
    high=n_trips,
    size=(N_BOOTSTRAP, n_trips),
)

draw_records = []
summary_records = []


def bootstrap_estimates(stats, changed_only):
    if changed_only:
        sums = stats["sum_changed"]
        counts = stats["count_changed"]
    else:
        sums = stats["sum_all"]
        counts = stats["count_all"]

    bootstrap_sum = sums[sample_indices].sum(axis=1)
    bootstrap_count = counts[sample_indices].sum(axis=1)

    estimates = np.full(
        N_BOOTSTRAP,
        np.nan,
        dtype=float,
    )

    valid = bootstrap_count > 0

    estimates[valid] = (
        bootstrap_sum[valid]
        / bootstrap_count[valid]
    )

    observed = sums.sum() / counts.sum()

    return observed, estimates, valid


for model in ["class", "conservative"]:
    for scope, changed_only in [
        ("all", False),
        ("changed", True),
    ]:
        observed, estimates, valid = (
            bootstrap_estimates(
                trip_stats[model],
                changed_only,
            )
        )

        valid_estimates = estimates[valid]

        if len(valid_estimates) == 0:
            raise RuntimeError(
                f"No valid bootstrap draws: {model}/{scope}"
            )

        ci_low, ci_high = np.quantile(
            valid_estimates,
            [ALPHA / 2, 1 - ALPHA / 2],
        )

        # Diagnostic sign proportions, not a formal
        # null-hypothesis bootstrap p-value.
        proportion_negative = np.mean(
            valid_estimates < 0
        )

        proportion_positive = np.mean(
            valid_estimates > 0
        )

        summary_records.append({
            "model": model,
            "scope": scope,
            "observed_delta": observed,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "bootstrap_std": np.std(
                valid_estimates,
                ddof=1,
            ),
            "proportion_negative": (
                proportion_negative
            ),
            "proportion_positive": (
                proportion_positive
            ),
            "valid_draws": len(valid_estimates),
            "total_draws": N_BOOTSTRAP,
        })

        for draw_id, value in enumerate(estimates):
            draw_records.append({
                "draw_id": draw_id,
                "model": model,
                "scope": scope,
                "delta": value,
            })


summary = pd.DataFrame(summary_records)
draws = pd.DataFrame(draw_records)

OUTPUT_SUMMARY.parent.mkdir(
    parents=True,
    exist_ok=True,
)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False,
)

draws.to_parquet(
    OUTPUT_DRAWS,
    index=False,
)

print("=== DATA COVERAGE ===")
print("OD pairs:", len(df))
print("Unique trips:", n_trips)
print("Bootstrap draws:", N_BOOTSTRAP)

print("\n=== CLUSTER BOOTSTRAP RESULTS ===")

display = summary[
    [
        "model",
        "scope",
        "observed_delta",
        "ci_low",
        "ci_high",
        "bootstrap_std",
        "proportion_negative",
        "valid_draws",
    ]
].copy()

for col in [
    "observed_delta",
    "ci_low",
    "ci_high",
    "bootstrap_std",
]:
    display[col] *= 100

print(
    display.round(3).to_string(index=False)
)

print(
    "\nDelta and CI are reported "
    "in percentage points."
)

print("\n=== INTERVAL INTERPRETATION ===")

for row in summary.itertuples(index=False):
    if row.ci_high < 0:
        interpretation = "entirely negative"
    elif row.ci_low > 0:
        interpretation = "entirely positive"
    else:
        interpretation = "includes zero"

    print(
        f"{row.model:13s} "
        f"{row.scope:8s} "
        f"CI: {interpretation}"
    )

print("\n=== CHANGED ROUTE COVERAGE ===")

for model in ["class", "conservative"]:
    changed = df[df[f"{model}_changed"]]

    print(
        f"{model:13s} "
        f"OD pairs={len(changed):3d}, "
        f"trips={changed['trip_id'].nunique():3d}"
    )

print("\nSaved summary:", OUTPUT_SUMMARY)
print("Saved draws:", OUTPUT_DRAWS)
