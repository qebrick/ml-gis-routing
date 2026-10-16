from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path(
    "data/processed/adaptive_alt_end_to_end_benchmark.parquet"
)
OUTPUT = Path(
    "data/processed/adaptive_alt_end_to_end_diagnostics.parquet"
)

df = pd.read_parquet(INPUT)

metrics = [
    "total_ms",
    "selection_ms",
    "preparation_ms",
    "search_ms",
    "expanded",
]

wide = df.pivot(
    index=["repeat", "pair_id", "destination"],
    columns="strategy",
    values=metrics,
)

wide.columns = [
    f"{metric}__{strategy}"
    for metric, strategy in wide.columns
]
wide = wide.reset_index()

gate = df[
    df["strategy"] == "distance_gated_alt"
][["repeat", "pair_id", "k"]]

wide = wide.merge(
    gate,
    on=["repeat", "pair_id"],
    validate="one_to_one",
)

for metric in metrics:
    wide[f"delta_{metric}_vs_k2"] = (
        wide[f"{metric}__distance_gated_alt"]
        - wide[f"{metric}__adaptive_alt_2"]
    )

print("=== BY CHOSEN K ===")

summary = (
    wide.groupby("k")
    .agg(
        observations=("pair_id", "size"),
        mean_delta_total_ms=(
            "delta_total_ms_vs_k2", "mean"
        ),
        sum_delta_total_ms=(
            "delta_total_ms_vs_k2", "sum"
        ),
        mean_delta_selection_ms=(
            "delta_selection_ms_vs_k2", "mean"
        ),
        mean_delta_preparation_ms=(
            "delta_preparation_ms_vs_k2", "mean"
        ),
        mean_delta_search_ms=(
            "delta_search_ms_vs_k2", "mean"
        ),
        mean_delta_expanded=(
            "delta_expanded_vs_k2", "mean"
        ),
    )
)

print(summary.round(6).to_string())

print("\n=== PER-OD BENEFIT CONCENTRATION ===")

per_od = (
    wide.groupby(["pair_id", "destination", "k"])
    .agg(
        mean_delta_ms=("delta_total_ms_vs_k2", "mean"),
        mean_delta_expanded=(
            "delta_expanded_vs_k2", "mean"
        ),
    )
    .reset_index()
)

per_od["saving_ms"] = -per_od["mean_delta_ms"]

print("Top 10 savings:")
print(
    per_od.sort_values(
        "saving_ms", ascending=False
    ).head(10).round(6).to_string(index=False)
)

print("\nTop 10 regressions:")
print(
    per_od.sort_values(
        "saving_ms", ascending=True
    ).head(10).round(6).to_string(index=False)
)

print("\n=== PER-REPEAT DELTAS ===")

repeat_summary = (
    wide.groupby("repeat")
    .agg(
        delta_total_ms=("delta_total_ms_vs_k2", "sum"),
        delta_selection_ms=(
            "delta_selection_ms_vs_k2", "sum"
        ),
        delta_preparation_ms=(
            "delta_preparation_ms_vs_k2", "sum"
        ),
        delta_search_ms=(
            "delta_search_ms_vs_k2", "sum"
        ),
    )
)

print(repeat_summary.round(4).to_string())

print("\n=== DISTRIBUTION ===")

for k in [2, 4]:
    values = wide.loc[
        wide["k"] == k, "delta_total_ms_vs_k2"
    ]

    print(
        f"k={k}: "
        f"mean={values.mean():+.6f}, "
        f"median={values.median():+.6f}, "
        f"p10={values.quantile(.10):+.6f}, "
        f"p90={values.quantile(.90):+.6f} ms"
    )

OUTPUT.parent.mkdir(
    parents=True, exist_ok=True
)
wide.to_parquet(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
