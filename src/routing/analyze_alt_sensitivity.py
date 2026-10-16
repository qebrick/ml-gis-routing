from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path(
    "data/processed/adaptive_alt_end_to_end_diagnostics.parquet"
)
OUTPUT = Path(
    "data/processed/adaptive_alt_sensitivity_analysis.csv"
)

df = pd.read_parquet(INPUT)

df["delta_ms"] = df["delta_total_ms_vs_k2"]

per_od = (
    df.groupby(["pair_id", "destination"], as_index=False)
    .agg(
        mean_delta_ms=("delta_ms", "mean"),
        k=("k", "first"),
    )
)

per_od["saving_ms"] = -per_od["mean_delta_ms"]

# Rank OD by observed benefit.
ranked = per_od.sort_values(
    "saving_ms", ascending=False
).reset_index(drop=True)

print("=== TOP SAVINGS ===")
print(
    ranked.head(10).round(5).to_string(index=False)
)

print("\n=== LEAVE-TOP-N-OUT ===")

records = []

for n in [0, 1, 3, 5, 10]:
    excluded = set(ranked.head(n)["pair_id"])

    remaining = df[
        ~df["pair_id"].isin(excluded)
    ]

    per_repeat = (
        remaining.groupby("repeat")["delta_ms"].sum()
    )

    mean_delta = (
        remaining["delta_ms"].mean()
    )

    records.append({
        "excluded_top_n": n,
        "remaining_od": remaining["pair_id"].nunique(),
        "mean_delta_ms_per_od": mean_delta,
        "median_total_delta_ms": per_repeat.median(),
        "repeats_gate_faster": int(
            (per_repeat < 0).sum()
        ),
    })

sensitivity = pd.DataFrame(records)

print(
    sensitivity.round(5).to_string(index=False)
)

print("\n=== DESTINATION CONTRIBUTIONS ===")

by_target = (
    per_od.groupby("destination", as_index=False)
    .agg(
        routes=("pair_id", "size"),
        total_saving_ms=("saving_ms", "sum"),
    )
    .sort_values(
        "total_saving_ms", ascending=False
    )
)

print("Top 10 beneficial targets:")
print(
    by_target.head(10).round(5).to_string(index=False)
)

print("\nTop 10 harmful targets:")
print(
    by_target.tail(10).round(5).to_string(index=False)
)

print("\n=== GATE ADVANTAGE WITHOUT TOP TARGETS ===")

for n in [0, 1, 3, 5, 10]:
    excluded_targets = set(
        by_target.head(n)["destination"]
    )

    remaining = per_od[
        ~per_od["destination"].isin(excluded_targets)
    ]

    print(
        f"Exclude top {n} targets: "
        f"OD={len(remaining)}, "
        f"mean delta={remaining['mean_delta_ms'].mean():+.5f} ms, "
        f"total delta={remaining['mean_delta_ms'].sum():+.4f} ms"
    )

OUTPUT.parent.mkdir(
    parents=True, exist_ok=True
)

sensitivity.to_csv(
    OUTPUT, index=False
)

print("\nSaved:", OUTPUT)
