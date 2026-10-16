from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path(
    "data/processed/porto_heldout_alt_benchmark.parquet"
)
OUTPUT = Path(
    "data/processed/porto_heldout_alt_diagnostics.parquet"
)

df = pd.read_parquet(INPUT)

strategies = [
    "alt2", "alt4", "distance_gate", "ml_tree"
]

metrics = [
    "total_ms",
    "decision_ms",
    "selection_ms",
    "preparation_ms",
    "search_ms",
    "expanded",
    "k",
]

assert set(df["strategy"]) == set(strategies)
assert df["repeat"].nunique() == 3
assert df["pair_id"].nunique() == 600

wide = df.pivot(
    index=[
        "repeat",
        "pair_id",
        "destination",
        "distance_bin",
    ],
    columns="strategy",
    values=metrics,
)

wide.columns = [
    f"{metric}__{strategy}"
    for metric, strategy in wide.columns
]
wide = wide.reset_index()

assert len(wide) == 1800

print("=== TIME COMPONENTS ===")

components = (
    df.groupby("strategy")
    [[
        "decision_ms",
        "selection_ms",
        "preparation_ms",
        "search_ms",
        "total_ms",
    ]]
    .mean()
    * 1000
)

print("Mean microseconds per OD:")
print(components.round(3).to_string())

print("\n=== DECISION AGREEMENT ===")

first = wide[wide["repeat"] == 0].copy()

first["decision_case"] = np.select(
    [
        (
            (first["k__distance_gate"] == 2)
            & (first["k__ml_tree"] == 2)
        ),
        (
            (first["k__distance_gate"] == 2)
            & (first["k__ml_tree"] == 4)
        ),
        (
            (first["k__distance_gate"] == 4)
            & (first["k__ml_tree"] == 2)
        ),
        (
            (first["k__distance_gate"] == 4)
            & (first["k__ml_tree"] == 4)
        ),
    ],
    ["both_2", "gate_2_ml_4", "gate_4_ml_2", "both_4"],
    default="ERROR",
)

assert not (first["decision_case"] == "ERROR").any()

cases = first[[
    "pair_id", "decision_case"
]]

wide = wide.merge(
    cases, on="pair_id", validate="many_to_one"
)

wide["ml_minus_gate_ms"] = (
    wide["total_ms__ml_tree"]
    - wide["total_ms__distance_gate"]
)

wide["gate_minus_alt2_ms"] = (
    wide["total_ms__distance_gate"]
    - wide["total_ms__alt2"]
)

wide["ml_minus_alt2_ms"] = (
    wide["total_ms__ml_tree"]
    - wide["total_ms__alt2"]
)

print(
    first.groupby(
        ["distance_bin", "decision_case"]
    )
    .size()
    .unstack(fill_value=0)
    .to_string()
)

print("\n=== ML VS GATE BY DECISION CASE ===")

case_summary = (
    wide.groupby("decision_case")
    .agg(
        observations=("pair_id", "size"),
        mean_delta_ms=("ml_minus_gate_ms", "mean"),
        total_delta_ms=("ml_minus_gate_ms", "sum"),
    )
)

case_summary["mean_delta_us"] = (
    case_summary["mean_delta_ms"] * 1000
)

print(case_summary.round(4).to_string())

print("\n=== ML VS GATE BY DISTANCE BIN ===")

bin_summary = (
    wide.groupby("distance_bin")
    .agg(
        observations=("pair_id", "size"),
        mean_delta_ms=("ml_minus_gate_ms", "mean"),
        total_delta_ms=("ml_minus_gate_ms", "sum"),
    )
)

print(bin_summary.round(4).to_string())

print("\n=== SEARCH VS PREPARATION ===")

for strategy in ["distance_gate", "ml_tree"]:
    print(f"\n{strategy} - alt2:")

    for metric in [
        "selection_ms",
        "preparation_ms",
        "search_ms",
        "total_ms",
    ]:
        delta = (
            wide[f"{metric}__{strategy}"]
            - wide[f"{metric}__alt2"]
        )

        print(
            f"  {metric:17s} "
            f"{delta.mean() * 1000:+.3f} us/OD"
        )

print("\n=== DESTINATION-CLUSTER BOOTSTRAP ===")

# Average across repeats for each OD first.
per_od = (
    wide.groupby(
        ["pair_id", "destination"],
        as_index=False,
    )
    .agg(
        gate_minus_alt2_ms=(
            "gate_minus_alt2_ms", "mean"
        ),
        ml_minus_alt2_ms=(
            "ml_minus_alt2_ms", "mean"
        ),
        ml_minus_gate_ms=(
            "ml_minus_gate_ms", "mean"
        ),
    )
)

# Resample destinations, not individual timing rows.
target_groups = [
    g.index.to_numpy()
    for _, g in per_od.groupby(
        "destination", sort=True
    )
]

rng = np.random.default_rng(42)
B = 3000

for metric in [
    "gate_minus_alt2_ms",
    "ml_minus_alt2_ms",
    "ml_minus_gate_ms",
]:
    values = per_od[metric].to_numpy()
    estimates = np.empty(B)

    for b in range(B):
        chosen = rng.integers(
            0,
            len(target_groups),
            size=len(target_groups),
        )

        indices = np.concatenate([
            target_groups[j] for j in chosen
        ])

        estimates[b] = values[indices].mean()

    low, high = np.quantile(
        estimates, [0.025, 0.975]
    )

    print(
        f"{metric}: "
        f"mean={values.mean():+.6f} ms/OD, "
        f"95% CI=[{low:+.6f}, {high:+.6f}]"
    )

OUTPUT.parent.mkdir(
    parents=True, exist_ok=True
)
wide.to_parquet(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
