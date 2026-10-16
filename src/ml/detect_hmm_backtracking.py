from pathlib import Path

import pandas as pd

TRAVERSALS_PATH = Path(
    "data/processed/hmm_edge_traversals_pilot.parquet"
)
VALIDATION_PATH = Path(
    "data/processed/osm_interval_validation.parquet"
)
OUTPUT_PATH = Path(
    "data/processed/hmm_backtracking_diagnostics.parquet"
)

traversals = pd.read_parquet(TRAVERSALS_PATH)
validation = pd.read_parquet(VALIDATION_PATH)

keys = ["trip_id", "point_index"]

traversals = traversals.sort_values(
    keys + ["edge_order"]
).copy()

grouped = traversals.groupby(keys, sort=False)

traversals["next_u"] = grouped["edge_u"].shift(-1)
traversals["next_v"] = grouped["edge_v"].shift(-1)
traversals["next_length_m"] = grouped[
    "traversed_length_m"
].shift(-1)

traversals["immediate_reverse"] = (
    (traversals["edge_u"] == traversals["next_v"])
    & (traversals["edge_v"] == traversals["next_u"])
)

traversals["reverse_pair_length_m"] = (
    traversals["traversed_length_m"]
    + traversals["next_length_m"].fillna(0)
)

reversals = traversals.loc[
    traversals["immediate_reverse"]
].copy()

print("=== REVERSE TRAVERSALS ===")
print("Reverse pairs:", len(reversals))
print(
    "Affected intervals:",
    reversals[keys].drop_duplicates().shape[0]
)

if len(reversals):
    print("\n=== REVERSE PAIR LENGTHS ===")
    print(
        reversals["reverse_pair_length_m"]
        .describe(
            percentiles=[0.5, 0.75, 0.9, 0.95]
        )
        .round(2)
        .to_string()
    )

    print("\n=== LONG REVERSE PAIRS ===")

    print(
        reversals.nlargest(
            15, "reverse_pair_length_m"
        )[
            keys
            + [
                "edge_u",
                "edge_v",
                "traversed_length_m",
                "next_length_m",
                "reverse_pair_length_m",
            ]
        ]
        .round(2)
        .to_string(index=False)
    )

reverse_summary = (
    reversals.groupby(keys)
    .agg(
        reverse_pairs=("immediate_reverse", "size"),
        max_reverse_pair_m=(
            "reverse_pair_length_m", "max"
        ),
    )
    .reset_index()
)

result = validation.merge(
    reverse_summary,
    on=keys,
    how="left",
    validate="one_to_one",
)

result["reverse_pairs"] = (
    result["reverse_pairs"].fillna(0).astype(int)
)
result["max_reverse_pair_m"] = (
    result["max_reverse_pair_m"].fillna(0)
)

result["flag_long_reverse"] = (
    result["max_reverse_pair_m"] >= 100
)

print("\n=== VALIDATED INTERVALS ===")
print("Intervals:", len(result))
print(
    "With any immediate reverse:",
    int((result["reverse_pairs"] > 0).sum())
)
print(
    "With reverse pair >= 100 m:",
    int(result["flag_long_reverse"].sum())
)

print("\n=== OSM ERRORS BY REVERSE FLAG ===")
print(
    result.groupby("flag_long_reverse")
    .agg(
        intervals=("trip_id", "size"),
        mean_error_s=("time_error_s", "mean"),
        mae_s=("absolute_time_error_s", "mean"),
        median_speed_kmh=("target_speed_kmh", "median"),
    )
    .round(2)
    .to_string()
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result.to_parquet(OUTPUT_PATH, index=False)

print("\nSaved:", OUTPUT_PATH)
