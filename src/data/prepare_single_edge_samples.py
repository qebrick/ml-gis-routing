from pathlib import Path

import pandas as pd

TRAVERSALS_PATH = Path(
    "data/processed/hmm_edge_traversals_pilot.parquet"
)
QUALITY_PATH = Path(
    "data/processed/hmm_interval_quality.parquet"
)
OUTPUT_PATH = Path(
    "data/processed/single_edge_samples_pilot.parquet"
)

traversals = pd.read_parquet(TRAVERSALS_PATH)
quality = pd.read_parquet(QUALITY_PATH)

KEY = ["hmm_fragment_id", "point_index"]

counts = (
    traversals.groupby(KEY)
    .size()
    .rename("edge_parts_count")
    .reset_index()
)

single = traversals.merge(
    counts,
    on=KEY,
    how="left",
    validate="many_to_one",
)

single = single[
    single["edge_parts_count"] == 1
].copy()

quality_cols = KEY + [
    "flag_route_shorter_than_gps",
    "flag_large_detour",
    "flag_zero_route_moving_gps",
]

single = single.merge(
    quality[quality_cols],
    on=KEY,
    how="left",
    validate="one_to_one",
    indicator=True,
)

if not single["_merge"].eq("both").all():
    missing = single.loc[
        single["_merge"] != "both",
        KEY
    ]
    raise RuntimeError(
        "Unmatched quality records:\n"
        + missing.to_string(index=False)
    )

single = single.drop(columns="_merge")

missing_flags = single[quality_cols[2:]].isna().any(axis=1)

print("Single-edge records before quality filtering:", len(single))
print("Records with missing quality flags:", int(missing_flags.sum()))

if missing_flags.any():
    print("\nExamples:")
    print(
        single.loc[
            missing_flags,
            ["hmm_fragment_id", "point_index", "trip_id"]
        ].head(10).to_string(index=False)
    )

    print("\nQuality key types:")
    print(quality[KEY].dtypes)

    print("\nTraversal key types:")
    print(traversals[KEY].dtypes)

    raise RuntimeError("Missing interval quality flags.")

single = single[
    ~single["flag_route_shorter_than_gps"]
    & ~single["flag_large_detour"]
    & ~single["flag_zero_route_moving_gps"]
    & (single["traversed_length_m"] >= 20)
    & (single["interval_seconds"] == 15)
].copy()

single["observed_time_seconds"] = (
    single["interval_seconds"]
)

single["observed_speed_kmh"] = (
    single["traversed_length_m"]
    / single["observed_time_seconds"]
    * 3.6
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
single.to_parquet(OUTPUT_PATH, index=False)

print("=== SINGLE-EDGE SAMPLES ===")
print("Original traversal records:", len(traversals))
print("Single-edge samples after filtering:", len(single))

print(
    "Unique directed edges:",
    single[["edge_u", "edge_v", "edge_key"]]
    .drop_duplicates()
    .shape[0],
)

print(
    "Unique trips:",
    single["trip_id"].nunique(),
)

print("\nTraversed length (m):")
print(
    single["traversed_length_m"]
    .describe(percentiles=[0.5, 0.9, 0.95])
    .to_string()
)

print("\nObserved speed (km/h):")
print(
    single["observed_speed_kmh"]
    .describe(percentiles=[0.5, 0.9, 0.95])
    .to_string()
)

print("\nSaved:", OUTPUT_PATH)
