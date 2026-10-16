from pathlib import Path
import ast

import numpy as np
import osmnx as ox
import pandas as pd

GRAPH_PATH = Path(
    "data/graphs/porto_drive_osm_weighted.graphml"
)
VALIDATION_PATH = Path(
    "data/processed/hmm_backtracking_diagnostics.parquet"
)
TRAVERSALS_PATH = Path(
    "data/processed/hmm_edge_traversals_pilot.parquet"
)
OUTPUT_PATH = Path(
    "data/processed/ml_interval_features.parquet"
)

G = ox.load_graphml(GRAPH_PATH)
df = pd.read_parquet(VALIDATION_PATH)
tr = pd.read_parquet(TRAVERSALS_PATH)

keys = ["trip_id", "point_index"]

# Use the primary quality scenario.
df = df.loc[
    df["max_reverse_pair_m"] < 100
].copy()

tr = tr.merge(
    df[keys],
    on=keys,
    how="inner",
    validate="many_to_one",
)


def highway_name(value):
    if isinstance(value, list):
        return str(value[0]) if value else "unknown"

    if isinstance(value, str) and value.startswith("["):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list) and parsed:
                return str(parsed[0])
        except (ValueError, SyntaxError):
            pass

    return str(value) if value is not None else "unknown"


def edge_attributes(u, v, k):
    attrs = G[u][v][k]
    return (
        highway_name(attrs.get("highway")),
        float(attrs["osm_speed_kmh"]),
    )


attributes = [
    edge_attributes(u, v, k)
    for u, v, k in zip(
        tr["edge_u"],
        tr["edge_v"],
        tr["edge_key"],
    )
]

tr["highway"] = [item[0] for item in attributes]
tr["osm_speed_kmh"] = [item[1] for item in attributes]

# Only road composition features, not the observed
# interval distance, number of traversals, or OSM time.
highway_groups = [
    "motorway",
    "primary",
    "secondary",
    "tertiary",
    "residential",
    "living_street",
]

tr["road_group"] = tr["highway"].map(
    lambda x: next(
        (name for name in highway_groups if x.startswith(name)),
        "other",
    )
)

tr["weighted_osm_speed"] = (
    tr["osm_speed_kmh"] * tr["traversed_length_m"]
)

aggregated = (
    tr.groupby(keys)
    .agg(
        total_length=("traversed_length_m", "sum"),
        weighted_speed_sum=("weighted_osm_speed", "sum"),
    )
    .reset_index()
)

aggregated["mean_osm_speed_kmh"] = (
    aggregated["weighted_speed_sum"]
    / aggregated["total_length"]
)

for road_group in highway_groups + ["other"]:
    column = f"share_{road_group}"

    part = (
        tr.loc[tr["road_group"] == road_group]
        .groupby(keys)["traversed_length_m"]
        .sum()
        .rename(column)
        .reset_index()
    )

    aggregated = aggregated.merge(
        part,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    aggregated[column] = (
        aggregated[column].fillna(0)
        / aggregated["total_length"]
    )

feature_columns = [
    "mean_osm_speed_kmh",
] + [
    f"share_{name}"
    for name in highway_groups + ["other"]
]

result = df.merge(
    aggregated[keys + feature_columns],
    on=keys,
    how="inner",
    validate="one_to_one",
)

result["target_time_ratio"] = (
    result["observed_time_s"] / result["osm_time_s"]
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result.to_parquet(OUTPUT_PATH, index=False)

print("=== INTERVAL FEATURES ===")
print("Rows:", len(result))
print("Trips:", result["trip_id"].nunique())

print("\n=== FEATURE DISTRIBUTIONS ===")
print(
    result[feature_columns]
    .describe()
    .round(3)
    .to_string()
)

print("\n=== TARGET TIME RATIO ===")
print(
    result["target_time_ratio"]
    .describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    )
    .round(3)
    .to_string()
)

print("\n=== CORRELATIONS WITH TARGET RATIO ===")
print(
    result[feature_columns]
    .corrwith(result["target_time_ratio"])
    .sort_values()
    .round(3)
    .to_string()
)

print("\nSaved:", OUTPUT_PATH)
