from pathlib import Path

import numpy as np
import osmnx as ox
import pandas as pd

GRAPH_PATH = Path("data/graphs/porto_drive_routing.graphml")
INTERVALS_PATH = Path(
    "data/processed/ml_interval_training_data.parquet"
)
TRAVERSALS_PATH = Path(
    "data/processed/hmm_edge_traversals_pilot.parquet"
)
OUTPUT_PATH = Path(
    "data/processed/osm_interval_validation.parquet"
)

G = ox.load_graphml(GRAPH_PATH)

intervals = pd.read_parquet(INTERVALS_PATH)
traversals = pd.read_parquet(TRAVERSALS_PATH)

keys = ["trip_id", "point_index"]

# Original graph contains self-loops that may appear
# in HMM traversals, so load it as a fallback.
G_original = ox.load_graphml(
    "data/graphs/porto_drive_osm_weighted.graphml"
)


def edge_speed(u, v, k):
    if G.has_edge(u, v, k):
        return float(G[u][v][k]["osm_speed_kmh"])

    if G_original.has_edge(u, v, k):
        return float(
            G_original[u][v][k]["osm_speed_kmh"]
        )

    return np.nan


traversals = traversals.merge(
    intervals[keys].drop_duplicates(),
    on=keys,
    how="inner",
    validate="many_to_one",
)

traversals["osm_speed_kmh"] = [
    edge_speed(u, v, k)
    for u, v, k in zip(
        traversals["edge_u"],
        traversals["edge_v"],
        traversals["edge_key"],
    )
]

missing = traversals["osm_speed_kmh"].isna().sum()

print("=== EDGE MATCHING ===")
print("Traversal records:", len(traversals))
print("Missing edge speeds:", missing)

if missing:
    raise RuntimeError(
        "Some HMM edges are absent from the OSM graph."
    )

traversals["osm_time_s"] = (
    3.6
    * traversals["traversed_length_m"]
    / traversals["osm_speed_kmh"]
)

summary = (
    traversals.groupby(keys)
    .agg(
        osm_time_s=("osm_time_s", "sum"),
        traversed_distance_m=("traversed_length_m", "sum"),
    )
    .reset_index()
)

result = intervals.merge(
    summary,
    on=keys,
    how="inner",
    validate="one_to_one",
)

result["observed_time_s"] = result["interval_seconds"]
result["time_error_s"] = (
    result["osm_time_s"] - result["observed_time_s"]
)
result["absolute_time_error_s"] = (
    result["time_error_s"].abs()
)

print("\n=== OSM INTERVAL VALIDATION ===")
print("Intervals:", len(result))
print(
    "Time MAE (s):",
    round(result["absolute_time_error_s"].mean(), 3),
)
print(
    "Time bias (s):",
    round(result["time_error_s"].mean(), 3),
)
print(
    "OSM predicted median time (s):",
    round(result["osm_time_s"].median(), 3),
)

print("\n=== ERROR DISTRIBUTION ===")
print(
    result["time_error_s"]
    .describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    )
    .round(2)
    .to_string()
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result.to_parquet(OUTPUT_PATH, index=False)

print("\nSaved:", OUTPUT_PATH)
