from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

GRAPH_PATH = Path("data/graphs/porto_drive_routing.graphml")
SEGMENTS_PATH = Path(
    "data/processed/map_matching_segments_v2.parquet"
)
OUTPUT_PATH = Path(
    "data/processed/route_objective_comparison.parquet"
)

MAX_SEGMENTS = 100
MIN_POINTS = 5
MAX_SNAP_DISTANCE_M = 100.0

G = ox.load_graphml(GRAPH_PATH)

for u, v, k, d in G.edges(keys=True, data=True):
    d["length"] = float(d["length"])
    d["osm_travel_time_s"] = float(d["osm_travel_time_s"])

# Nearest-node queries require a projected graph.
G_projected = ox.project_graph(G)

segments = pd.read_parquet(SEGMENTS_PATH)

print("=== INPUT DATA ===")
print("Rows:", len(segments))
print("Columns:", list(segments.columns))

required = {
    "segment_id",
    "point_index",
    "lon",
    "lat",
}

missing = required - set(segments.columns)

if missing:
    raise RuntimeError(
        f"Missing required columns: {sorted(missing)}"
    )

rows = []

for segment_id, group in segments.groupby("segment_id"):
    group = group.sort_values("point_index")

    if len(group) < MIN_POINTS:
        continue

    start = group.iloc[0]
    end = group.iloc[-1]

    rows.append({
        "segment_id": segment_id,
        "trip_id": start["trip_id"],
        "gps_points": len(group),
        "start_lon": float(start["lon"]),
        "start_lat": float(start["lat"]),
        "end_lon": float(end["lon"]),
        "end_lat": float(end["lat"]),
    })

pairs = pd.DataFrame(rows)

if pairs.empty:
    raise RuntimeError("No valid GPS segments.")

pairs = pairs.head(MAX_SEGMENTS).copy()

# Transform GPS coordinates into the projected graph CRS.
import geopandas as gpd

starts = gpd.GeoSeries(
    gpd.points_from_xy(
        pairs["start_lon"],
        pairs["start_lat"],
    ),
    crs="EPSG:4326",
).to_crs(G_projected.graph["crs"])

ends = gpd.GeoSeries(
    gpd.points_from_xy(
        pairs["end_lon"],
        pairs["end_lat"],
    ),
    crs="EPSG:4326",
).to_crs(G_projected.graph["crs"])

start_nodes, start_distances = ox.distance.nearest_nodes(
    G_projected,
    X=starts.x.to_numpy(),
    Y=starts.y.to_numpy(),
    return_dist=True,
)

end_nodes, end_distances = ox.distance.nearest_nodes(
    G_projected,
    X=ends.x.to_numpy(),
    Y=ends.y.to_numpy(),
    return_dist=True,
)

pairs["start_node"] = start_nodes
pairs["end_node"] = end_nodes
pairs["start_snap_m"] = start_distances
pairs["end_snap_m"] = end_distances


def path_metrics(path, optimize_for):
    distance_m = 0.0
    time_s = 0.0

    for u, v in zip(path[:-1], path[1:]):
        edge = min(
            G[u][v].values(),
            key=lambda d: d[optimize_for],
        )

        distance_m += edge["length"]
        time_s += edge["osm_travel_time_s"]

    return distance_m, time_s


results = []
skipped_snap = 0
skipped_unreachable = 0
skipped_same_node = 0

for _, row in pairs.iterrows():
    if (
        row["start_snap_m"] > MAX_SNAP_DISTANCE_M
        or row["end_snap_m"] > MAX_SNAP_DISTANCE_M
    ):
        skipped_snap += 1
        continue

    source = row["start_node"]
    target = row["end_node"]

    if source == target:
        skipped_same_node += 1
        continue

    try:
        shortest_path = nx.shortest_path(
            G,
            source,
            target,
            weight="length",
            method="dijkstra",
        )

        fastest_path = nx.shortest_path(
            G,
            source,
            target,
            weight="osm_travel_time_s",
            method="dijkstra",
        )

    except nx.NetworkXNoPath:
        skipped_unreachable += 1
        continue

    shortest_distance, shortest_time = path_metrics(
        shortest_path,
        optimize_for="length",
    )

    fastest_distance, fastest_time = path_metrics(
        fastest_path,
        optimize_for="osm_travel_time_s",
    )

    if fastest_time > shortest_time + 1e-6:
        raise RuntimeError(
            "Fastest route is slower than shortest route."
        )

    if shortest_distance > fastest_distance + 1e-6:
        raise RuntimeError(
            "Shortest route is longer than fastest route."
        )

    results.append({
        "segment_id": row["segment_id"],
        "trip_id": row["trip_id"],
        "gps_points": row["gps_points"],
        "shortest_distance_m": shortest_distance,
        "shortest_time_s": shortest_time,
        "fastest_distance_m": fastest_distance,
        "fastest_time_s": fastest_time,
        "distance_difference_m": (
            fastest_distance - shortest_distance
        ),
        "time_saving_s": (
            shortest_time - fastest_time
        ),
    })

result = pd.DataFrame(results)

if result.empty:
    raise RuntimeError("No reachable route pairs.")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result.to_parquet(OUTPUT_PATH, index=False)

print("\n=== ROUTE COMPARISON ===")
print("Candidate segments:", len(pairs))
print("Compared routes:", len(result))
print("Skipped: snap distance:", skipped_snap)
print("Skipped: unreachable:", skipped_unreachable)
print("Skipped: same node:", skipped_same_node)

print("\n=== DISTANCE (METERS) ===")
print(
    result[
        ["shortest_distance_m", "fastest_distance_m"]
    ].describe().round(2).to_string()
)

print("\n=== TIME (SECONDS) ===")
print(
    result[
        ["shortest_time_s", "fastest_time_s"]
    ].describe().round(2).to_string()
)

print("\n=== DIFFERENCES ===")
print(
    result[
        ["distance_difference_m", "time_saving_s"]
    ].describe(
        percentiles=[0.5, 0.9, 0.95]
    ).round(2).to_string()
)

print(
    "\nRoutes with positive OSM-estimated time saving:",
    int((result["time_saving_s"] > 1e-6).sum()),
)

print("\nSaved:", OUTPUT_PATH)
