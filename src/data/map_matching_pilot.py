from pathlib import Path

import numpy as np
import pandas as pd
import osmnx as ox
from pyproj import Transformer

GRAPH_PATH = Path("data/graphs/porto_drive.graphml")
GPS_PATH = Path("data/processed/gps_intervals.parquet")
OUT_PATH = Path("data/processed/map_matching_pilot.parquet")

MARGIN = 0.05
MIN_TRIP_SHARE = 0.80
N_TRIPS = 100
SEED = 42

# Load graph and GPS intervals.
G = ox.load_graphml(GRAPH_PATH)
df = pd.read_parquet(GPS_PATH)

# Convert intervals into an ordered table of GPS points.
starts = df[["trip_id", "point_index", "lon_1", "lat_1"]].rename(
    columns={"lon_1": "lon", "lat_1": "lat"}
)
ends = df[["trip_id", "point_index", "lon_2", "lat_2"]].copy()
ends["point_index"] += 1
ends = ends.rename(columns={"lon_2": "lon", "lat_2": "lat"})

points = pd.concat([starts, ends], ignore_index=True)
points = points.drop_duplicates(
    ["trip_id", "point_index"], keep="first"
).copy()

# Determine the graph bounding box.
nodes = pd.DataFrame(
    [(float(d["x"]), float(d["y"])) for _, d in G.nodes(data=True)],
    columns=["lon", "lat"],
)
min_lon, max_lon = nodes.lon.min(), nodes.lon.max()
min_lat, max_lat = nodes.lat.min(), nodes.lat.max()

points["inside_expanded"] = (
    points["lon"].between(min_lon - MARGIN, max_lon + MARGIN)
    & points["lat"].between(min_lat - MARGIN, max_lat + MARGIN)
)

# Retain trips with at least 80% of points in the expanded area.
trip_share = points.groupby("trip_id")["inside_expanded"].mean()
eligible_ids = trip_share[trip_share >= MIN_TRIP_SHARE].index

eligible = points[points["trip_id"].isin(eligible_ids)].copy()
rng = np.random.default_rng(SEED)
chosen_ids = rng.choice(
    eligible_ids.to_numpy(),
    size=min(N_TRIPS, len(eligible_ids)),
    replace=False,
)
sample = eligible[eligible["trip_id"].isin(chosen_ids)].copy()
sample = sample.sort_values(["trip_id", "point_index"]).reset_index(drop=True)

print("Eligible trips:", len(eligible_ids))
print("Pilot trips:", sample["trip_id"].nunique())
print("GPS points to snap:", len(sample))

# Project graph and GPS points to a metric CRS.
G_proj = ox.project_graph(G)
transformer = Transformer.from_crs(
    "EPSG:4326", G_proj.graph["crs"], always_xy=True
)
xs, ys = transformer.transform(
    sample["lon"].to_numpy(),
    sample["lat"].to_numpy(),
)

# Snap each point to its nearest directed edge.
edge_ids, distances = ox.distance.nearest_edges(
    G_proj,
    X=np.asarray(xs),
    Y=np.asarray(ys),
    return_dist=True,
)

sample["edge_u"] = [edge[0] for edge in edge_ids]
sample["edge_v"] = [edge[1] for edge in edge_ids]
sample["edge_key"] = [edge[2] for edge in edge_ids]
sample["distance_to_edge_m"] = np.asarray(distances, dtype=float)

# Mark points outside the original graph bounding box.
sample["inside_graph_bbox"] = (
    sample["lon"].between(min_lon, max_lon)
    & sample["lat"].between(min_lat, max_lat)
)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
sample.to_parquet(OUT_PATH, index=False)

print("\n=== PILOT RESULTS ===")
print("Saved:", OUT_PATH)
print("Points inside original bbox:",
      f"{sample['inside_graph_bbox'].mean() * 100:.2f}%")
print("Distance to nearest edge (m):")
print(sample["distance_to_edge_m"].describe(
    percentiles=[0.50, 0.90, 0.95, 0.99]
).to_string())

for threshold in (25, 50, 100, 250):
    pct = (sample["distance_to_edge_m"] <= threshold).mean() * 100
    print(f"Within {threshold:>3} m: {pct:.2f}%")

# Count changes in nearest-edge assignments within each trip.
edge_cols = ["edge_u", "edge_v", "edge_key"]

previous_edges = sample.groupby("trip_id")[edge_cols].shift(1)

sample["edge_changed"] = (
    sample[edge_cols].ne(previous_edges).any(axis=1)
)

# First point of each trip is not a transition.
first_point = sample.groupby("trip_id").cumcount() == 0
sample.loc[first_point, "edge_changed"] = False

print(
    "\nPoints where nearest-edge assignment changes:",
    int(sample["edge_changed"].sum())
)

print(
    "Share of transitions with changed nearest edge:",
    f"{sample['edge_changed'].sum() / max(len(sample) - sample['trip_id'].nunique(), 1) * 100:.2f}%"
)

print("\nThis is nearest-edge snapping, not sequence-aware map matching.")
