import numpy as np
import pandas as pd
import osmnx as ox
from pyproj import Transformer

GRAPH_PATH = "data/graphs/porto_drive.graphml"
GPS_PATH = "data/processed/gps_intervals.parquet"
MARGIN = 0.05
MIN_TRIP_SHARE = 0.80
SAMPLE_SIZE = 5000
SEED = 42

G = ox.load_graphml(GRAPH_PATH)
df = pd.read_parquet(GPS_PATH)

# Создаём таблицу уникальных GPS-точек.
starts = df[["trip_id", "lon_1", "lat_1"]].rename(
    columns={"lon_1": "lon", "lat_1": "lat"}
)
ends = df[["trip_id", "lon_2", "lat_2"]].rename(
    columns={"lon_2": "lon", "lat_2": "lat"}
)
pts = pd.concat([starts, ends], ignore_index=True)
pts = pts.drop_duplicates(["trip_id", "lon", "lat"]).copy()

nodes = pd.DataFrame(
    [(float(d["x"]), float(d["y"])) for _, d in G.nodes(data=True)],
    columns=["lon", "lat"],
)

min_lon, max_lon = nodes.lon.min(), nodes.lon.max()
min_lat, max_lat = nodes.lat.min(), nodes.lat.max()

pts["inside_expanded"] = (
    pts["lon"].between(min_lon - MARGIN, max_lon + MARGIN)
    & pts["lat"].between(min_lat - MARGIN, max_lat + MARGIN)
)

trip_share = pts.groupby("trip_id")["inside_expanded"].mean()
eligible_ids = trip_share[trip_share >= MIN_TRIP_SHARE].index
eligible = pts[pts["trip_id"].isin(eligible_ids)].copy()

print("=== TRIP FILTER ===")
print(f"All trips: {len(trip_share):,}")
print(f"Eligible trips: {len(eligible_ids):,}")
print(f"Eligible trip share: {len(eligible_ids) / len(trip_share) * 100:.2f}%")
print(f"Unique points in eligible trips: {len(eligible):,}")

# Для диагностики выбираем точки внутри исходного bbox.
inside_graph_bbox = (
    eligible["lon"].between(min_lon, max_lon)
    & eligible["lat"].between(min_lat, max_lat)
)
candidate_points = eligible[inside_graph_bbox].copy()

print(f"Eligible-trip points inside original bbox: {len(candidate_points):,}")
print(
    "Fraction of eligible-trip points inside original bbox:",
    f"{inside_graph_bbox.mean() * 100:.2f}%",
)

if candidate_points.empty:
    raise RuntimeError("No candidate GPS points inside the graph bounding box.")

sample = candidate_points.sample(
    n=min(SAMPLE_SIZE, len(candidate_points)),
    random_state=SEED,
)

# Проецируем граф и GPS-точки в метрическую систему координат.
G_proj = ox.project_graph(G)
transformer = Transformer.from_crs(
    "EPSG:4326", G_proj.graph["crs"], always_xy=True
)
xs, ys = transformer.transform(
    sample["lon"].to_numpy(),
    sample["lat"].to_numpy(),
)

_, distances = ox.distance.nearest_edges(
    G_proj,
    X=np.asarray(xs),
    Y=np.asarray(ys),
    return_dist=True,
)
distances = np.asarray(distances, dtype=float)

print(f"\n=== NEAREST EDGE DISTANCE ({len(distances):,} points) ===")
for threshold in (25, 50, 100, 250):
    pct = (distances <= threshold).mean() * 100
    print(f"Within {threshold:>3} m: {pct:.2f}%")

print("\nDistance percentiles (m):")
print(
    pd.Series(distances)
    .quantile([0.50, 0.75, 0.90, 0.95, 0.99])
    .to_string()
)

print("\nDiagnostic finished. No data has been modified.")
