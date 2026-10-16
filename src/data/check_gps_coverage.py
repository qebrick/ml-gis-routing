from pathlib import Path

import numpy as np
import pandas as pd
import osmnx as ox
from pyproj import Transformer

GRAPH_PATH = Path("data/graphs/porto_drive.graphml")
GPS_PATH = Path("data/processed/gps_intervals.parquet")
SAMPLE_SIZE = 5000
SEED = 42

G = ox.load_graphml(GRAPH_PATH)
df = pd.read_parquet(GPS_PATH)

# Проверяем оба конца каждого GPS-интервала.
lon = np.concatenate([df["lon_1"].to_numpy(), df["lon_2"].to_numpy()])
lat = np.concatenate([df["lat_1"].to_numpy(), df["lat_2"].to_numpy()])

nodes = pd.DataFrame(
    [(float(d["x"]), float(d["y"])) for _, d in G.nodes(data=True)],
    columns=["lon", "lat"],
)

min_lon, max_lon = nodes["lon"].min(), nodes["lon"].max()
min_lat, max_lat = nodes["lat"].min(), nodes["lat"].max()

inside = (
    (lon >= min_lon) & (lon <= max_lon)
    & (lat >= min_lat) & (lat <= max_lat)
)

print("=== GPS COVERAGE ===")
print(f"Total GPS points: {len(lon):,}")
print(f"Inside graph bounding box: {inside.mean() * 100:.2f}%")
print(f"Outside bounding box: {(~inside).mean() * 100:.2f}%")
print(
    "GPS bounds:",
    f"lon {lon.min():.5f} .. {lon.max():.5f};",
    f"lat {lat.min():.5f} .. {lat.max():.5f}",
)

# Для ближайших рёбер используем случайную выборку точек.
rng = np.random.default_rng(SEED)
n = min(SAMPLE_SIZE, len(lon))
idx = rng.choice(len(lon), size=n, replace=False)

G_proj = ox.project_graph(G)
transformer = Transformer.from_crs(
    "EPSG:4326", G_proj.graph["crs"], always_xy=True
)
xs, ys = transformer.transform(lon[idx], lat[idx])

print(f"\n=== NEAREST ROAD EDGE ({n:,} sampled points) ===")
try:
    _, distances = ox.distance.nearest_edges(
        G_proj, X=np.asarray(xs), Y=np.asarray(ys), return_dist=True
    )
    distances = np.asarray(distances, dtype=float)
    for threshold in (25, 50, 100, 250):
        pct = (distances <= threshold).mean() * 100
        print(f"Within {threshold:>3} m of an edge: {pct:.2f}%")
    print("Distance percentiles (m):")
    print(pd.Series(distances).quantile([0.50, 0.90, 0.95, 0.99]).to_string())
except Exception as exc:
    print("Nearest-edge check failed:", repr(exc))
    raise

print("\nCoverage check finished.")