import numpy as np
import pandas as pd
import osmnx as ox

GRAPH_PATH = "data/graphs/porto_drive.graphml"
GPS_PATH = "data/processed/gps_intervals.parquet"

df = pd.read_parquet(GPS_PATH)
G = ox.load_graphml(GRAPH_PATH)

nodes = pd.DataFrame(
    [(float(d["x"]), float(d["y"])) for _, d in G.nodes(data=True)],
    columns=["lon", "lat"],
)

bounds = {
    "min_lon": nodes.lon.min(),
    "max_lon": nodes.lon.max(),
    "min_lat": nodes.lat.min(),
    "max_lat": nodes.lat.max(),
}

print("Graph bounds:", bounds)

# Собираем оба конца GPS-интервалов в единый набор точек.
starts = df[["trip_id", "lon_1", "lat_1"]].rename(
    columns={"lon_1": "lon", "lat_1": "lat"}
)
ends = df[["trip_id", "lon_2", "lat_2"]].rename(
    columns={"lon_2": "lon", "lat_2": "lat"}
)
pts = pd.concat([starts, ends], ignore_index=True).drop_duplicates(
    ["trip_id", "lon", "lat"]
)

# Проверяем координаты на NaN и бесконечности.
valid = (
    np.isfinite(pts["lon"]) & np.isfinite(pts["lat"])
    & pts["lon"].between(-180, 180)
    & pts["lat"].between(-90, 90)
)
print("\nTotal unique trip-points:", len(pts))
print("Invalid coordinates:", int((~valid).sum()))
pts = pts.loc[valid].copy()

# Расширение прямоугольника графа на 0.02° и 0.05°.
# Это только предварительный фильтр, не окончательная привязка к дорогам.
for margin in (0.02, 0.05):
    inside = (
        pts["lon"].between(bounds["min_lon"] - margin, bounds["max_lon"] + margin)
        & pts["lat"].between(bounds["min_lat"] - margin, bounds["max_lat"] + margin)
    )
    pts[f"inside_{margin}"] = inside

    by_trip = pts.groupby("trip_id")[f"inside_{margin}"].mean()
    print(f"\n=== Expanded bbox margin {margin:.2f} degrees ===")
    print(f"Points inside: {inside.mean() * 100:.2f}%")
    print(f"Trips with >= 80% points inside: {(by_trip >= 0.80).sum():,}")
    print(f"Trips with >= 50% points inside: {(by_trip >= 0.50).sum():,}")
    print(f"Total trips: {len(by_trip):,}")

# Поездки с наиболее низкой долей точек внутри расширенной области.
by_trip = pts.groupby("trip_id")["inside_0.05"].mean().sort_values()
print("\n=== 10 trips with lowest share inside expanded area ===")
print(by_trip.head(10).to_string())

# Распределение географического центра каждой поездки.
centers = pts.groupby("trip_id").agg(
    median_lon=("lon", "median"),
    median_lat=("lat", "median"),
)
print("\n=== Trip median coordinate quantiles ===")
print(centers.quantile([0, 0.1, 0.5, 0.9, 1]).to_string())

print("\nDiagnostic finished. No data has been modified.")
