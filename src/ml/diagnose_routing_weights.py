from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path(
    "data/processed/ml_edge_predictions.parquet"
)

df = pd.read_parquet(INPUT_PATH).copy()

df["speed_ratio_ml_osm"] = (
    df["routing_speed_kmh"] / df["osm_speed_kmh"]
)

df["speed_difference_kmh"] = (
    df["routing_speed_kmh"] - df["osm_speed_kmh"]
)

df["length_bin"] = pd.cut(
    df["edge_length_m"],
    bins=[0, 25, 50, 100, 200, 400, 1000, np.inf],
    labels=[
        "0-25",
        "25-50",
        "50-100",
        "100-200",
        "200-400",
        "400-1000",
        "1000+",
    ],
    right=False,
)

print("=== OVERALL COMPARISON ===")
print("Edges:", len(df))

print(
    "ML slower than OSM:",
    int((df["routing_speed_kmh"] < df["osm_speed_kmh"]).sum()),
)

print(
    "ML faster than OSM:",
    int((df["routing_speed_kmh"] > df["osm_speed_kmh"]).sum()),
)

print(
    "Equal speeds:",
    int(np.isclose(
        df["routing_speed_kmh"],
        df["osm_speed_kmh"],
    ).sum()),
)

print("\nSpeed ratio ML / OSM:")
print(
    df["speed_ratio_ml_osm"]
    .describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    )
    .round(3)
    .to_string()
)

print("\n=== BY EDGE LENGTH ===")

summary = (
    df.groupby("length_bin", observed=True)
    .agg(
        edges=("edge_length_m", "size"),
        median_length_m=("edge_length_m", "median"),
        median_osm_speed=("osm_speed_kmh", "median"),
        median_ml_speed=("routing_speed_kmh", "median"),
        median_ratio=("speed_ratio_ml_osm", "median"),
        fallback_share=("outside_train_length", "mean"),
    )
)

summary["fallback_share"] *= 100

print(summary.round(2).to_string())

print("\n=== BY HIGHWAY ===")

highway_summary = (
    df.groupby("highway")
    .agg(
        edges=("highway", "size"),
        median_osm_speed=("osm_speed_kmh", "median"),
        median_ml_speed=("routing_speed_kmh", "median"),
        median_ratio=("speed_ratio_ml_osm", "median"),
    )
    .sort_values("edges", ascending=False)
    .head(12)
)

print(highway_summary.round(2).to_string())

print("\n=== EXTREME SLOWDOWNS ===")

extreme = df.nsmallest(
    15,
    "speed_ratio_ml_osm",
)[
    [
        "edge_u",
        "edge_v",
        "highway",
        "edge_length_m",
        "osm_speed_kmh",
        "raw_ml_speed_kmh",
        "routing_speed_kmh",
        "speed_ratio_ml_osm",
    ]
]

print(extreme.round(2).to_string(index=False))
