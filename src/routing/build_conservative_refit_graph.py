from pathlib import Path
import json

import numpy as np
import osmnx as ox
import pandas as pd
from scipy.optimize import lsq_linear

FEATURES_PATH = Path(
    "data/processed/ml_road_class_time_features.parquet"
)
SOURCE_GRAPH = Path(
    "data/graphs/porto_drive_class_weighted.graphml"
)
OUTPUT_GRAPH = Path(
    "data/graphs/porto_drive_conservative_refit.graphml"
)
OUTPUT_COEFS = Path(
    "data/processed/conservative_refit_coefficients.json"
)

LAMBDA = 100.0

trainable_classes = [
    "motorway",
    "primary",
    "secondary",
    "tertiary",
]

df = pd.read_parquet(FEATURES_PATH)

X = df[
    [f"osm_time_{name}_s" for name in trainable_classes]
].to_numpy(dtype=float)

fixed_time = (
    df["osm_time_residential_s"].to_numpy(dtype=float)
    + df["osm_time_other_s"].to_numpy(dtype=float)
)

y = (
    df["observed_time_s"].to_numpy(dtype=float)
    - fixed_time
)

n, p = X.shape

A = np.vstack([
    X / np.sqrt(n),
    np.sqrt(LAMBDA) * np.eye(p),
])

b = np.concatenate([
    y / np.sqrt(n),
    np.sqrt(LAMBDA) * np.ones(p),
])

fit = lsq_linear(
    A,
    b,
    bounds=(0.5, 2.0),
)

if not fit.success:
    raise RuntimeError(fit.message)

coefficients = {
    name: float(value)
    for name, value in zip(trainable_classes, fit.x)
}

coefficients["residential"] = 1.0
coefficients["other"] = 1.0

G = ox.load_graphml(SOURCE_GRAPH)

rows = []

for u, v, key, attrs in G.edges(
    keys=True,
    data=True,
):
    road_class = attrs["class_road_group"]

    if road_class not in coefficients:
        raise ValueError(
            f"Unknown road class: {road_class}"
        )

    osm_time = float(attrs["osm_travel_time_s"])
    factor = coefficients[road_class]
    adjusted_time = osm_time * factor

    if not (
        np.isfinite(adjusted_time)
        and adjusted_time > 0
    ):
        raise ValueError(
            f"Invalid weight: {(u, v, key)}"
        )

    attrs["conservative_time_factor"] = factor
    attrs["conservative_travel_time_s"] = (
        adjusted_time
    )

    rows.append({
        "road_class": road_class,
        "osm_time_s": osm_time,
        "conservative_time_s": adjusted_time,
    })

OUTPUT_GRAPH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_COEFS.parent.mkdir(
    parents=True,
    exist_ok=True,
)

ox.save_graphml(G, OUTPUT_GRAPH)

OUTPUT_COEFS.write_text(
    json.dumps(
        {
            "lambda": LAMBDA,
            "training_routes": n,
            "training_trips": int(
                df["trip_id"].nunique()
            ),
            "coefficients": coefficients,
        },
        indent=2,
    ),
    encoding="utf-8",
)

summary = pd.DataFrame(rows)

report = (
    summary.groupby("road_class")
    .agg(
        edges=("osm_time_s", "size"),
        osm_total_s=("osm_time_s", "sum"),
        conservative_total_s=(
            "conservative_time_s", "sum"
        ),
    )
)

report["change_pct"] = (
    100
    * (
        report["conservative_total_s"]
        / report["osm_total_s"]
        - 1
    )
)

print("=== CONSERVATIVE REFIT COEFFICIENTS ===")

for name, value in coefficients.items():
    print(f"{name:15s} {value:.5f}")

print("\n=== GRAPH VALIDATION ===")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print(
    "Invalid weights:",
    int(
        (~np.isfinite(
            summary["conservative_time_s"]
        )).sum()
        + (
            summary["conservative_time_s"] <= 0
        ).sum()
    ),
)

print("\n=== WEIGHTS BY CLASS ===")
print(report.round(3).to_string())

print("\nSaved graph:", OUTPUT_GRAPH)
print("Saved coefficients:", OUTPUT_COEFS)
