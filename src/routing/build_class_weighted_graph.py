from pathlib import Path
import ast
import json

import numpy as np
import osmnx as ox
import pandas as pd
from scipy.optimize import lsq_linear

DATA_PATH = Path(
    "data/processed/ml_road_class_time_features.parquet"
)
GRAPH_PATH = Path(
    "data/graphs/porto_drive_routing.graphml"
)
OUTPUT_GRAPH = Path(
    "data/graphs/porto_drive_class_weighted.graphml"
)
OUTPUT_COEFS = Path(
    "data/processed/class_weight_coefficients.json"
)

LAMBDA = 100.0

classes = [
    "motorway",
    "primary",
    "secondary",
    "tertiary",
    "residential",
]

features = [
    f"osm_time_{name}_s"
    for name in classes
]

df = pd.read_parquet(DATA_PATH)

X = df[features].to_numpy(dtype=float)
y = (
    df["observed_time_s"].to_numpy(dtype=float)
    - df["osm_time_other_s"].to_numpy(dtype=float)
)

n = len(df)

A = np.vstack([
    X / np.sqrt(n),
    np.sqrt(LAMBDA) * np.eye(len(classes)),
])

b = np.concatenate([
    y / np.sqrt(n),
    np.sqrt(LAMBDA) * np.ones(len(classes)),
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
    for name, value in zip(classes, fit.x)
}
coefficients["other"] = 1.0


def normalize_highway(value):
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


def road_class(value):
    highway = normalize_highway(value)

    for name in classes:
        if highway.startswith(name):
            return name

    return "other"


G = ox.load_graphml(GRAPH_PATH)

records = []

for u, v, k, attrs in G.edges(
    keys=True,
    data=True,
):
    base_time = float(attrs["osm_travel_time_s"])

    road = road_class(attrs.get("highway"))
    factor = coefficients[road]

    new_time = base_time * factor

    if not (
        np.isfinite(new_time)
        and new_time > 0
    ):
        raise RuntimeError(
            f"Invalid weight for edge {(u, v, k)}"
        )

    attrs["class_road_group"] = road
    attrs["class_time_factor"] = factor
    attrs["class_travel_time_s"] = new_time

    records.append({
        "road_class": road,
        "osm_time_s": base_time,
        "class_time_s": new_time,
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
            "coefficients": coefficients,
        },
        indent=2,
    ),
    encoding="utf-8",
)

summary = pd.DataFrame(records)

print("=== FITTED COEFFICIENTS ===")

for name, value in coefficients.items():
    print(f"{name:15s} {value:.4f}")

print("\n=== GRAPH ===")
print("Nodes:", len(G.nodes))
print("Edges:", len(G.edges))

print("\n=== EDGE WEIGHTS BY ROAD CLASS ===")

report = (
    summary.groupby("road_class")
    .agg(
        edges=("osm_time_s", "size"),
        osm_total_s=("osm_time_s", "sum"),
        class_total_s=("class_time_s", "sum"),
    )
)

report["time_change_pct"] = (
    100
    * (
        report["class_total_s"]
        / report["osm_total_s"]
        - 1
    )
)

print(report.round(2).to_string())

print("\nSaved graph:", OUTPUT_GRAPH)
print("Saved coefficients:", OUTPUT_COEFS)
