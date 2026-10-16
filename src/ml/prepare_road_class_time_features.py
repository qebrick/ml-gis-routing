from pathlib import Path
import ast

import numpy as np
import osmnx as ox
import pandas as pd

ROUTES_PATH = Path(
    "data/processed/ml_distance_based_routes.parquet"
)
TRAVERSALS_PATH = Path(
    "data/processed/hmm_edge_traversals_pilot.parquet"
)
GRAPH_PATH = Path(
    "data/graphs/porto_drive_osm_weighted.graphml"
)
OUTPUT_PATH = Path(
    "data/processed/ml_road_class_time_features.parquet"
)

routes = pd.read_parquet(ROUTES_PATH)
routes = routes.loc[
    routes["target_distance_m"] == 500
].copy()

tr = pd.read_parquet(TRAVERSALS_PATH)
G = ox.load_graphml(GRAPH_PATH)

classes = [
    "motorway",
    "primary",
    "secondary",
    "tertiary",
    "residential",
    "other",
]


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

    for name in classes[:-1]:
        if highway.startswith(name):
            return name

    return "other"


attributes = []

for u, v, k in zip(
    tr["edge_u"],
    tr["edge_v"],
    tr["edge_key"],
):
    edge = G[u][v][k]

    attributes.append((
        road_class(edge.get("highway")),
        float(edge["osm_speed_kmh"]),
    ))

tr["road_class"] = [x[0] for x in attributes]
tr["osm_speed_kmh"] = [x[1] for x in attributes]

tr["partial_osm_time_s"] = (
    3.6
    * tr["traversed_length_m"]
    / tr["osm_speed_kmh"]
)

records = []

for route in routes.itertuples(index=False):
    part = tr.loc[
        (tr["trip_id"] == route.trip_id)
        & (
            tr["hmm_fragment_id"]
            == route.hmm_fragment_id
        )
        & (
            tr["point_index"]
            >= route.start_point_index
        )
        & (
            tr["point_index"]
            <= route.end_point_index
        )
    ]

    times = (
        part.groupby("road_class")[
            "partial_osm_time_s"
        ]
        .sum()
    )

    record = {
        "trip_id": route.trip_id,
        "hmm_fragment_id": route.hmm_fragment_id,
        "start_point_index": route.start_point_index,
        "end_point_index": route.end_point_index,
        "observed_time_s": route.observed_time_s,
        "osm_time_s": route.osm_time_s,
        "route_distance_m": route.route_distance_m,
    }

    for name in classes:
        record[f"osm_time_{name}_s"] = float(
            times.get(name, 0.0)
        )

    records.append(record)

result = pd.DataFrame(records)

feature_columns = [
    f"osm_time_{name}_s"
    for name in classes
]

result["sum_class_time_s"] = (
    result[feature_columns].sum(axis=1)
)

result["time_difference_s"] = (
    result["sum_class_time_s"]
    - result["osm_time_s"]
)

print("=== CLASS TIME CONSISTENCY ===")
print("Routes:", len(result))
print(
    "Max absolute difference:",
    result["time_difference_s"].abs().max()
)
print(
    "Mean absolute difference:",
    result["time_difference_s"].abs().mean()
)

print("\n=== CLASS TIME DISTRIBUTIONS ===")
print(
    result[feature_columns]
    .describe()
    .round(2)
    .to_string()
)

print("\n=== CLASS COVERAGE ===")
for column in feature_columns:
    print(
        column,
        "routes:",
        int((result[column] > 0).sum()),
        "trips:",
        result.loc[
            result[column] > 0, "trip_id"
        ].nunique(),
    )

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)
result.to_parquet(
    OUTPUT_PATH,
    index=False,
)

print("\nSaved:", OUTPUT_PATH)
