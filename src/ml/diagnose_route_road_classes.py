from pathlib import Path
import ast

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

routes = pd.read_parquet(ROUTES_PATH)
routes = routes.loc[
    routes["target_distance_m"] == 500
].copy()

tr = pd.read_parquet(TRAVERSALS_PATH)
G = ox.load_graphml(GRAPH_PATH)


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


def road_group(highway):
    for name in [
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "residential",
        "living_street",
    ]:
        if highway.startswith(name):
            return name

    return "other"


tr["road_group"] = [
    road_group(
        normalize_highway(
            G[u][v][k].get("highway")
        )
    )
    for u, v, k in zip(
        tr["edge_u"],
        tr["edge_v"],
        tr["edge_key"],
    )
]

records = []

for route in routes.itertuples(index=False):
    segment = tr.loc[
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

    if segment.empty:
        continue

    lengths = (
        segment.groupby("road_group")[
            "traversed_length_m"
        ]
        .sum()
        .sort_values(ascending=False)
    )

    dominant = lengths.index[0]
    dominant_share = lengths.iloc[0] / lengths.sum()

    records.append({
        "trip_id": route.trip_id,
        "dominant_road_group": dominant,
        "dominant_share": dominant_share,
        "observed_time_s": route.observed_time_s,
        "osm_time_s": route.osm_time_s,
        "time_ratio": route.time_ratio,
        "osm_error_s": route.osm_error_s,
        "osm_abs_error_s": route.osm_abs_error_s,
    })

result = pd.DataFrame(records)

print("=== ROUTES BY DOMINANT ROAD CLASS ===")

summary = (
    result.groupby("dominant_road_group")
    .agg(
        routes=("trip_id", "size"),
        trips=("trip_id", "nunique"),
        median_dominant_share=(
            "dominant_share", "median"
        ),
        median_time_ratio=(
            "time_ratio", "median"
        ),
        mean_osm_error_s=(
            "osm_error_s", "mean"
        ),
        osm_mae_s=(
            "osm_abs_error_s", "mean"
        ),
    )
    .sort_values("routes", ascending=False)
)

print(summary.round(3).to_string())

print("\n=== STRONGLY DOMINATED ROUTES (>= 80%) ===")

strong = result.loc[
    result["dominant_share"] >= 0.8
]

print(
    strong.groupby("dominant_road_group")
    .agg(
        routes=("trip_id", "size"),
        median_time_ratio=(
            "time_ratio", "median"
        ),
        mean_osm_error_s=(
            "osm_error_s", "mean"
        ),
    )
    .round(3)
    .to_string()
)
