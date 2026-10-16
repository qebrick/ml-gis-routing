from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

GRAPH_PATH = Path(
    "data/graphs/porto_drive_conservative_refit.graphml"
)
OD_PATH = Path(
    "data/processed/gps_od_route_comparison.parquet"
)
TRAVERSALS_PATH = Path(
    "data/processed/hmm_edge_traversals_pilot.parquet"
)
OUTPUT_PATH = Path(
    "data/processed/gps_path_overlap_comparison.parquet"
)

G = ox.load_graphml(GRAPH_PATH)
od = pd.read_parquet(OD_PATH)
tr = pd.read_parquet(TRAVERSALS_PATH)

OBJECTIVES = {
    "osm": "osm_travel_time_s",
    "class": "class_travel_time_s",
    "conservative": "conservative_travel_time_s",
}


def edge_id(u, v, key):
    return (int(u), int(v), int(key))


def weight_fn(attribute):
    def weight(u, v, parallel_edges):
        return min(
            float(attrs[attribute])
            for attrs in parallel_edges.values()
        )
    return weight


def routed_edges(origin, destination, attribute):
    path = nx.shortest_path(
        G,
        source=origin,
        target=destination,
        weight=weight_fn(attribute),
    )

    edges = set()

    for u, v in zip(path[:-1], path[1:]):
        key, _ = min(
            G[u][v].items(),
            key=lambda item: float(
                item[1][attribute]
            ),
        )
        edges.add(edge_id(u, v, key))

    return edges


def gps_edges_for_fragment(trip_id, fragment_id):
    part = tr.loc[
        (tr["trip_id"] == trip_id)
        & (tr["hmm_fragment_id"] == fragment_id)
        & (tr["traversed_length_m"] > 0)
    ]

    return {
        edge_id(row.edge_u, row.edge_v, row.edge_key)
        for row in part.itertuples(index=False)
    }


def edge_length(edge):
    u, v, key = edge
    return float(G[u][v][key]["length"])


def gps_recall(gps_edges, route_edges):
    denominator = sum(
        edge_length(e) for e in gps_edges
    )

    if denominator <= 0:
        return np.nan

    numerator = sum(
        edge_length(e)
        for e in gps_edges & route_edges
    )

    return numerator / denominator


rows = []
missing_gps_edges = 0

for pair in od.itertuples(index=False):
    gps_edges = gps_edges_for_fragment(
        pair.trip_id,
        pair.hmm_fragment_id,
    )

    valid_gps_edges = set()

    for edge in gps_edges:
        u, v, key = edge
        if G.has_edge(u, v, key):
            valid_gps_edges.add(edge)
        else:
            missing_gps_edges += 1

    if not valid_gps_edges:
        continue

    route_edges = {}

    for name, attribute in OBJECTIVES.items():
        route_edges[name] = routed_edges(
            pair.origin,
            pair.destination,
            attribute,
        )

    recalls = {
        name: gps_recall(
            valid_gps_edges,
            route_edges[name],
        )
        for name in OBJECTIVES
    }

    rows.append({
        "trip_id": pair.trip_id,
        "hmm_fragment_id": pair.hmm_fragment_id,
        "class_changed": pair.class_changed,
        "conservative_changed": (
            pair.conservative_changed
        ),
        "gps_unique_edges": len(valid_gps_edges),
        "recall_osm": recalls["osm"],
        "recall_class": recalls["class"],
        "recall_conservative": (
            recalls["conservative"]
        ),
        "delta_class": (
            recalls["class"] - recalls["osm"]
        ),
        "delta_conservative": (
            recalls["conservative"]
            - recalls["osm"]
        ),
    })

result = pd.DataFrame(rows)

if result.empty:
    raise RuntimeError("No valid GPS overlap comparisons")

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)
result.to_parquet(OUTPUT_PATH, index=False)

print("=== GPS PATH OVERLAP COVERAGE ===")
print("OD pairs:", len(result))
print("Trips:", result["trip_id"].nunique())
print("Missing GPS edges:", missing_gps_edges)

print("\n=== ALL ROUTES ===")
print(
    result[
        [
            "recall_osm",
            "recall_class",
            "recall_conservative",
            "delta_class",
            "delta_conservative",
        ]
    ]
    .agg(["mean", "median", "std"])
    .round(4)
    .to_string()
)

print("\n=== CHANGED ROUTES ONLY ===")

for model in ["class", "conservative"]:
    subset = result.loc[
        result[f"{model}_changed"]
    ]

    delta = subset[f"delta_{model}"]

    print(f"\n{model.upper()}")
    print("Routes:", len(subset))
    print("Mean recall change:", round(delta.mean(), 4))
    print("Median recall change:", round(delta.median(), 4))
    print("Better overlap:", int((delta > 1e-9).sum()))
    print("Worse overlap:", int((delta < -1e-9).sum()))
    print(
        "Equal overlap:",
        int((delta.abs() <= 1e-9).sum()),
    )

print("\n=== TRIP-LEVEL CONSERVATIVE SUMMARY ===")

trip_summary = (
    result.groupby("trip_id")["delta_conservative"]
    .mean()
)

print("Trips:", len(trip_summary))
print(
    "Mean trip-level delta:",
    round(trip_summary.mean(), 4)
)
print(
    "Trips with positive delta:",
    int((trip_summary > 1e-9).sum())
)
print(
    "Trips with negative delta:",
    int((trip_summary < -1e-9).sum())
)

print("\nSaved:", OUTPUT_PATH)
