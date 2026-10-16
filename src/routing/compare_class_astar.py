from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

GRAPH_PATH = Path(
    "data/graphs/porto_drive_class_weighted.graphml"
)
OUTPUT_PATH = Path(
    "data/processed/class_astar_comparison.parquet"
)

G = ox.load_graphml(GRAPH_PATH)

RANDOM_SEED = 42
N_PAIRS = 100

rng = np.random.default_rng(RANDOM_SEED)

nodes = list(G.nodes)


def edge_cost(u, v, weight_name):
    edges = G[u][v]

    return min(
        float(attrs[weight_name])
        for attrs in edges.values()
    )


def astar_weight(weight_name):
    def weight(u, v, edge_dict):
        # NetworkX MultiDiGraph passes a dict
        # of parallel-edge attribute dictionaries.
        return min(
            float(attrs[weight_name])
            for attrs in edge_dict.values()
        )

    return weight


def route_metrics(path, optimization_weight):
    distance = 0.0
    osm_time = 0.0
    class_time = 0.0
    residential_length = 0.0

    for u, v in zip(path[:-1], path[1:]):
        edges = G[u][v]

        # Select the same parallel edge that
        # minimizes the route objective.
        attrs = min(
            edges.values(),
            key=lambda a: float(
                a[optimization_weight]
            ),
        )

        length = float(attrs["length"])

        distance += length
        osm_time += float(
            attrs["osm_travel_time_s"]
        )
        class_time += float(
            attrs["class_travel_time_s"]
        )

        if (
            attrs.get("class_road_group")
            == "residential"
        ):
            residential_length += length

    return {
        "distance_m": distance,
        "osm_time_s": osm_time,
        "class_time_s": class_time,
        "residential_share": (
            residential_length / distance
            if distance > 0 else 0.0
        ),
    }


# Use nodes from the largest strongly connected
# component so every sampled OD is reachable.
largest_scc = max(
    nx.strongly_connected_components(G),
    key=len,
)

candidate_nodes = list(largest_scc)

print("=== GRAPH ===")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Largest SCC nodes:", len(candidate_nodes))

records = []

for pair_id in range(N_PAIRS):
    origin, destination = rng.choice(
        candidate_nodes,
        size=2,
        replace=False,
    )

    origin = int(origin)
    destination = int(destination)

    osm_path = nx.astar_path(
        G,
        origin,
        destination,
        heuristic=lambda u, v: 0.0,
        weight=astar_weight("osm_travel_time_s"),
    )

    class_path = nx.astar_path(
        G,
        origin,
        destination,
        heuristic=lambda u, v: 0.0,
        weight=astar_weight("class_travel_time_s"),
    )

    osm = route_metrics(
        osm_path,
        "osm_travel_time_s",
    )

    calibrated = route_metrics(
        class_path,
        "class_travel_time_s",
    )

    records.append({
        "pair_id": pair_id,
        "origin": origin,
        "destination": destination,
        "same_node_path": osm_path == class_path,
        "osm_distance_m": osm["distance_m"],
        "class_distance_m": calibrated["distance_m"],
        "osm_route_osm_time_s": osm["osm_time_s"],
        "class_route_osm_time_s": calibrated["osm_time_s"],
        "osm_route_class_time_s": osm["class_time_s"],
        "class_route_class_time_s": calibrated["class_time_s"],
        "osm_residential_share": osm["residential_share"],
        "class_residential_share": calibrated["residential_share"],
    })

df = pd.DataFrame(records)

df["distance_change_pct"] = (
    100
    * (
        df["class_distance_m"]
        / df["osm_distance_m"]
        - 1
    )
)

df["osm_time_change_pct"] = (
    100
    * (
        df["class_route_osm_time_s"]
        / df["osm_route_osm_time_s"]
        - 1
    )
)

df["class_objective_gain_pct"] = (
    100
    * (
        df["osm_route_class_time_s"]
        - df["class_route_class_time_s"]
    )
    / df["osm_route_class_time_s"]
)

df["residential_share_change_pp"] = (
    100
    * (
        df["class_residential_share"]
        - df["osm_residential_share"]
    )
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_parquet(OUTPUT_PATH, index=False)

print("\n=== ROUTE CHANGES ===")
print("OD pairs:", len(df))
print(
    "Different node paths:",
    int((~df["same_node_path"]).sum())
)
print(
    "Different node paths (%):",
    round(
        100 * (~df["same_node_path"]).mean(),
        2,
    )
)

print("\n=== ROUTE METRICS ===")

columns = [
    "distance_change_pct",
    "osm_time_change_pct",
    "class_objective_gain_pct",
    "residential_share_change_pp",
]

print(
    df[columns]
    .describe(percentiles=[0.25, 0.5, 0.75])
    .round(3)
    .to_string()
)

print("\n=== CHANGED ROUTES ONLY ===")

changed = df.loc[
    ~df["same_node_path"]
]

if not changed.empty:
    print(
        changed[columns]
        .describe()
        .round(3)
        .to_string()
    )

print("\n=== OBJECTIVE CONSISTENCY ===")

violations = (
    df["class_route_class_time_s"]
    > df["osm_route_class_time_s"] + 1e-6
).sum()

print(
    "Class objective violations:",
    int(violations)
)

print("\nSaved:", OUTPUT_PATH)
