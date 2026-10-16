from pathlib import Path
from heapq import heappush, heappop
from itertools import count
from time import perf_counter
from math import radians, sin, cos, asin, sqrt

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
OUTPUT_PATH = Path(
    "data/processed/alt_landmark_scaling.parquet"
)

WEIGHT = "osm_travel_time_s"
EARTH_RADIUS_M = 6371008.8
LANDMARK_COUNTS = [4, 8, 16, 32]
TOL = 1e-7

G = ox.load_graphml(GRAPH_PATH)
od = pd.read_parquet(OD_PATH)

H = nx.DiGraph()
H.add_nodes_from(G.nodes)

for u, v, key, attrs in G.edges(keys=True, data=True):
    w = float(attrs[WEIGHT])

    if not np.isfinite(w) or w <= 0:
        raise ValueError(
            f"Invalid edge weight: {(u, v, key)}"
        )

    if (
        not H.has_edge(u, v)
        or w < H[u][v]["weight"]
    ):
        H.add_edge(u, v, weight=w)

coords = {
    node: (
        radians(float(attrs["y"])),
        radians(float(attrs["x"])),
    )
    for node, attrs in G.nodes(data=True)
}


def geo_distance(u, v):
    lat1, lon1 = coords[u]
    lat2, lon2 = coords[v]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    return 2 * EARTH_RADIUS_M * asin(
        min(1.0, sqrt(max(0.0, a)))
    )


largest_scc = max(
    nx.strongly_connected_components(H),
    key=len,
)

scc_nodes = sorted(largest_scc)

# Preserve the original four extreme landmarks.
landmarks = []

for axis, reverse in [
    (0, False),
    (0, True),
    (1, False),
    (1, True),
]:
    candidate = sorted(
        scc_nodes,
        key=lambda n: (
            coords[n][axis],
            n,
        ),
        reverse=reverse,
    )[0]

    if candidate not in landmarks:
        landmarks.append(candidate)

# Farthest-point sampling based on minimum
# geographic distance to existing landmarks.
min_distance = {
    node: min(
        geo_distance(node, landmark)
        for landmark in landmarks
    )
    for node in scc_nodes
}

while len(landmarks) < max(LANDMARK_COUNTS):
    candidate = max(
        (
            node for node in scc_nodes
            if node not in landmarks
        ),
        key=lambda node: (
            min_distance[node],
            -node,
        ),
    )

    landmarks.append(candidate)

    for node in scc_nodes:
        min_distance[node] = min(
            min_distance[node],
            geo_distance(node, candidate),
        )

print("=== LANDMARK SELECTION ===")
print("Graph nodes:", H.number_of_nodes())
print("Graph edges:", H.number_of_edges())
print("Largest SCC:", len(largest_scc))
print("Landmarks selected:", len(landmarks))
print("First four:", landmarks[:4])

reversed_graph = H.reverse(copy=False)

forward = {}
backward = {}
preprocessing_times = {}

start_preprocessing = perf_counter()

for index, landmark in enumerate(landmarks, 1):
    forward[landmark] = (
        nx.single_source_dijkstra_path_length(
            H,
            landmark,
            weight="weight",
        )
    )

    backward[landmark] = (
        nx.single_source_dijkstra_path_length(
            reversed_graph,
            landmark,
            weight="weight",
        )
    )

    if index in LANDMARK_COUNTS:
        preprocessing_times[index] = (
            perf_counter() - start_preprocessing
        )


def alt_h(node, target, count_landmarks):
    bound = 0.0

    for landmark in landmarks[:count_landmarks]:
        from_l = forward[landmark]
        to_l = backward[landmark]

        if node in from_l and target in from_l:
            bound = max(
                bound,
                from_l[target] - from_l[node],
            )

        if node in to_l and target in to_l:
            bound = max(
                bound,
                to_l[node] - to_l[target],
            )

    return max(0.0, bound)


def search(source, target, count_landmarks):
    best = {source: 0.0}
    parent = {}
    cache = {}

    sequence = count()
    queue = []

    def h(node):
        if node not in cache:
            cache[node] = alt_h(
                node,
                target,
                count_landmarks,
            )
        return cache[node]

    heappush(
        queue,
        (h(source), next(sequence), 0.0, source),
    )

    expanded = 0

    while queue:
        _, _, g, node = heappop(queue)

        if g > best.get(node, float("inf")):
            continue

        expanded += 1

        if node == target:
            path = [target]

            while path[-1] != source:
                path.append(parent[path[-1]])

            path.reverse()

            return {
                "cost": g,
                "expanded": expanded,
                "path": path,
            }

        for neighbor, attrs in H[node].items():
            new_g = g + attrs["weight"]

            if new_g < best.get(
                neighbor, float("inf")
            ):
                best[neighbor] = new_g
                parent[neighbor] = node

                heappush(
                    queue,
                    (
                        new_g + h(neighbor),
                        next(sequence),
                        new_g,
                        neighbor,
                    ),
                )

    raise RuntimeError(
        f"No path: {source}->{target}"
    )


records = []

for pair in od.itertuples(index=False):
    source = int(pair.origin)
    target = int(pair.destination)

    reference = nx.shortest_path_length(
        H,
        source,
        target,
        weight="weight",
    )

    for count_landmarks in LANDMARK_COUNTS:
        start = perf_counter()

        result = search(
            source,
            target,
            count_landmarks,
        )

        elapsed_ms = (
            perf_counter() - start
        ) * 1000

        if abs(result["cost"] - reference) > TOL:
            raise RuntimeError(
                f"Optimality failure: "
                f"{source}->{target}, "
                f"landmarks={count_landmarks}"
            )

        heuristic_at_source = alt_h(
            source,
            target,
            count_landmarks,
        )

        records.append({
            "trip_id": str(pair.trip_id),
            "origin": source,
            "destination": target,
            "landmarks": count_landmarks,
            "optimal_cost_s": reference,
            "heuristic_at_source_s": (
                heuristic_at_source
            ),
            "source_coverage": (
                heuristic_at_source / reference
                if reference > 0
                else 1.0
            ),
            "expanded": result["expanded"],
            "runtime_ms": elapsed_ms,
        })


df = pd.DataFrame(records)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)
df.to_parquet(OUTPUT_PATH, index=False)

summary = (
    df.groupby("landmarks")
    .agg(
        od_pairs=("expanded", "size"),
        mean_expanded=("expanded", "mean"),
        total_expanded=("expanded", "sum"),
        total_runtime_ms=("runtime_ms", "sum"),
        mean_source_coverage=(
            "source_coverage",
            "mean",
        ),
        median_source_coverage=(
            "source_coverage",
            "median",
        ),
    )
    .reset_index()
)

summary["preprocessing_s"] = (
    summary["landmarks"]
    .map(preprocessing_times)
)

baseline_expanded = 150410

summary["reduction_vs_dijkstra_pct"] = (
    100
    * (
        1
        - summary["total_expanded"]
        / baseline_expanded
    )
)

print("\n=== LANDMARK SCALING RESULTS ===")

print(
    summary.round(4).to_string(index=False)
)

print("\n=== OPTIMALITY ===")
print("Checked routes:", len(df))
print(
    "All route costs equal Dijkstra:",
    True,
)

print("\nSaved:", OUTPUT_PATH)
