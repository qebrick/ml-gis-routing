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
    "data/processed/alt_astar_benchmark.parquet"
)

WEIGHT = "osm_travel_time_s"
EARTH_RADIUS_M = 6371008.8
TOL = 1e-7

G = ox.load_graphml(GRAPH_PATH)
od = pd.read_parquet(OD_PATH)

# Build a simple directed graph using the minimum
# OSM time among parallel edges.
H = nx.DiGraph()
H.add_nodes_from(G.nodes)

for u, v, key, attrs in G.edges(keys=True, data=True):
    w = float(attrs[WEIGHT])

    if not np.isfinite(w) or w <= 0:
        raise ValueError(
            f"Invalid weight: {(u, v, key)}"
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


alpha = min(
    attrs["weight"] / geo_distance(u, v)
    for u, v, attrs in H.edges(data=True)
    if geo_distance(u, v) > 0
)

alpha = float(np.nextafter(alpha, 0.0))

# Restrict landmark selection to the largest
# strongly connected component, where both
# directed distances are finite.
largest_scc = max(
    nx.strongly_connected_components(H),
    key=len,
)

scc_nodes = list(largest_scc)

# Geographic extremes provide a simple,
# reproducible landmark-selection strategy.
landmarks = []

for axis, reverse in [
    (0, False),
    (0, True),
    (1, False),
    (1, True),
]:
    node = sorted(
        scc_nodes,
        key=lambda n: coords[n][axis],
        reverse=reverse,
    )[0]

    if node not in landmarks:
        landmarks.append(node)

print("=== LANDMARK PREPROCESSING ===")
print("Graph nodes:", H.number_of_nodes())
print("Graph edges:", H.number_of_edges())
print("Largest SCC:", len(largest_scc))
print("Landmarks:", landmarks)

forward = {}
backward = {}

reversed_graph = H.reverse(copy=False)

t0 = perf_counter()

for landmark in landmarks:
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

preprocessing_s = perf_counter() - t0

print(
    "Preprocessing time (s):",
    round(preprocessing_s, 3),
)


def alt_heuristic(node, target):
    bound = 0.0

    for landmark in landmarks:
        from_l = forward[landmark]
        to_l = backward[landmark]

        # d(L,t) - d(L,n)
        if node in from_l and target in from_l:
            bound = max(
                bound,
                from_l[target] - from_l[node],
            )

        # d(n,L) - d(t,L)
        if node in to_l and target in to_l:
            bound = max(
                bound,
                to_l[node] - to_l[target],
            )

    return max(0.0, bound)


def shortest_path(source, target, heuristic):
    best = {source: 0.0}
    parent = {}

    queue = []
    sequence = count()

    cache = {}

    def h(node):
        if node not in cache:
            if heuristic == "dijkstra":
                value = 0.0
            elif heuristic == "geo":
                value = alpha * geo_distance(
                    node, target
                )
            elif heuristic == "alt":
                value = alt_heuristic(
                    node, target
                )
            elif heuristic == "combined":
                value = max(
                    alpha * geo_distance(node, target),
                    alt_heuristic(node, target),
                )
            else:
                raise ValueError(heuristic)

            cache[node] = value

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

            return g, expanded, path

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


rows = []

for pair in od.itertuples(index=False):
    source = int(pair.origin)
    target = int(pair.destination)

    results = {}

    for heuristic in [
        "dijkstra",
        "geo",
        "alt",
        "combined",
    ]:
        start = perf_counter()

        cost, expanded, path = shortest_path(
            source, target, heuristic
        )

        elapsed_ms = (
            perf_counter() - start
        ) * 1000

        results[heuristic] = {
            "cost": cost,
            "expanded": expanded,
            "path": path,
            "elapsed_ms": elapsed_ms,
        }

    baseline = results["dijkstra"]

    for heuristic in ["geo", "alt", "combined"]:
        diff = (
            results[heuristic]["cost"]
            - baseline["cost"]
        )

        if abs(diff) > TOL:
            raise RuntimeError(
                f"Optimality failure: "
                f"{heuristic}, {source}->{target}, "
                f"diff={diff}"
            )

    row = {
        "trip_id": str(pair.trip_id),
        "origin": source,
        "destination": target,
    }

    for heuristic, values in results.items():
        row[f"{heuristic}_cost_s"] = values["cost"]
        row[f"{heuristic}_expanded"] = (
            values["expanded"]
        )
        row[f"{heuristic}_ms"] = (
            values["elapsed_ms"]
        )
        row[f"{heuristic}_same_path"] = (
            values["path"] == baseline["path"]
        )

    rows.append(row)


result = pd.DataFrame(rows)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)
result.to_parquet(OUTPUT_PATH, index=False)

print("\n=== OPTIMALITY ===")

for heuristic in ["geo", "alt", "combined"]:
    print(
        heuristic,
        "same paths:",
        int(result[f"{heuristic}_same_path"].sum()),
        "/",
        len(result),
    )

print("\n=== SEARCH EFFICIENCY ===")

for heuristic in [
    "dijkstra",
    "geo",
    "alt",
    "combined",
]:
    print(
        f"{heuristic:10s}",
        "mean expanded:",
        round(
            result[f"{heuristic}_expanded"].mean(),
            2,
        ),
        "total expanded:",
        int(
            result[f"{heuristic}_expanded"].sum()
        ),
        "total runtime (ms):",
        round(
            result[f"{heuristic}_ms"].sum(),
            3,
        ),
    )

print("\n=== REDUCTION VS DIJKSTRA ===")

baseline_expanded = (
    result["dijkstra_expanded"].sum()
)

baseline_runtime = (
    result["dijkstra_ms"].sum()
)

for heuristic in ["geo", "alt", "combined"]:
    reduction = (
        100
        * (
            1
            - result[f"{heuristic}_expanded"].sum()
            / baseline_expanded
        )
    )

    runtime_reduction = (
        100
        * (
            1
            - result[f"{heuristic}_ms"].sum()
            / baseline_runtime
        )
    )

    print(
        f"{heuristic:10s}",
        "expansion reduction (%):",
        round(reduction, 3),
        "runtime reduction (%):",
        round(runtime_reduction, 3),
    )

print("\nSaved:", OUTPUT_PATH)
