from pathlib import Path
from heapq import heappush, heappop
from itertools import count
from time import perf_counter

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
    "data/processed/alt_target_precompute_benchmark.parquet"
)

COUNTS = [4, 16, 32]
REPEATS = 5
SEED = 42
TOL = 1e-7

G = ox.load_graphml(GRAPH_PATH)

H = nx.DiGraph()
H.add_nodes_from(G.nodes)

for u, v, key, attrs in G.edges(keys=True, data=True):
    w = float(attrs["osm_travel_time_s"])

    if not np.isfinite(w) or w <= 0:
        raise ValueError("Invalid OSM edge weight")

    if not H.has_edge(u, v) or w < H[u][v]["weight"]:
        H.add_edge(u, v, weight=w)

nodes = list(H.nodes)
node_index = {node: i for i, node in enumerate(nodes)}
N = len(nodes)

adj = [[] for _ in range(N)]

for u, v, attrs in H.edges(data=True):
    adj[node_index[u]].append(
        (node_index[v], attrs["weight"])
    )

# Reproduce landmark selection from the preceding experiment.
from math import radians, sin, cos, asin, sqrt

R = 6371008.8

coords = {
    node: (
        radians(float(G.nodes[node]["y"])),
        radians(float(G.nodes[node]["x"])),
    )
    for node in nodes
}


def geo_distance(u, v):
    a1, o1 = coords[u]
    a2, o2 = coords[v]

    z = (
        sin((a2 - a1) / 2) ** 2
        + cos(a1) * cos(a2)
        * sin((o2 - o1) / 2) ** 2
    )

    return 2 * R * asin(
        min(1.0, sqrt(max(0.0, z)))
    )


scc = sorted(
    max(
        nx.strongly_connected_components(H),
        key=len,
    )
)

landmarks = []

for axis, reverse in [
    (0, False),
    (0, True),
    (1, False),
    (1, True),
]:
    candidate = sorted(
        scc,
        key=lambda n: (coords[n][axis], n),
        reverse=reverse,
    )[0]

    if candidate not in landmarks:
        landmarks.append(candidate)

min_geo = {
    node: min(
        geo_distance(node, l)
        for l in landmarks
    )
    for node in scc
}

while len(landmarks) < 32:
    candidate = max(
        (n for n in scc if n not in landmarks),
        key=lambda n: (min_geo[n], -n),
    )

    landmarks.append(candidate)

    for node in scc:
        min_geo[node] = min(
            min_geo[node],
            geo_distance(node, candidate),
        )

forward = np.full((32, N), np.inf)
backward = np.full((32, N), np.inf)

reverse_graph = H.reverse(copy=False)

for j, landmark in enumerate(landmarks):
    for node, value in nx.single_source_dijkstra_path_length(
        H, landmark, weight="weight"
    ).items():
        forward[j, node_index[node]] = value

    for node, value in nx.single_source_dijkstra_path_length(
        reverse_graph, landmark, weight="weight"
    ).items():
        backward[j, node_index[node]] = value


def target_heuristic(target, n_landmarks):
    """
    Compute admissible ALT lower bounds for every graph node.

    Missing directed distances contribute no bound.
    """
    f = forward[:n_landmarks]
    b = backward[:n_landmarks]

    ft = f[:, target]
    bt = b[:, target]

    result = np.zeros(N, dtype=np.float64)

    # Each landmark contributes up to two lower bounds.
    # Processing one row at a time limits temporary memory.
    for j in range(n_landmarks):
        if np.isfinite(ft[j]):
            valid = np.isfinite(f[j])
            candidate = np.zeros(N)
            candidate[valid] = ft[j] - f[j, valid]
            np.maximum(result, candidate, out=result)

        if np.isfinite(bt[j]):
            valid = np.isfinite(b[j])
            candidate = np.zeros(N)
            candidate[valid] = b[j, valid] - bt[j]
            np.maximum(result, candidate, out=result)

    result[target] = 0.0
    return result


def astar(source, target, heuristic_values):
    best = {source: 0.0}
    sequence = count()

    queue = [
        (
            float(heuristic_values[source]),
            next(sequence),
            0.0,
            source,
        )
    ]

    expanded = 0

    while queue:
        _, _, g, node = heappop(queue)

        if g > best.get(node, np.inf):
            continue

        expanded += 1

        if node == target:
            return g, expanded

        for neighbor, weight in adj[node]:
            new_g = g + weight

            if new_g < best.get(neighbor, np.inf):
                best[neighbor] = new_g

                heappush(
                    queue,
                    (
                        new_g + float(heuristic_values[neighbor]),
                        next(sequence),
                        new_g,
                        neighbor,
                    ),
                )

    raise RuntimeError("No path")


od = pd.read_parquet(OD_PATH)

pairs = [
    (
        node_index[int(row.origin)],
        node_index[int(row.destination)],
    )
    for row in od.itertuples(index=False)
]

references = [
    nx.shortest_path_length(
        H, nodes[source], nodes[target],
        weight="weight",
    )
    for source, target in pairs
]

variants = [
    (n, mode)
    for n in COUNTS
    for mode in ["per_query", "cached_target"]
]

rng = np.random.default_rng(SEED)

# Warm-up and correctness checks.
for n in COUNTS:
    for (source, target), reference in zip(
        pairs[:5], references[:5]
    ):
        h = target_heuristic(target, n)
        cost, _ = astar(source, target, h)

        if abs(cost - reference) > TOL:
            raise RuntimeError("Warm-up optimality failure")

records = []

for repeat in range(REPEATS):
    order = list(range(len(variants)))
    rng.shuffle(order)

    for variant_index in order:
        n, mode = variants[variant_index]

        cache = {}
        preparation_ms = 0.0
        search_ms = 0.0
        total_expanded = 0

        for (source, target), reference in zip(
            pairs, references
        ):
            t0 = perf_counter()

            if mode == "per_query":
                h = target_heuristic(target, n)
            else:
                if target not in cache:
                    cache[target] = target_heuristic(
                        target, n
                    )
                h = cache[target]

            preparation_ms += (
                perf_counter() - t0
            ) * 1000

            t0 = perf_counter()
            cost, expanded = astar(source, target, h)
            search_ms += (
                perf_counter() - t0
            ) * 1000

            if abs(cost - reference) > TOL:
                raise RuntimeError(
                    f"Optimality failure: "
                    f"{mode}, landmarks={n}"
                )

            total_expanded += expanded

        records.append({
            "repeat": repeat,
            "landmarks": n,
            "mode": mode,
            "preparation_ms": preparation_ms,
            "search_ms": search_ms,
            "total_ms": preparation_ms + search_ms,
            "expanded": total_expanded,
            "cached_targets": len(cache),
        })

    print(f"Completed repeat {repeat + 1}/{REPEATS}")

result = pd.DataFrame(records)

OUTPUT_PATH.parent.mkdir(
    parents=True, exist_ok=True
)
result.to_parquet(OUTPUT_PATH, index=False)

summary = (
    result.groupby(["landmarks", "mode"])
    .agg(
        median_preparation_ms=(
            "preparation_ms", "median"
        ),
        median_search_ms=(
            "search_ms", "median"
        ),
        median_total_ms=(
            "total_ms", "median"
        ),
        expanded=("expanded", "first"),
        cached_targets=("cached_targets", "first"),
    )
    .reset_index()
)

print("\n=== TARGET PRECOMPUTE BENCHMARK ===")
print(
    summary.round(3).to_string(index=False)
)

print("\n=== COVERAGE ===")
print("OD pairs:", len(pairs))
print(
    "Unique targets:",
    len({target for _, target in pairs})
)
print("Graph nodes:", N)

print("\n=== OPTIMALITY ===")
print("All runs matched Dijkstra costs: True")

print("\nSaved:", OUTPUT_PATH)
