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
    "data/processed/alt_vectorized_benchmark.parquet"
)

WEIGHT = "osm_travel_time_s"
EARTH_RADIUS_M = 6371008.8
COUNTS = [4, 8, 16, 32]
REPEATS = 5
SEED = 42
TOL = 1e-7

G = ox.load_graphml(GRAPH_PATH)

# Use minimum cost among parallel directed edges.
H = nx.DiGraph()
H.add_nodes_from(G.nodes)

for u, v, key, attrs in G.edges(
    keys=True, data=True
):
    w = float(attrs[WEIGHT])

    if not np.isfinite(w) or w <= 0:
        raise ValueError("Invalid edge weight")

    if (
        not H.has_edge(u, v)
        or w < H[u][v]["weight"]
    ):
        H.add_edge(u, v, weight=w)

nodes = list(H.nodes)
node_index = {
    node: i for i, node in enumerate(nodes)
}

adj = [[] for _ in nodes]

for u, v, attrs in H.edges(data=True):
    adj[node_index[u]].append(
        (node_index[v], attrs["weight"])
    )

coords = {
    node: (
        radians(float(G.nodes[node]["y"])),
        radians(float(G.nodes[node]["x"])),
    )
    for node in nodes
}


def geo_distance(u, v):
    lat1, lon1 = coords[u]
    lat2, lon2 = coords[v]

    a = (
        sin((lat2 - lat1) / 2) ** 2
        + cos(lat1) * cos(lat2)
        * sin((lon2 - lon1) / 2) ** 2
    )

    return 2 * EARTH_RADIUS_M * asin(
        min(1.0, sqrt(max(0.0, a)))
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
        key=lambda n: (
            coords[n][axis], n
        ),
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

# Dense arrays of landmark distances.
# Shape: (landmarks, graph nodes).
forward = np.full(
    (32, len(nodes)),
    np.inf,
    dtype=np.float64,
)

backward = np.full_like(forward, np.inf)

reverse_graph = H.reverse(copy=False)

t0 = perf_counter()

for j, landmark in enumerate(landmarks):
    distances = (
        nx.single_source_dijkstra_path_length(
            H, landmark, weight="weight"
        )
    )

    for node, distance in distances.items():
        forward[j, node_index[node]] = distance

    distances = (
        nx.single_source_dijkstra_path_length(
            reverse_graph,
            landmark,
            weight="weight",
        )
    )

    for node, distance in distances.items():
        backward[j, node_index[node]] = distance

preprocessing_s = perf_counter() - t0

# Dictionary-based representation of the same arrays.
forward_dict = [
    {
        i: float(value)
        for i, value in enumerate(row)
        if np.isfinite(value)
    }
    for row in forward
]

backward_dict = [
    {
        i: float(value)
        for i, value in enumerate(row)
        if np.isfinite(value)
    }
    for row in backward
]


def make_heuristic(target, n_landmarks, mode):
    if mode == "vectorized":
        ft = forward[:n_landmarks, target]
        bt = backward[:n_landmarks, target]

        def heuristic(node):
            fn = forward[:n_landmarks, node]
            bn = backward[:n_landmarks, node]

            # Missing directed distances must not
            # produce invalid inf-inf expressions.
            valid_forward = (
                np.isfinite(ft) & np.isfinite(fn)
            )
            valid_backward = (
                np.isfinite(bn) & np.isfinite(bt)
            )

            bound = 0.0

            if valid_forward.any():
                bound = max(
                    bound,
                    float(np.max(
                        ft[valid_forward]
                        - fn[valid_forward]
                    )),
                )

            if valid_backward.any():
                bound = max(
                    bound,
                    float(np.max(
                        bn[valid_backward]
                        - bt[valid_backward]
                    )),
                )

            return bound

        return heuristic

    if mode == "dictionary":
        def heuristic(node):
            bound = 0.0

            for j in range(n_landmarks):
                fd = forward_dict[j]
                bd = backward_dict[j]

                if node in fd and target in fd:
                    bound = max(
                        bound,
                        fd[target] - fd[node],
                    )

                if node in bd and target in bd:
                    bound = max(
                        bound,
                        bd[node] - bd[target],
                    )

            return bound

        return heuristic

    raise ValueError(mode)


def astar(source, target, n_landmarks, mode):
    heuristic = make_heuristic(
        target, n_landmarks, mode
    )

    cache = {}

    def h(node):
        if node not in cache:
            cache[node] = heuristic(node)
        return cache[node]

    best = {source: 0.0}
    sequence = count()

    queue = [
        (h(source), next(sequence), 0.0, source)
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

            if new_g < best.get(
                neighbor, np.inf
            ):
                best[neighbor] = new_g

                heappush(
                    queue,
                    (
                        new_g + h(neighbor),
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

# Independent reference costs.
references = [
    nx.shortest_path_length(
        H,
        nodes[source],
        nodes[target],
        weight="weight",
    )
    for source, target in pairs
]

variants = [
    (mode, n)
    for mode in ["dictionary", "vectorized"]
    for n in COUNTS
]

rng = np.random.default_rng(SEED)

# Warm up every implementation.
for mode, n in variants:
    for source, target in pairs[:5]:
        astar(source, target, n, mode)

records = []

for repeat in range(REPEATS):
    order = list(range(len(variants)))
    rng.shuffle(order)

    for variant_index in order:
        mode, n = variants[variant_index]

        start = perf_counter()
        total_expanded = 0

        for (source, target), reference in zip(
            pairs, references
        ):
            cost, expanded = astar(
                source, target, n, mode
            )

            if abs(cost - reference) > TOL:
                raise RuntimeError(
                    f"Optimality failure: "
                    f"{mode}/{n}"
                )

            total_expanded += expanded

        runtime_ms = (
            perf_counter() - start
        ) * 1000

        records.append({
            "repeat": repeat,
            "mode": mode,
            "landmarks": n,
            "od_pairs": len(pairs),
            "total_expanded": total_expanded,
            "total_runtime_ms": runtime_ms,
        })

    print(
        f"Completed repeat {repeat + 1}/{REPEATS}"
    )

result = pd.DataFrame(records)

OUTPUT_PATH.parent.mkdir(
    parents=True, exist_ok=True
)
result.to_parquet(OUTPUT_PATH, index=False)

summary = (
    result.groupby(
        ["mode", "landmarks"]
    )
    .agg(
        median_runtime_ms=(
            "total_runtime_ms", "median"
        ),
        mean_runtime_ms=(
            "total_runtime_ms", "mean"
        ),
        min_runtime_ms=(
            "total_runtime_ms", "min"
        ),
        max_runtime_ms=(
            "total_runtime_ms", "max"
        ),
        expanded=(
            "total_expanded", "first"
        ),
    )
    .reset_index()
)

print("\n=== PREPROCESSING ===")
print("Landmarks:", landmarks)
print(
    "32-landmark preprocessing (s):",
    round(preprocessing_s, 3)
)

print("\n=== REPEATED BENCHMARK ===")
print(
    summary.round(3).to_string(index=False)
)

print("\n=== VECTORIZATION SPEEDUP ===")

for n in COUNTS:
    d = summary[
        (summary["mode"] == "dictionary")
        & (summary["landmarks"] == n)
    ].iloc[0]

    v = summary[
        (summary["mode"] == "vectorized")
        & (summary["landmarks"] == n)
    ].iloc[0]

    print(
        f"{n:2d} landmarks: "
        f"{d['median_runtime_ms'] / v['median_runtime_ms']:.3f}x; "
        f"expanded dict={int(d['expanded'])}, "
        f"vector={int(v['expanded'])}"
    )

print("\n=== OPTIMALITY ===")
print(
    "All runs matched Dijkstra costs:",
    True
)

print("\nSaved:", OUTPUT_PATH)
