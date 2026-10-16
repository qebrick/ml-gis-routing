from pathlib import Path
from heapq import heappush, heappop
from itertools import count
from time import perf_counter

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

GRAPH_PATH = Path("data/graphs/porto_drive_conservative_refit.graphml")
OD_PATH = Path("data/processed/gps_od_route_comparison.parquet")
LANDMARK_PATH = Path("data/processed/adaptive_alt_landmarks.csv")
OUTPUT_PATH = Path("data/processed/adaptive_alt_precomputed_benchmark.parquet")

K_VALUES = [1, 2, 4, 8, 16, 32]
REPEATS = 5
SEED = 42

G = ox.load_graphml(GRAPH_PATH)
H = nx.DiGraph()
H.add_nodes_from(G.nodes)

for u, v, _, attrs in G.edges(keys=True, data=True):
    w = float(attrs["osm_travel_time_s"])
    if not np.isfinite(w) or w <= 0:
        raise ValueError("Invalid weight")
    if not H.has_edge(u, v) or w < H[u][v]["weight"]:
        H.add_edge(u, v, weight=w)

nodes = list(H.nodes)
idx = {n: i for i, n in enumerate(nodes)}
N = len(nodes)

adj = [[] for _ in range(N)]
for u, v, attrs in H.edges(data=True):
    adj[idx[u]].append((idx[v], attrs["weight"]))

landmark_ids = pd.read_csv(LANDMARK_PATH)["node_id"].tolist()

# Match CSV identifiers to the actual GraphML node types.
node_lookup = {str(node): node for node in H.nodes}
landmarks = [node_lookup[str(node)] for node in landmark_ids]

K = len(landmarks)
assert K == 32
assert len(set(landmarks)) == K
assert all(node in idx for node in landmarks)

forward = np.full((K, N), np.inf)
backward = np.full((K, N), np.inf)
reverse_H = H.reverse(copy=False)

t0 = perf_counter()
for j, landmark in enumerate(landmarks):
    for n, d in nx.single_source_dijkstra_path_length(
        H, landmark, weight="weight"
    ).items():
        forward[j, idx[n]] = d

    for n, d in nx.single_source_dijkstra_path_length(
        reverse_H, landmark, weight="weight"
    ).items():
        backward[j, idx[n]] = d

preprocessing_s = perf_counter() - t0


def rank_landmarks(source, target):
    a = np.zeros(K)
    b = np.zeros(K)

    valid = (
        np.isfinite(forward[:, target])
        & np.isfinite(forward[:, source])
    )
    np.subtract(
        forward[:, target],
        forward[:, source],
        out=a,
        where=valid,
    )

    valid = (
        np.isfinite(backward[:, source])
        & np.isfinite(backward[:, target])
    )
    np.subtract(
        backward[:, source],
        backward[:, target],
        out=b,
        where=valid,
    )

    scores = np.maximum(np.maximum(a, b), 0.0)
    return np.argsort(-scores, kind="stable")


def build_heuristic(target, chosen):
    chosen = np.asarray(chosen, dtype=int)
    m = len(chosen)

    first = np.zeros((m, N))
    second = np.zeros((m, N))

    f = forward[chosen]
    b = backward[chosen]

    valid_first = (
        np.isfinite(f[:, target])[:, None]
        & np.isfinite(f)
    )
    np.subtract(
        f[:, target, None],
        f,
        out=first,
        where=valid_first,
    )

    valid_second = (
        np.isfinite(b)
        & np.isfinite(b[:, target])[:, None]
    )
    np.subtract(
        b,
        b[:, target, None],
        out=second,
        where=valid_second,
    )

    h = np.maximum(
        np.maximum(first.max(axis=0), second.max(axis=0)),
        0.0,
    )
    h[target] = 0.0
    return h


def astar(source, target, h):
    best = {source: 0.0}
    serial = count()
    heap = [(float(h[source]), 0.0, next(serial), 0.0, source)]
    expanded = 0

    while heap:
        _, _, _, g, node = heappop(heap)

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
                    heap,
                    (
                        new_g + float(h[neighbor]),
                        -new_g,
                        next(serial),
                        new_g,
                        neighbor,
                    ),
                )

    raise RuntimeError("No path")


od = pd.read_parquet(OD_PATH)
pairs = [
    (
        idx[node_lookup[str(row.origin)]],
        idx[node_lookup[str(row.destination)]],
        node_lookup[str(row.origin)],
        node_lookup[str(row.destination)],
    )
    for row in od.itertuples(index=False)
]

references = [
    nx.shortest_path_length(H, u, v, weight="weight")
    for _, _, u, v in pairs
]

print("=== PREPROCESSING ===")
print("Graph nodes:", N)
print("Landmarks:", K)
print("First four:", landmarks[:4])
print("Preprocessing (s):", round(preprocessing_s, 3))
print("OD pairs:", len(pairs))

# Warm up each k without recording timings.
for source, target, _, _ in pairs[:3]:
    ranked = rank_landmarks(source, target)
    for k in K_VALUES:
        h = build_heuristic(target, ranked[:k])
        astar(source, target, h)

rng = np.random.default_rng(SEED)
records = []

for repeat in range(REPEATS):
    order = K_VALUES.copy()
    rng.shuffle(order)

    for k in order:
        for pair_id, (source, target, _, _) in enumerate(pairs):
            t0 = perf_counter()
            ranked = rank_landmarks(source, target)
            selection_ms = (perf_counter() - t0) * 1000

            t0 = perf_counter()
            h = build_heuristic(target, ranked[:k])
            preparation_ms = (perf_counter() - t0) * 1000

            t0 = perf_counter()
            cost, expanded = astar(source, target, h)
            search_ms = (perf_counter() - t0) * 1000

            if abs(cost - references[pair_id]) > 1e-6:
                raise RuntimeError(
                    f"Optimality failure: k={k}, OD={pair_id}"
                )

            records.append({
                "repeat": repeat,
                "pair_id": pair_id,
                "k": k,
                "expanded": expanded,
                "selection_ms": selection_ms,
                "preparation_ms": preparation_ms,
                "search_ms": search_ms,
                "total_ms": selection_ms + preparation_ms + search_ms,
            })

    print(f"Completed repeat {repeat + 1}/{REPEATS}")

result = pd.DataFrame(records)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result.to_parquet(OUTPUT_PATH, index=False)

per_repeat = (
    result.groupby(["repeat", "k"])
    .agg(
        expanded=("expanded", "sum"),
        selection_ms=("selection_ms", "sum"),
        preparation_ms=("preparation_ms", "sum"),
        search_ms=("search_ms", "sum"),
        total_ms=("total_ms", "sum"),
    )
    .reset_index()
)

summary = (
    per_repeat.groupby("k")
    .agg(
        expanded=("expanded", "first"),
        median_selection_ms=("selection_ms", "median"),
        median_preparation_ms=("preparation_ms", "median"),
        median_search_ms=("search_ms", "median"),
        median_total_ms=("total_ms", "median"),
    )
    .reset_index()
)

print("\n=== PRECOMPUTED ALT SUMMARY ===")
print(summary.round(3).to_string(index=False))

best = summary.loc[summary["median_total_ms"].idxmin()]
print("\n=== FASTEST K ===")
print("k:", int(best["k"]))
print("Median total time (ms):", round(best["median_total_ms"], 3))
print("Total expansions:", int(best["expanded"]))

print("\n=== OPTIMALITY ===")
print("All routes matched Dijkstra: True")
print("\nSaved:", OUTPUT_PATH)
