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
    "data/processed/adaptive_alt_k_benchmark.parquet"
)
LANDMARK_PATH = Path(
    "data/processed/adaptive_alt_landmarks.csv"
)

K_VALUES = [1, 2, 4, 6, 8, 12, 16, 32]
MAX_K = 32
REPEATS = 5
SEED = 42

G = ox.load_graphml(GRAPH_PATH)

H = nx.DiGraph()
H.add_nodes_from(G.nodes)

for u, v, key, attrs in G.edges(keys=True, data=True):
    weight = float(attrs["osm_travel_time_s"])

    if not np.isfinite(weight) or weight <= 0:
        raise ValueError("Invalid edge weight")

    if (
        not H.has_edge(u, v)
        or weight < H[u][v]["weight"]
    ):
        H.add_edge(u, v, weight=weight)

nodes = list(H.nodes)
idx = {node: i for i, node in enumerate(nodes)}
N = len(nodes)

adj = [[] for _ in range(N)]

for u, v, attrs in H.edges(data=True):
    adj[idx[u]].append(
        (idx[v], attrs["weight"])
    )

# Reproduce the previous geographic landmark set.
scc = max(
    nx.strongly_connected_components(H),
    key=len,
)

scc_nodes = list(scc)

lat = {
    n: float(G.nodes[n]["y"])
    for n in scc_nodes
}
lon = {
    n: float(G.nodes[n]["x"])
    for n in scc_nodes
}

landmarks = list(dict.fromkeys([
    min(scc_nodes, key=lambda n: lon[n]),
    max(scc_nodes, key=lambda n: lon[n]),
    min(scc_nodes, key=lambda n: lat[n]),
    max(scc_nodes, key=lambda n: lat[n]),
]))

xy = np.array([
    [lat[n], lon[n]]
    for n in scc_nodes
])

min_dist2 = np.full(len(scc_nodes), np.inf)

for landmark in landmarks:
    point = np.array([
        lat[landmark],
        lon[landmark],
    ])

    min_dist2 = np.minimum(
        min_dist2,
        np.sum((xy - point) ** 2, axis=1),
    )

while len(landmarks) < MAX_K:
    choice = int(np.argmax(min_dist2))
    landmark = scc_nodes[choice]

    if landmark in landmarks:
        min_dist2[choice] = -1.0
        continue

    landmarks.append(landmark)

    point = xy[choice]

    min_dist2 = np.minimum(
        min_dist2,
        np.sum((xy - point) ** 2, axis=1),
    )

LANDMARK_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

pd.DataFrame({
    "landmark_index": range(MAX_K),
    "node_id": landmarks,
}).to_csv(LANDMARK_PATH, index=False)

forward = np.full((MAX_K, N), np.inf)
backward = np.full((MAX_K, N), np.inf)

reverse_H = H.reverse(copy=False)

t0 = perf_counter()

for j, landmark in enumerate(landmarks):
    for node, distance in (
        nx.single_source_dijkstra_path_length(
            H, landmark, weight="weight"
        ).items()
    ):
        forward[j, idx[node]] = distance

    for node, distance in (
        nx.single_source_dijkstra_path_length(
            reverse_H, landmark, weight="weight"
        ).items()
    ):
        backward[j, idx[node]] = distance

preprocessing_s = perf_counter() - t0


def alt_value(node, target, chosen):
    best = 0.0

    for j in chosen:
        a = forward[j, target]
        b = forward[j, node]

        if np.isfinite(a) and np.isfinite(b):
            best = max(best, a - b)

        a = backward[j, node]
        b = backward[j, target]

        if np.isfinite(a) and np.isfinite(b):
            best = max(best, a - b)

    return best


def rank_landmarks(source, target):
    scores = [
        alt_value(source, target, (j,))
        for j in range(MAX_K)
    ]

    return tuple(sorted(
        range(MAX_K),
        key=lambda j: (-scores[j], j),
    ))


def astar(source, target, chosen):
    best = {source: 0.0}
    cache = {}
    serial = count()
    heap = []

    def heuristic(node):
        if node not in cache:
            cache[node] = alt_value(
                node, target, chosen
            )
        return cache[node]

    heappush(
        heap,
        (
            heuristic(source),
            0.0,
            next(serial),
            0.0,
            source,
        ),
    )

    expanded = 0

    while heap:
        _, _, _, g, node = heappop(heap)

        if g > best.get(node, np.inf):
            continue

        expanded += 1

        if node == target:
            return g, expanded, len(cache)

        for neighbor, weight in adj[node]:
            new_g = g + weight

            if new_g < best.get(neighbor, np.inf):
                best[neighbor] = new_g

                heappush(
                    heap,
                    (
                        new_g + heuristic(neighbor),
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
        idx[int(row.origin)],
        idx[int(row.destination)],
        int(row.origin),
        int(row.destination),
    )
    for row in od.itertuples(index=False)
]

references = [
    nx.shortest_path_length(
        H,
        origin,
        destination,
        weight="weight",
    )
    for _, _, origin, destination in pairs
]

print("=== PREPROCESSING ===")
print("Landmarks:", len(landmarks))
print("First four:", landmarks[:4])
print("Preprocessing (s):", round(preprocessing_s, 3))
print("OD pairs:", len(pairs))

# Warm-up without recording results.
for source, target, _, _ in pairs[:5]:
    ranked = rank_landmarks(source, target)
    for k in K_VALUES:
        astar(source, target, ranked[:k])

rng = np.random.default_rng(SEED)
records = []

for repeat in range(REPEATS):
    order = K_VALUES.copy()
    rng.shuffle(order)

    for k in order:
        for pair_id, (
            source, target, origin, destination
        ) in enumerate(pairs):

            start = perf_counter()
            ranked = rank_landmarks(source, target)
            chosen = ranked[:k]

            selection_ms = (
                perf_counter() - start
            ) * 1000

            start = perf_counter()

            cost, expanded, evaluated = astar(
                source, target, chosen
            )

            search_ms = (
                perf_counter() - start
            ) * 1000

            if abs(cost - references[pair_id]) > 1e-6:
                raise RuntimeError(
                    f"Optimality failure: k={k}, "
                    f"OD={pair_id}"
                )

            records.append({
                "repeat": repeat,
                "pair_id": pair_id,
                "k": k,
                "expanded": expanded,
                "evaluated": evaluated,
                "selection_ms": selection_ms,
                "search_ms": search_ms,
                "total_ms": selection_ms + search_ms,
            })

    print(
        f"Completed repeat {repeat + 1}/{REPEATS}"
    )

result = pd.DataFrame(records)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)
result.to_parquet(OUTPUT_PATH, index=False)

per_repeat = (
    result.groupby(["repeat", "k"])
    .agg(
        expanded=("expanded", "sum"),
        evaluated=("evaluated", "sum"),
        selection_ms=("selection_ms", "sum"),
        search_ms=("search_ms", "sum"),
        total_ms=("total_ms", "sum"),
    )
    .reset_index()
)

summary = (
    per_repeat.groupby("k")
    .agg(
        expanded=("expanded", "first"),
        evaluated=("evaluated", "first"),
        median_selection_ms=(
            "selection_ms", "median"
        ),
        median_search_ms=(
            "search_ms", "median"
        ),
        median_total_ms=(
            "total_ms", "median"
        ),
    )
    .reset_index()
)

baseline = int(
    summary.loc[
        summary["k"] == 4, "expanded"
    ].iloc[0]
)

summary["expansion_change_vs_k4_pct"] = (
    100 * (
        summary["expanded"] / baseline - 1
    )
)

print("\n=== ADAPTIVE K SUMMARY ===")
print(
    summary.round(3).to_string(index=False)
)

best_row = summary.loc[
    summary["median_total_ms"].idxmin()
]

print("\n=== FASTEST K ===")
print("k:", int(best_row["k"]))
print(
    "Median total time (ms):",
    round(best_row["median_total_ms"], 3),
)
print(
    "Total expansions:",
    int(best_row["expanded"]),
)

print("\n=== OPTIMALITY ===")
print("All routes matched Dijkstra: True")
print("\nSaved:", OUTPUT_PATH)
print("Saved:", LANDMARK_PATH)
