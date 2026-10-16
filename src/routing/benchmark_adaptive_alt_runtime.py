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
    "data/processed/adaptive_alt_runtime.parquet"
)

SEED = 42
REPEATS = 10
K = 32
STRATEGIES = ["fixed_4", "source_top_4", "all_32"]

G = ox.load_graphml(GRAPH_PATH)

H = nx.DiGraph()
H.add_nodes_from(G.nodes)

for u, v, key, attrs in G.edges(keys=True, data=True):
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

# Same landmark-selection procedure as the
# preceding diagnostic experiment.
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
    [lat[n], lon[n]] for n in scc_nodes
])

min_dist2 = np.full(len(scc_nodes), np.inf)

for landmark in landmarks:
    point = np.array([lat[landmark], lon[landmark]])
    min_dist2 = np.minimum(
        min_dist2,
        np.sum((xy - point) ** 2, axis=1),
    )

while len(landmarks) < K:
    choice = int(np.argmax(min_dist2))
    landmark = scc_nodes[choice]

    if landmark in landmarks:
        min_dist2[choice] = -1
        continue

    landmarks.append(landmark)

    point = xy[choice]
    min_dist2 = np.minimum(
        min_dist2,
        np.sum((xy - point) ** 2, axis=1),
    )

forward = np.full((K, N), np.inf)
backward = np.full((K, N), np.inf)

start = perf_counter()

for j, landmark in enumerate(landmarks):
    distances = nx.single_source_dijkstra_path_length(
        H, landmark, weight="weight"
    )

    for n, d in distances.items():
        forward[j, idx[n]] = d

    distances = nx.single_source_dijkstra_path_length(
        H.reverse(copy=False),
        landmark,
        weight="weight",
    )

    for n, d in distances.items():
        backward[j, idx[n]] = d

preprocessing_s = perf_counter() - start


def alt_value(node, target, chosen):
    """
    Directed ALT. Only finite distance pairs
    contribute to the lower bound.
    """
    best = 0.0

    for j in chosen:
        lt = forward[j, target]
        ln = forward[j, node]

        if np.isfinite(lt) and np.isfinite(ln):
            best = max(best, lt - ln)

        nl = backward[j, node]
        tl = backward[j, target]

        if np.isfinite(nl) and np.isfinite(tl):
            best = max(best, nl - tl)

    return best


def select_source_top4(source, target):
    scores = [
        alt_value(source, target, [j])
        for j in range(K)
    ]

    # Stable descending order, deterministic
    # tie-breaking by landmark index.
    return tuple(sorted(
        range(K),
        key=lambda j: (-scores[j], j),
    )[:4])


def astar(source, target, chosen):
    best = {source: 0.0}
    heap = []
    serial = count()
    cache = {}

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
        H, origin, destination, weight="weight"
    )
    for _, _, origin, destination in pairs
]

print("=== PREPROCESSING ===")
print("Landmarks:", K)
print("First four:", landmarks[:4])
print("Preprocessing (s):", round(preprocessing_s, 3))
print("OD pairs:", len(pairs))

# Validate source-top selection cost separately.
selections = []
selection_ms = []

for source, target, _, _ in pairs:
    t0 = perf_counter()
    chosen = select_source_top4(source, target)
    selection_ms.append(
        (perf_counter() - t0) * 1000
    )
    selections.append(chosen)

# Warm up all variants.
for strategy in STRATEGIES:
    for pair_id in range(min(5, len(pairs))):
        source, target, _, _ = pairs[pair_id]

        chosen = {
            "fixed_4": tuple(range(4)),
            "source_top_4": selections[pair_id],
            "all_32": tuple(range(32)),
        }[strategy]

        astar(source, target, chosen)

rng = np.random.default_rng(SEED)
records = []

for repeat in range(REPEATS):
    order = STRATEGIES.copy()
    rng.shuffle(order)

    for strategy in order:
        for pair_id, (
            source, target, origin, destination
        ) in enumerate(pairs):

            chosen = {
                "fixed_4": tuple(range(4)),
                "source_top_4": selections[pair_id],
                "all_32": tuple(range(32)),
            }[strategy]

            t0 = perf_counter()

            cost, expanded, evaluated = astar(
                source, target, chosen
            )

            search_ms = (
                perf_counter() - t0
            ) * 1000

            if abs(cost - references[pair_id]) > 1e-6:
                raise RuntimeError(
                    f"Nonoptimal: {strategy}, OD={pair_id}"
                )

            records.append({
                "repeat": repeat,
                "pair_id": pair_id,
                "strategy": strategy,
                "expanded": expanded,
                "evaluated_heuristics": evaluated,
                "search_ms": search_ms,
            })

    print(f"Completed repeat {repeat + 1}/{REPEATS}")

result = pd.DataFrame(records)

OUTPUT_PATH.parent.mkdir(
    parents=True, exist_ok=True
)
result.to_parquet(OUTPUT_PATH, index=False)

per_repeat = (
    result.groupby(["repeat", "strategy"])
    .agg(
        expanded=("expanded", "sum"),
        evaluated=("evaluated_heuristics", "sum"),
        search_ms=("search_ms", "sum"),
    )
    .reset_index()
)

summary = (
    per_repeat.groupby("strategy")
    .agg(
        expanded=("expanded", "first"),
        evaluated=("evaluated", "first"),
        median_search_ms=("search_ms", "median"),
        min_search_ms=("search_ms", "min"),
        max_search_ms=("search_ms", "max"),
    )
    .reset_index()
)

# Selection time is measured once separately,
# not mixed into the randomized search benchmark.
selection_total = sum(selection_ms)

summary["selection_ms"] = np.where(
    summary["strategy"] == "source_top_4",
    selection_total,
    0.0,
)

summary["search_plus_selection_ms"] = (
    summary["median_search_ms"]
    + summary["selection_ms"]
)

fixed_expanded = int(
    summary.loc[
        summary["strategy"] == "fixed_4",
        "expanded",
    ].iloc[0]
)

summary["expansion_change_pct"] = (
    100 * (
        summary["expanded"] / fixed_expanded - 1
    )
)

print("\n=== RUNTIME SUMMARY ===")
print(
    summary.round(3).to_string(index=False)
)

print("\n=== SOURCE SELECTION ===")
print(
    "Total selection time (ms):",
    round(selection_total, 3),
)
print(
    "Mean selection per OD (ms):",
    round(np.mean(selection_ms), 4),
)

print("\n=== OPTIMALITY ===")
print("All costs matched Dijkstra: True")
print("\nSaved:", OUTPUT_PATH)
