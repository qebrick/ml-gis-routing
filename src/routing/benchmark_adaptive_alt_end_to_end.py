from pathlib import Path
from collections import Counter
from time import perf_counter_ns
import heapq

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from sklearn.model_selection import GroupKFold

GRAPH_PATH = Path(
    "data/graphs/porto_drive_conservative_refit.graphml"
)
LANDMARK_PATH = Path(
    "data/processed/adaptive_alt_landmarks.csv"
)
OD_PATH = Path(
    "data/processed/gps_od_route_comparison.parquet"
)
OUTPUT_PATH = Path(
    "data/processed/adaptive_alt_end_to_end_benchmark.parquet"
)

WEIGHT = "osm_travel_time_s"
KS = (2, 4)
N_REPEATS = 5
SEED = 42

# --------------------------------------------------
# Graph
# --------------------------------------------------

G = ox.load_graphml(GRAPH_PATH)

# Directed graph with minimum weight among parallel
# edges, matching the earlier routing experiments.
H = nx.DiGraph()
H.add_nodes_from(G.nodes(data=True))

for u, v, data in G.edges(data=True):
    w = float(data[WEIGHT])

    if not np.isfinite(w) or w < 0:
        raise ValueError(f"Invalid weight: {u}, {v}, {w}")

    if (
        not H.has_edge(u, v)
        or w < H[u][v][WEIGHT]
    ):
        H.add_edge(u, v, **{WEIGHT: w})

nodes = list(H.nodes)
idx = {node: i for i, node in enumerate(nodes)}
lookup = {str(node): node for node in nodes}
n_nodes = len(nodes)

adj = [
    [
        (idx[v], float(data[WEIGHT]))
        for v, data in H[u].items()
    ]
    for u in nodes
]

# --------------------------------------------------
# OD and landmarks
# --------------------------------------------------

od = pd.read_parquet(OD_PATH).reset_index(drop=True)

od["origin_node"] = [
    lookup[str(v)] for v in od["origin"]
]
od["destination_node"] = [
    lookup[str(v)] for v in od["destination"]
]

landmark_ids = pd.read_csv(
    LANDMARK_PATH
)["node_id"].tolist()

landmarks = [
    lookup[str(v)] for v in landmark_ids
]

assert len(landmarks) == 32
assert len(set(landmarks)) == 32

# --------------------------------------------------
# Precompute directed ALT distances
# --------------------------------------------------

pre_start = perf_counter_ns()

forward = np.full(
    (32, n_nodes), np.inf, dtype=np.float64
)
backward = np.full(
    (32, n_nodes), np.inf, dtype=np.float64
)

R = H.reverse(copy=False)

for j, landmark in enumerate(landmarks):
    f = nx.single_source_dijkstra_path_length(
        H, landmark, weight=WEIGHT
    )
    b = nx.single_source_dijkstra_path_length(
        R, landmark, weight=WEIGHT
    )

    for node, value in f.items():
        forward[j, idx[node]] = value

    for node, value in b.items():
        backward[j, idx[node]] = value

preprocessing_ms = (
    perf_counter_ns() - pre_start
) / 1e6

# --------------------------------------------------
# Geographic distances (coordinates are read from
# graph; computation itself is timed per query)
# --------------------------------------------------

lat = np.array([
    float(H.nodes[node]["y"]) for node in nodes
])
lon = np.array([
    float(H.nodes[node]["x"]) for node in nodes
])

def air_distance_m(s, t):
    p1 = np.radians(lat[s])
    p2 = np.radians(lat[t])
    dp = p2 - p1
    dl = np.radians(lon[t] - lon[s])

    a = (
        np.sin(dp / 2) ** 2
        + np.cos(p1) * np.cos(p2)
        * np.sin(dl / 2) ** 2
    )

    return float(
        2 * 6371000.0
        * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    )

# --------------------------------------------------
# Directed ALT contributions
# h(v,t) = max(
#   d(L,t) - d(L,v),
#   d(v,L) - d(t,L),
#   0
# )
# --------------------------------------------------

def landmark_contributions(s, t):
    first = np.zeros(32, dtype=np.float64)
    second = np.zeros(32, dtype=np.float64)

    valid = (
        np.isfinite(forward[:, t])
        & np.isfinite(forward[:, s])
    )

    np.subtract(
        forward[:, t],
        forward[:, s],
        out=first,
        where=valid,
    )

    valid = (
        np.isfinite(backward[:, s])
        & np.isfinite(backward[:, t])
    )

    np.subtract(
        backward[:, s],
        backward[:, t],
        out=second,
        where=valid,
    )

    return np.maximum(
        np.maximum(first, second), 0.0
    )

def select_landmarks(s, t, k):
    contributions = landmark_contributions(s, t)

    # Stable tie-breaking by original landmark index.
    order = np.lexsort((
        np.arange(32),
        -contributions,
    ))

    return order[:k]

def build_heuristic(chosen, t):
    h = np.zeros(n_nodes, dtype=np.float64)

    for j in chosen:
        a = np.zeros(n_nodes, dtype=np.float64)
        b = np.zeros(n_nodes, dtype=np.float64)

        valid = (
            np.isfinite(forward[j, t])
            & np.isfinite(forward[j])
        )

        np.subtract(
            forward[j, t],
            forward[j],
            out=a,
            where=valid,
        )

        valid = (
            np.isfinite(backward[j])
            & np.isfinite(backward[j, t])
        )

        np.subtract(
            backward[j],
            backward[j, t],
            out=b,
            where=valid,
        )

        np.maximum(h, a, out=h)
        np.maximum(h, b, out=h)

    h[t] = 0.0
    return h

# --------------------------------------------------
# A* with high-g tie-breaking
# --------------------------------------------------

def astar(s, t, h):
    if s == t:
        return 0.0, 0

    dist = np.full(n_nodes, np.inf)
    dist[s] = 0.0

    counter = 0
    heap = [(float(h[s]), -0.0, counter, s)]
    expanded = 0

    while heap:
        f, negative_g, _, u = heapq.heappop(heap)
        g = -negative_g

        if g > dist[u] + 1e-9:
            continue

        expanded += 1

        if u == t:
            return g, expanded

        for v, w in adj[u]:
            ng = g + w

            if ng < dist[v] - 1e-9:
                dist[v] = ng
                counter += 1

                heapq.heappush(
                    heap,
                    (
                        ng + float(h[v]),
                        -ng,
                        counter,
                        v,
                    ),
                )

    return np.inf, expanded

# --------------------------------------------------
# GroupKFold thresholds by destination
# --------------------------------------------------

features = od[[
    "origin_node",
    "destination_node",
]].copy()

features["air_distance_m"] = [
    air_distance_m(
        idx[row.origin_node],
        idx[row.destination_node],
    )
    for row in features.itertuples()
]

groups = od["destination"].astype(str).to_numpy()

thresholds = np.empty(len(od), dtype=np.float64)
fold_ids = np.empty(len(od), dtype=np.int64)

cv = GroupKFold(n_splits=5)

for fold, (train, test) in enumerate(
    cv.split(features, groups=groups), start=1
):
    threshold = float(
        features.iloc[train][
            "air_distance_m"
        ].quantile(0.75)
    )

    thresholds[test] = threshold
    fold_ids[test] = fold

# --------------------------------------------------
# Dijkstra reference
# --------------------------------------------------

reference = {}

for pair_id, row in enumerate(od.itertuples()):
    s = row.origin_node
    t = row.destination_node

    reference[pair_id] = nx.shortest_path_length(
        H, s, t, weight=WEIGHT
    )

# --------------------------------------------------
# Warm-up
# --------------------------------------------------

for pair_id in range(min(20, len(od))):
    row = od.iloc[pair_id]
    s = idx[row["origin_node"]]
    t = idx[row["destination_node"]]

    chosen = select_landmarks(s, t, 4)
    h = build_heuristic(chosen, t)
    astar(s, t, h)

# --------------------------------------------------
# Randomized end-to-end benchmark
# --------------------------------------------------

rng = np.random.default_rng(SEED)

strategies = [
    "adaptive_alt_2",
    "adaptive_alt_4",
    "distance_gated_alt",
]

records = []

for repeat in range(N_REPEATS):
    pair_order = rng.permutation(len(od))

    for pair_id in pair_order:
        row = od.iloc[int(pair_id)]

        s = idx[row["origin_node"]]
        t = idx[row["destination_node"]]

        strategy_order = rng.permutation(strategies)

        for strategy in strategy_order:
            start = perf_counter_ns()

            if strategy == "distance_gated_alt":
                distance = air_distance_m(s, t)
                k = (
                    4 if distance >= thresholds[pair_id]
                    else 2
                )
            elif strategy == "adaptive_alt_2":
                k = 2
            else:
                k = 4

            selected = select_landmarks(s, t, k)
            selection_end = perf_counter_ns()

            h = build_heuristic(selected, t)
            preparation_end = perf_counter_ns()

            cost, expanded = astar(s, t, h)
            end = perf_counter_ns()

            expected = reference[int(pair_id)]

            if not np.isclose(
                cost, expected,
                rtol=1e-8,
                atol=1e-6,
            ):
                raise AssertionError(
                    f"Nonoptimal: pair={pair_id}, "
                    f"strategy={strategy}, "
                    f"cost={cost}, expected={expected}"
                )

            records.append({
                "repeat": repeat,
                "pair_id": int(pair_id),
                "destination": str(
                    row["destination_node"]
                ),
                "fold": int(fold_ids[pair_id]),
                "strategy": strategy,
                "k": int(k),
                "expanded": int(expanded),
                "selection_ms": (
                    selection_end - start
                ) / 1e6,
                "preparation_ms": (
                    preparation_end - selection_end
                ) / 1e6,
                "search_ms": (
                    end - preparation_end
                ) / 1e6,
                "total_ms": (
                    end - start
                ) / 1e6,
                "optimal": True,
            })

    print(
        f"Completed repeat {repeat + 1}/{N_REPEATS}",
        flush=True,
    )

result = pd.DataFrame(records)

OUTPUT_PATH.parent.mkdir(
    parents=True, exist_ok=True
)
result.to_parquet(OUTPUT_PATH, index=False)

# --------------------------------------------------
# Summary
# --------------------------------------------------

summary = (
    result.groupby(["repeat", "strategy"])
    .agg(
        total_ms=("total_ms", "sum"),
        expanded=("expanded", "sum"),
    )
    .reset_index()
)

print("\n=== PREPROCESSING ===")
print("Graph nodes:", n_nodes)
print("Graph edges:", H.number_of_edges())
print("ALT preprocessing (ms):", round(preprocessing_ms, 3))
print("OD pairs:", len(od))
print("Repeats:", N_REPEATS)

print("\n=== END-TO-END SUMMARY ===")

print(
    summary.groupby("strategy")
    .agg(
        median_total_ms=("total_ms", "median"),
        min_total_ms=("total_ms", "min"),
        max_total_ms=("total_ms", "max"),
        expanded=("expanded", "first"),
    )
    .round(3)
    .to_string()
)

print("\n=== BY REPEAT ===")
print(
    summary.pivot(
        index="repeat",
        columns="strategy",
        values="total_ms",
    ).round(3).to_string()
)

print("\n=== GATE COVERAGE ===")

gate = result[
    (result["strategy"] == "distance_gated_alt")
    & (result["repeat"] == 0)
]

print("k=2:", int((gate["k"] == 2).sum()))
print("k=4:", int((gate["k"] == 4).sum()))

print("\n=== PAIRED COMPARISON ===")

wide = result.pivot(
    index=["repeat", "pair_id", "destination"],
    columns="strategy",
    values="total_ms",
).reset_index()

for baseline in [
    "adaptive_alt_2",
    "adaptive_alt_4",
]:
    delta = (
        wide["distance_gated_alt"]
        - wide[baseline]
    )

    print(
        f"Gate - {baseline}: "
        f"mean={delta.mean():+.5f} ms/OD, "
        f"faster={int((delta < 0).sum())}, "
        f"slower={int((delta > 0).sum())}"
    )

    per_od = (
        wide.assign(delta=delta)
        .groupby(
            ["pair_id", "destination"],
            as_index=False,
        )["delta"].mean()
    )

    clusters = [
        g["delta"].to_numpy()
        for _, g in per_od.groupby("destination")
    ]

    boot_rng = np.random.default_rng(42)
    boot = np.empty(5000)

    for b in range(len(boot)):
        selected = boot_rng.integers(
            0, len(clusters), len(clusters)
        )
        boot[b] = np.concatenate([
            clusters[j] for j in selected
        ]).mean()

    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(
        "  Target-cluster 95% CI:",
        f"[{lo:+.5f}, {hi:+.5f}] ms/OD",
    )

print("\n=== OPTIMALITY ===")
print("All routes optimal:", result["optimal"].all())
print("Saved:", OUTPUT_PATH)
