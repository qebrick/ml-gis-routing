from pathlib import Path
from heapq import heappush, heappop
from itertools import count
from time import perf_counter

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.model_selection import GroupShuffleSplit

GRAPH_PATH = Path(
    "data/graphs/porto_drive_conservative_refit.graphml"
)
DIAGNOSTICS_PATH = Path(
    "data/processed/alt_heuristic_diagnostics.parquet"
)
OD_PATH = Path(
    "data/processed/gps_od_route_comparison.parquet"
)
OUTPUT_PATH = Path(
    "data/processed/ml_guided_astar_benchmark.parquet"
)

SEED = 42
TOL = 1e-6

FEATURES = [
    "alt_cost_s",
    "node_lat",
    "node_lon",
    "target_lat",
    "target_lon",
    "delta_lat",
    "delta_lon",
    "abs_delta_lat",
    "abs_delta_lon",
]

G = ox.load_graphml(GRAPH_PATH)

H = nx.DiGraph()
H.add_nodes_from(G.nodes)

for u, v, key, attrs in G.edges(keys=True, data=True):
    w = float(attrs["osm_travel_time_s"])

    if not np.isfinite(w) or w <= 0:
        raise ValueError("Invalid edge weight")

    if not H.has_edge(u, v) or w < H[u][v]["weight"]:
        H.add_edge(u, v, weight=w)

nodes = list(H.nodes)
index = {node: i for i, node in enumerate(nodes)}
N = len(nodes)

adj = [[] for _ in range(N)]

for u, v, attrs in H.edges(data=True):
    adj[index[u]].append(
        (index[v], attrs["weight"])
    )

lat = np.array([
    float(G.nodes[node]["y"]) for node in nodes
])
lon = np.array([
    float(G.nodes[node]["x"]) for node in nodes
])

diagnostics = pd.read_parquet(DIAGNOSTICS_PATH)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=SEED,
)

train_idx, test_idx = next(
    splitter.split(
        diagnostics,
        groups=diagnostics["target"],
    )
)

train = diagnostics.iloc[train_idx].copy()
test_targets = set(
    diagnostics.iloc[test_idx]["target"].astype(int)
)

def add_features(frame):
    frame = frame.copy()

    node_idx = frame["node"].map(
        lambda n: index[int(n)]
    ).to_numpy()

    target_idx = frame["target"].map(
        lambda n: index[int(n)]
    ).to_numpy()

    frame["node_lat"] = lat[node_idx]
    frame["node_lon"] = lon[node_idx]
    frame["target_lat"] = lat[target_idx]
    frame["target_lon"] = lon[target_idx]

    frame["delta_lat"] = (
        frame["target_lat"] - frame["node_lat"]
    )
    frame["delta_lon"] = (
        frame["target_lon"] - frame["node_lon"]
    )

    frame["abs_delta_lat"] = (
        frame["delta_lat"].abs()
    )
    frame["abs_delta_lon"] = (
        frame["delta_lon"].abs()
    )

    return frame


train = add_features(train)

model = CatBoostRegressor(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="MAE",
    random_seed=SEED,
    verbose=False,
    thread_count=-1,
)

print("=== TRAINING ===")
print("Training pairs:", len(train))
print("Training targets:", train["target"].nunique())
print("Held-out targets:", len(test_targets))

t0 = perf_counter()
model.fit(train[FEATURES], train["gap_s"])
print(
    "Training time (s):",
    round(perf_counter() - t0, 3),
)

# Reconstruct the exact ALT-4 lower bound from
# diagnostic rows. For all reachable nodes and
# each test target, this is already available.
od = pd.read_parquet(OD_PATH)
od = od[
    od["destination"].astype(int).isin(test_targets)
].copy()

pairs = [
    (
        index[int(row.origin)],
        index[int(row.destination)],
    )
    for row in od.itertuples(index=False)
]

if not pairs:
    raise RuntimeError("No held-out GPS OD pairs")

needed_targets = sorted({
    nodes[target] for _, target in pairs
})

diag_test = diagnostics[
    diagnostics["target"].isin(needed_targets)
]

safe_by_target = {}
ml_by_target = {}
inference_ms = {}

for target_node in needed_targets:
    target_idx = index[target_node]

    rows = diag_test[
        diag_test["target"] == target_node
    ]

    # ALT values are available for reachable
    # nodes. Unreachable nodes receive 0,
    # which is always an admissible bound.
    safe = np.zeros(N, dtype=np.float64)

    row_indices = rows["node"].map(
        lambda n: index[int(n)]
    ).to_numpy()

    safe[row_indices] = rows[
        "alt_cost_s"
    ].to_numpy(dtype=float)

    safe[target_idx] = 0.0

    safe_by_target[target_idx] = safe

    # Generate predictions for every graph node.
    features = pd.DataFrame({
        "alt_cost_s": safe,
        "node_lat": lat,
        "node_lon": lon,
        "target_lat": lat[target_idx],
        "target_lon": lon[target_idx],
    })

    features["delta_lat"] = (
        features["target_lat"] - features["node_lat"]
    )
    features["delta_lon"] = (
        features["target_lon"] - features["node_lon"]
    )

    features["abs_delta_lat"] = (
        features["delta_lat"].abs()
    )
    features["abs_delta_lon"] = (
        features["delta_lon"].abs()
    )

    t0 = perf_counter()

    predicted_gap = np.maximum(
        model.predict(features[FEATURES]),
        0.0,
    )

    ml = safe + predicted_gap
    ml[target_idx] = 0.0

    inference_ms[target_idx] = (
        perf_counter() - t0
    ) * 1000

    ml_by_target[target_idx] = ml


def astar_alt(source, target, safe):
    best = {source: 0.0}
    sequence = count()

    heap = [
        (float(safe[source]), next(sequence),
         0.0, source)
    ]

    expanded = 0

    while heap:
        _, _, g, node = heappop(heap)

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
                        new_g + float(safe[neighbor]),
                        next(sequence),
                        new_g,
                        neighbor,
                    ),
                )

    raise RuntimeError("No path")


def astar_ml_guided(source, target, safe, ml):
    """
    ML priority determines expansion order.

    ALT lower bounds determine when the
    incumbent solution is provably optimal.

    Both queues use lazy deletion and shared
    current g-values. Nodes may be reopened.
    """
    best = {source: 0.0}
    expanded_at = {}

    sequence = count()

    ml_heap = []
    safe_heap = []

    active = {}

    def push(node, g):
        token = next(sequence)
        active[node] = token

        heappush(
            ml_heap,
            (
                g + float(ml[node]),
                token,
                node,
                g,
            ),
        )

        heappush(
            safe_heap,
            (
                g + float(safe[node]),
                token,
                node,
                g,
            ),
        )

    push(source, 0.0)

    incumbent = np.inf
    expanded = 0

    while True:
        # Discard invalid entries in safe heap.
        while safe_heap:
            _, token, node, g = safe_heap[0]

            if (
                active.get(node) != token
                or g != best.get(node, np.inf)
            ):
                heappop(safe_heap)
            else:
                break

        lower_bound = (
            safe_heap[0][0]
            if safe_heap else np.inf
        )

        if incumbent <= lower_bound + 1e-9:
            if np.isfinite(incumbent):
                return incumbent, expanded
            raise RuntimeError("No path")

        # Choose next node using ML ordering.
        while ml_heap:
            _, token, node, g = heappop(ml_heap)

            if (
                active.get(node) == token
                and g == best.get(node, np.inf)
            ):
                break
        else:
            raise RuntimeError(
                "ML heap exhausted unexpectedly"
            )

        del active[node]

        expanded += 1
        expanded_at[node] = g

        if node == target:
            incumbent = min(incumbent, g)
            continue

        for neighbor, weight in adj[node]:
            new_g = g + weight

            if new_g < best.get(neighbor, np.inf):
                best[neighbor] = new_g

                # The destination itself gives
                # a complete feasible route.
                if neighbor == target:
                    incumbent = min(
                        incumbent, new_g
                    )

                # Reopening is allowed, including
                # nodes expanded previously.
                push(neighbor, new_g)

    raise RuntimeError("Unreachable")


records = []

print("\n=== ROUTING BENCHMARK ===")
print("Held-out GPS OD pairs:", len(pairs))
print("Unique held-out targets:", len(needed_targets))

for source, target in pairs:
    safe = safe_by_target[target]
    ml = ml_by_target[target]

    reference = nx.shortest_path_length(
        H,
        nodes[source],
        nodes[target],
        weight="weight",
    )

    t0 = perf_counter()
    alt_cost, alt_expanded = astar_alt(
        source, target, safe
    )
    alt_ms = (perf_counter() - t0) * 1000

    t0 = perf_counter()
    ml_cost, ml_expanded = astar_ml_guided(
        source, target, safe, ml
    )
    ml_ms = (perf_counter() - t0) * 1000

    if (
        abs(alt_cost - reference) > TOL
        or abs(ml_cost - reference) > TOL
    ):
        raise RuntimeError(
            f"Optimality failure: "
            f"{nodes[source]}->{nodes[target]}"
        )

    records.append({
        "source": nodes[source],
        "target": nodes[target],
        "reference_cost_s": reference,
        "alt_expanded": alt_expanded,
        "ml_expanded": ml_expanded,
        "alt_search_ms": alt_ms,
        "ml_search_ms": ml_ms,
    })

result = pd.DataFrame(records)

OUTPUT_PATH.parent.mkdir(
    parents=True, exist_ok=True
)
result.to_parquet(OUTPUT_PATH, index=False)

alt_expanded = result["alt_expanded"].sum()
ml_expanded = result["ml_expanded"].sum()

alt_time = result["alt_search_ms"].sum()
ml_time = result["ml_search_ms"].sum()

total_inference = sum(inference_ms.values())

print("\n=== SEARCH EFFICIENCY ===")
print("ALT expanded:", alt_expanded)
print("ML expanded:", ml_expanded)
print(
    "Expansion reduction (%):",
    round(
        100 * (1 - ml_expanded / alt_expanded),
        3,
    ),
)

print("\n=== RUNTIME ===")
print("ALT search (ms):", round(alt_time, 3))
print("ML search (ms):", round(ml_time, 3))
print(
    "ML inference for unique targets (ms):",
    round(total_inference, 3),
)
print(
    "ML search + inference (ms):",
    round(ml_time + total_inference, 3),
)

print("\n=== PER-ROUTE COMPARISON ===")
print(
    "ML expands fewer:",
    int(
        (result["ml_expanded"]
         < result["alt_expanded"]).sum()
    ),
)
print(
    "ML expands more:",
    int(
        (result["ml_expanded"]
         > result["alt_expanded"]).sum()
    ),
)
print(
    "Equal expansions:",
    int(
        (result["ml_expanded"]
         == result["alt_expanded"]).sum()
    ),
)

print("\n=== OPTIMALITY ===")
print("All routes match Dijkstra: True")

print("\nSaved:", OUTPUT_PATH)
