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
PREDICTIONS_PATH = Path(
    "data/processed/ml_alt_gap_predictions.parquet"
)
OD_PATH = Path(
    "data/processed/gps_od_route_comparison.parquet"
)
OUTPUT_PATH = Path(
    "data/processed/astar_tiebreak_strategies.parquet"
)

STRATEGIES = ["fifo", "high_g", "low_h", "ml"]
REPEATS = 10
SEED = 42
TOL = 1e-6

G = ox.load_graphml(GRAPH_PATH)

H = nx.DiGraph()
H.add_nodes_from(G.nodes)

for u, v, key, attrs in G.edges(
    keys=True, data=True
):
    w = float(attrs["osm_travel_time_s"])

    if not np.isfinite(w) or w <= 0:
        raise ValueError("Invalid edge weight")

    if (
        not H.has_edge(u, v)
        or w < H[u][v]["weight"]
    ):
        H.add_edge(u, v, weight=w)

nodes = list(H.nodes)
idx = {node: i for i, node in enumerate(nodes)}
N = len(nodes)

adj = [[] for _ in nodes]

for u, v, attrs in H.edges(data=True):
    adj[idx[u]].append(
        (idx[v], attrs["weight"])
    )

pred = pd.read_parquet(PREDICTIONS_PATH)
od = pd.read_parquet(OD_PATH)

test_targets = set(
    pred["target"].astype(int)
)

od = od[
    od["destination"].astype(int).isin(
        test_targets
    )
].copy()

target_arrays = {}

for target_node, rows in pred.groupby("target"):
    target_node = int(target_node)

    safe = np.zeros(N, dtype=float)
    ml = np.zeros(N, dtype=float)

    positions = rows["node"].map(
        lambda n: idx[int(n)]
    ).to_numpy()

    safe[positions] = (
        rows["alt_cost_s"].to_numpy()
    )

    ml[positions] = (
        rows["ml_predicted_cost_s"].to_numpy()
    )

    safe[idx[target_node]] = 0.0
    ml[idx[target_node]] = 0.0

    target_arrays[target_node] = (safe, ml)


def astar(source, target, safe, ml, strategy):
    best = {source: 0.0}
    serial = count()
    heap = []

    def push(node, g):
        h = float(safe[node])
        f = g + h

        if strategy == "fifo":
            secondary = 0.0
        elif strategy == "high_g":
            secondary = -g
        elif strategy == "low_h":
            secondary = h
        elif strategy == "ml":
            secondary = float(ml[node])
        else:
            raise ValueError(strategy)

        heappush(
            heap,
            (
                f,
                secondary,
                next(serial),
                g,
                node,
            ),
        )

    push(source, 0.0)
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

            if new_g < best.get(
                neighbor, np.inf
            ):
                best[neighbor] = new_g
                push(neighbor, new_g)

    raise RuntimeError("No path")


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

print("=== COVERAGE ===")
print("OD pairs:", len(pairs))
print(
    "Unique targets:",
    len({p[3] for p in pairs})
)

# Warm up all variants.
for strategy in STRATEGIES:
    for source, target, _, destination in pairs[:5]:
        safe, ml = target_arrays[destination]
        astar(source, target, safe, ml, strategy)

rng = np.random.default_rng(SEED)
records = []

for repeat in range(REPEATS):
    order = STRATEGIES.copy()
    rng.shuffle(order)

    for strategy in order:
        for pair_id, (
            source,
            target,
            origin,
            destination,
        ) in enumerate(pairs):
            safe, ml = target_arrays[destination]

            t0 = perf_counter()

            cost, expanded = astar(
                source,
                target,
                safe,
                ml,
                strategy,
            )

            runtime_ms = (
                perf_counter() - t0
            ) * 1000

            if abs(cost - references[pair_id]) > TOL:
                raise RuntimeError(
                    f"Optimality failure: {strategy}"
                )

            records.append({
                "repeat": repeat,
                "pair_id": pair_id,
                "origin": origin,
                "destination": destination,
                "strategy": strategy,
                "expanded": expanded,
                "runtime_ms": runtime_ms,
            })

    print(
        f"Completed repeat "
        f"{repeat + 1}/{REPEATS}"
    )

result = pd.DataFrame(records)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)
result.to_parquet(OUTPUT_PATH, index=False)

per_repeat = (
    result.groupby(["repeat", "strategy"])
    .agg(
        total_expanded=("expanded", "sum"),
        total_runtime_ms=("runtime_ms", "sum"),
    )
    .reset_index()
)

summary = (
    per_repeat.groupby("strategy")
    .agg(
        total_expanded=("total_expanded", "first"),
        median_runtime_ms=(
            "total_runtime_ms", "median"
        ),
        min_runtime_ms=(
            "total_runtime_ms", "min"
        ),
        max_runtime_ms=(
            "total_runtime_ms", "max"
        ),
    )
    .reset_index()
)

baseline = int(
    summary.loc[
        summary["strategy"] == "fifo",
        "total_expanded",
    ].iloc[0]
)

summary["expansion_change_pct"] = (
    100
    * (
        summary["total_expanded"] / baseline
        - 1
    )
)

print("\n=== STRATEGY SUMMARY ===")
print(
    summary.round(3).to_string(index=False)
)

# Compare per-route expansions using the first
# repetition; expansion counts are deterministic.
first = result[result["repeat"] == 0]

wide = first.pivot(
    index="pair_id",
    columns="strategy",
    values="expanded",
)

print("\n=== PAIRWISE VS FIFO ===")

for strategy in ["high_g", "low_h", "ml"]:
    delta = wide[strategy] - wide["fifo"]

    print(
        strategy,
        "better:", int((delta < 0).sum()),
        "worse:", int((delta > 0).sum()),
        "equal:", int((delta == 0).sum()),
        "total delta:", int(delta.sum()),
    )

print("\n=== ML VS HIGH_G ===")

delta = wide["ml"] - wide["high_g"]

print("ML better:", int((delta < 0).sum()))
print("ML worse:", int((delta > 0).sum()))
print("Equal:", int((delta == 0).sum()))
print("Total expansion delta:", int(delta.sum()))

print("\n=== CONSISTENCY CHECK ===")

print(
    "high_g and low_h same expansions:",
    bool(
        (wide["high_g"] == wide["low_h"]).all()
    ),
)

print("\n=== OPTIMALITY ===")
print("All routes matched Dijkstra: True")
print("\nSaved:", OUTPUT_PATH)
