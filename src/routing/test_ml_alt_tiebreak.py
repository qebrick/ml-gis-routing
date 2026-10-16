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
    "data/processed/ml_alt_tiebreak_comparison.parquet"
)

G = ox.load_graphml(GRAPH_PATH)

H = nx.DiGraph()
H.add_nodes_from(G.nodes)

for u, v, key, attrs in G.edges(keys=True, data=True):
    w = float(attrs["osm_travel_time_s"])
    if not H.has_edge(u, v) or w < H[u][v]["weight"]:
        H.add_edge(u, v, weight=w)

nodes = list(H.nodes)
idx = {node: i for i, node in enumerate(nodes)}
N = len(nodes)

adj = [[] for _ in nodes]
for u, v, attrs in H.edges(data=True):
    adj[idx[u]].append((idx[v], attrs["weight"]))

pred = pd.read_parquet(PREDICTIONS_PATH)
od = pd.read_parquet(OD_PATH)

test_targets = set(pred["target"].astype(int))

od = od[
    od["destination"].astype(int).isin(test_targets)
].copy()

target_arrays = {}

for target, rows in pred.groupby("target"):
    target = int(target)

    safe = np.zeros(N, dtype=float)
    ml = np.zeros(N, dtype=float)

    positions = rows["node"].map(
        lambda n: idx[int(n)]
    ).to_numpy()

    safe[positions] = rows["alt_cost_s"].to_numpy()
    ml[positions] = rows["ml_predicted_cost_s"].to_numpy()

    safe[idx[target]] = 0.0
    ml[idx[target]] = 0.0

    target_arrays[target] = (safe, ml)


def astar(source, target, safe, ml, use_ml):
    best = {source: 0.0}
    serial = count()
    heap = []

    def push(node, g):
        secondary = float(ml[node]) if use_ml else 0.0
        heappush(
            heap,
            (
                g + float(safe[node]),
                secondary,
                next(serial),
                g,
                node,
            ),
        )

    push(source, 0.0)

    expanded = 0
    exact_tie_pops = 0

    while heap:
        f, _, _, g, node = heappop(heap)

        if g > best.get(node, np.inf):
            continue

        expanded += 1

        # Count whether another valid queue entry
        # has exactly the same primary priority.
        while heap and (
            heap[0][3] > best.get(heap[0][4], np.inf)
        ):
            heappop(heap)

        if heap and heap[0][0] == f:
            exact_tie_pops += 1

        if node == target:
            return g, expanded, exact_tie_pops

        for neighbor, weight in adj[node]:
            new_g = g + weight

            if new_g < best.get(neighbor, np.inf):
                best[neighbor] = new_g
                push(neighbor, new_g)

    raise RuntimeError("No path")


records = []

for row in od.itertuples(index=False):
    source_node = int(row.origin)
    target_node = int(row.destination)

    source = idx[source_node]
    target = idx[target_node]

    safe, ml = target_arrays[target_node]

    reference = nx.shortest_path_length(
        H,
        source_node,
        target_node,
        weight="weight",
    )

    outputs = {}

    for label, use_ml in [
        ("alt", False),
        ("alt_ml_tiebreak", True),
    ]:
        start = perf_counter()

        cost, expanded, ties = astar(
            source, target, safe, ml, use_ml
        )

        elapsed_ms = (
            perf_counter() - start
        ) * 1000

        if abs(cost - reference) > 1e-6:
            raise RuntimeError(
                f"Nonoptimal route: {label}"
            )

        outputs[label] = (expanded, ties, elapsed_ms)

    records.append({
        "origin": source_node,
        "destination": target_node,
        "alt_expanded": outputs["alt"][0],
        "ml_tiebreak_expanded": outputs["alt_ml_tiebreak"][0],
        "alt_exact_tie_pops": outputs["alt"][1],
        "ml_exact_tie_pops": outputs["alt_ml_tiebreak"][1],
        "alt_runtime_ms": outputs["alt"][2],
        "ml_tiebreak_runtime_ms": outputs["alt_ml_tiebreak"][2],
    })

result = pd.DataFrame(records)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)
result.to_parquet(OUTPUT_PATH, index=False)

alt_total = result["alt_expanded"].sum()
ml_total = result["ml_tiebreak_expanded"].sum()

print("=== TEST COVERAGE ===")
print("OD pairs:", len(result))
print("Unique targets:", result["destination"].nunique())

print("\n=== EXACT PRIORITY TIES ===")
print(
    "ALT pops with an equal-priority queue entry:",
    int(result["alt_exact_tie_pops"].sum()),
)
print(
    "Total ALT expansions:",
    int(alt_total),
)
print(
    "Tie frequency (%):",
    round(
        100 * result["alt_exact_tie_pops"].sum()
        / alt_total,
        4,
    ),
)

print("\n=== SEARCH COMPARISON ===")
print("ALT expanded:", int(alt_total))
print("ALT+ML tie-break expanded:", int(ml_total))
print(
    "Expansion change (%):",
    round(100 * (ml_total / alt_total - 1), 3),
)
print(
    "ML better routes:",
    int(
        (result["ml_tiebreak_expanded"]
         < result["alt_expanded"]).sum()
    ),
)
print(
    "ML worse routes:",
    int(
        (result["ml_tiebreak_expanded"]
         > result["alt_expanded"]).sum()
    ),
)
print(
    "Equal routes:",
    int(
        (result["ml_tiebreak_expanded"]
         == result["alt_expanded"]).sum()
    ),
)

print("\n=== SINGLE-RUN TIMING ===")
print(
    "ALT search (ms):",
    round(result["alt_runtime_ms"].sum(), 3),
)
print(
    "ALT+ML search (ms):",
    round(result["ml_tiebreak_runtime_ms"].sum(), 3),
)

print("\n=== OPTIMALITY ===")
print("All costs matched Dijkstra: True")
print("\nSaved:", OUTPUT_PATH)
