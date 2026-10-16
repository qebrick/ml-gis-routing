from pathlib import Path
from heapq import heappush, heappop
from itertools import count
from time import perf_counter
from math import radians, sin, cos, asin, sqrt
import json

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
    "data/processed/admissible_astar_benchmark.parquet"
)
META_PATH = Path(
    "data/processed/admissible_astar_benchmark_meta.json"
)

WEIGHT = "osm_travel_time_s"
EARTH_RADIUS_M = 6371008.8
TOL = 1e-7

G = ox.load_graphml(GRAPH_PATH)
od = pd.read_parquet(OD_PATH)

coords = {
    node: (
        radians(float(attrs["y"])),
        radians(float(attrs["x"])),
    )
    for node, attrs in G.nodes(data=True)
}


def distance_m(u, v):
    lat1, lon1 = coords[u]
    lat2, lon2 = coords[v]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )

    return 2 * EARTH_RADIUS_M * asin(
        min(1.0, sqrt(max(0.0, a)))
    )


# Construct the exact adjacency costs used by the search.
# For parallel directed edges, choose the cheapest one.
adjacency = {}

alpha = float("inf")
positive_distance_edges = 0

for u, v, key, attrs in G.edges(keys=True, data=True):
    w = float(attrs[WEIGHT])

    if not np.isfinite(w) or w <= 0:
        raise ValueError(
            f"Invalid edge weight: {(u, v, key)} = {w}"
        )

    adjacency.setdefault(u, {})

    current = adjacency[u].get(v, float("inf"))
    adjacency[u][v] = min(current, w)

    d = distance_m(u, v)

    if d > 0:
        alpha = min(alpha, w / d)
        positive_distance_edges += 1

if not np.isfinite(alpha) or alpha <= 0:
    raise RuntimeError("Cannot construct heuristic")

# Slightly reduce the coefficient to avoid
# floating-point boundary violations.
alpha = float(np.nextafter(alpha, 0.0))


def shortest_path(source, target, use_heuristic):
    if source not in G or target not in G:
        raise ValueError("Unknown source or target")

    heuristic_cache = {}

    def h(node):
        if not use_heuristic:
            return 0.0

        if node not in heuristic_cache:
            heuristic_cache[node] = (
                alpha * distance_m(node, target)
            )

        return heuristic_cache[node]

    best_g = {source: 0.0}
    parent = {}

    queue = []
    sequence = count()

    heappush(
        queue,
        (h(source), next(sequence), 0.0, source),
    )

    expanded = 0
    popped = 0
    relaxations = 0

    while queue:
        f, _, g, node = heappop(queue)
        popped += 1

        if g > best_g.get(node, float("inf")):
            continue

        expanded += 1

        if node == target:
            path = [target]

            while path[-1] != source:
                path.append(parent[path[-1]])

            path.reverse()

            return {
                "cost_s": g,
                "path": path,
                "expanded": expanded,
                "popped": popped,
                "relaxations": relaxations,
                "heuristic_evaluations": len(heuristic_cache),
            }

        for neighbor, weight in adjacency.get(node, {}).items():
            new_g = g + weight

            if new_g < best_g.get(
                neighbor, float("inf")
            ):
                best_g[neighbor] = new_g
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

                relaxations += 1

    raise RuntimeError(
        f"No path from {source} to {target}"
    )


# Check consistency on every graph edge.
violations = 0
max_violation = 0.0

for u, v, key, attrs in G.edges(keys=True, data=True):
    # By the triangle inequality, it is sufficient
    # to check the edge-level geographic lower bound.
    lower_bound = alpha * distance_m(u, v)
    w = float(attrs[WEIGHT])

    excess = lower_bound - w
    max_violation = max(max_violation, excess)

    if excess > TOL:
        violations += 1

print("=== HEURISTIC VALIDATION ===")
print("Graph nodes:", G.number_of_nodes())
print("Graph edges:", G.number_of_edges())
print("Positive-distance edges:", positive_distance_edges)
print("Alpha (s/m):", round(alpha, 9))
print(
    "Implied maximum speed (km/h):",
    round(3.6 / alpha, 3),
)
print("Edge-bound violations:", violations)
print("Maximum edge-bound excess (s):", max_violation)

if violations:
    raise RuntimeError("Heuristic lower-bound violation")

rows = []

for pair in od.itertuples(index=False):
    source = int(pair.origin)
    target = int(pair.destination)

    t0 = perf_counter()
    dijkstra = shortest_path(
        source, target, use_heuristic=False
    )
    dijkstra_ms = 1000 * (perf_counter() - t0)

    t0 = perf_counter()
    astar = shortest_path(
        source, target, use_heuristic=True
    )
    astar_ms = 1000 * (perf_counter() - t0)

    cost_difference = (
        astar["cost_s"] - dijkstra["cost_s"]
    )

    if abs(cost_difference) > TOL:
        raise RuntimeError(
            f"Cost mismatch for {source}->{target}: "
            f"{cost_difference}"
        )

    rows.append({
        "trip_id": str(pair.trip_id),
        "hmm_fragment_id": str(pair.hmm_fragment_id),
        "origin": source,
        "destination": target,
        "optimal_cost_s": dijkstra["cost_s"],
        "dijkstra_expanded": dijkstra["expanded"],
        "astar_expanded": astar["expanded"],
        "dijkstra_popped": dijkstra["popped"],
        "astar_popped": astar["popped"],
        "dijkstra_ms": dijkstra_ms,
        "astar_ms": astar_ms,
        "astar_heuristic_evaluations": (
            astar["heuristic_evaluations"]
        ),
        "cost_difference_s": cost_difference,
        "same_node_path": (
            dijkstra["path"] == astar["path"]
        ),
    })

result = pd.DataFrame(rows)

result["expansion_reduction_pct"] = (
    100
    * (
        1
        - result["astar_expanded"]
        / result["dijkstra_expanded"]
    )
)

result["runtime_reduction_pct"] = (
    100
    * (
        1
        - result["astar_ms"]
        / result["dijkstra_ms"]
    )
)

OUTPUT_PATH.parent.mkdir(
    parents=True, exist_ok=True
)
result.to_parquet(OUTPUT_PATH, index=False)

META_PATH.write_text(
    json.dumps(
        {
            "alpha_seconds_per_meter": alpha,
            "implied_max_speed_kmh": 3.6 / alpha,
            "edge_bound_violations": violations,
            "od_pairs": len(result),
        },
        indent=2,
    ),
    encoding="utf-8",
)

print("\n=== OPTIMALITY ===")
print("OD pairs:", len(result))
print(
    "Max absolute cost difference (s):",
    result["cost_difference_s"].abs().max(),
)
print(
    "Identical node paths:",
    int(result["same_node_path"].sum()),
    "/",
    len(result),
)

print("\n=== SEARCH EFFICIENCY ===")

columns = [
    "dijkstra_expanded",
    "astar_expanded",
    "expansion_reduction_pct",
    "dijkstra_ms",
    "astar_ms",
    "runtime_reduction_pct",
]

print(
    result[columns]
    .agg(["mean", "median", "min", "max"])
    .round(3)
    .to_string()
)

print("\n=== AGGREGATE COMPARISON ===")
print(
    "Total Dijkstra expansions:",
    int(result["dijkstra_expanded"].sum()),
)
print(
    "Total A* expansions:",
    int(result["astar_expanded"].sum()),
)
print(
    "Total expansion reduction (%):",
    round(
        100
        * (
            1
            - result["astar_expanded"].sum()
            / result["dijkstra_expanded"].sum()
        ),
        3,
    ),
)
print(
    "Total Dijkstra runtime (ms):",
    round(result["dijkstra_ms"].sum(), 3),
)
print(
    "Total A* runtime (ms):",
    round(result["astar_ms"].sum(), 3),
)

print("\nSaved:", OUTPUT_PATH)
print("Saved metadata:", META_PATH)
