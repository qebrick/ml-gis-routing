from pathlib import Path
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
    "data/processed/alt_heuristic_diagnostics.parquet"
)

WEIGHT = "osm_travel_time_s"
EARTH_RADIUS_M = 6371008.8

G = ox.load_graphml(GRAPH_PATH)
od = pd.read_parquet(OD_PATH)

H = nx.DiGraph()
H.add_nodes_from(G.nodes)

for u, v, key, attrs in G.edges(keys=True, data=True):
    w = float(attrs[WEIGHT])

    if (
        not H.has_edge(u, v)
        or w < H[u][v]["weight"]
    ):
        H.add_edge(u, v, weight=w)

coords = {
    node: (
        radians(float(attrs["y"])),
        radians(float(attrs["x"])),
    )
    for node, attrs in G.nodes(data=True)
}

largest_scc = max(
    nx.strongly_connected_components(H),
    key=len,
)

scc_nodes = list(largest_scc)

landmarks = []

for axis, reverse in [
    (0, False),
    (0, True),
    (1, False),
    (1, True),
]:
    node = sorted(
        scc_nodes,
        key=lambda n: coords[n][axis],
        reverse=reverse,
    )[0]

    if node not in landmarks:
        landmarks.append(node)

reversed_graph = H.reverse(copy=False)

forward = {}
backward = {}

for landmark in landmarks:
    forward[landmark] = (
        nx.single_source_dijkstra_path_length(
            H,
            landmark,
            weight="weight",
        )
    )

    backward[landmark] = (
        nx.single_source_dijkstra_path_length(
            reversed_graph,
            landmark,
            weight="weight",
        )
    )


def alt_heuristic(node, target):
    bound = 0.0

    for landmark in landmarks:
        from_l = forward[landmark]
        to_l = backward[landmark]

        if node in from_l and target in from_l:
            bound = max(
                bound,
                from_l[target] - from_l[node],
            )

        if node in to_l and target in to_l:
            bound = max(
                bound,
                to_l[node] - to_l[target],
            )

    return max(0.0, bound)


# Use unique GPS-derived destinations.
targets = sorted(
    set(int(x) for x in od["destination"])
)

records = []

print("=== DATA COVERAGE ===")
print("Graph nodes:", H.number_of_nodes())
print("Graph edges:", H.number_of_edges())
print("Landmarks:", landmarks)
print("Unique GPS destinations:", len(targets))

for target_index, target in enumerate(targets, 1):
    # Distances from every reachable node TO target.
    true_distances = (
        nx.single_source_dijkstra_path_length(
            reversed_graph,
            target,
            weight="weight",
        )
    )

    for node, true_cost in true_distances.items():
        if true_cost <= 0:
            continue

        alt_cost = alt_heuristic(node, target)

        gap = true_cost - alt_cost

        if gap < -1e-6:
            raise RuntimeError(
                f"Inadmissible heuristic: "
                f"{node}->{target}, gap={gap}"
            )

        gap = max(0.0, gap)

        records.append({
            "node": node,
            "target": target,
            "true_cost_s": true_cost,
            "alt_cost_s": alt_cost,
            "gap_s": gap,
            "coverage_ratio": (
                alt_cost / true_cost
            ),
            "relative_gap": (
                gap / true_cost
            ),
        })

    if (
        target_index % 20 == 0
        or target_index == len(targets)
    ):
        print(
            f"Processed targets: "
            f"{target_index}/{len(targets)}"
        )

result = pd.DataFrame(records)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result.to_parquet(
    OUTPUT_PATH,
    index=False,
)

print("\n=== ALT HEURISTIC QUALITY ===")

print(
    result[
        [
            "true_cost_s",
            "alt_cost_s",
            "gap_s",
            "coverage_ratio",
            "relative_gap",
        ]
    ]
    .describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    )
    .round(4)
    .to_string()
)

print("\n=== EXACTNESS ===")

print(
    "Exact within 1 second (%):",
    round(
        100 * (result["gap_s"] <= 1).mean(),
        3,
    ),
)

print(
    "Exact within 5 seconds (%):",
    round(
        100 * (result["gap_s"] <= 5).mean(),
        3,
    ),
)

print(
    "Coverage >= 90% (%):",
    round(
        100
        * (result["coverage_ratio"] >= 0.9).mean(),
        3,
    ),
)

print("\n=== QUALITY BY TRUE TRAVEL TIME ===")

bins = [0, 30, 60, 120, 300, 600, float("inf")]

labels = [
    "0-30s",
    "30-60s",
    "60-120s",
    "120-300s",
    "300-600s",
    "600s+",
]

result["time_bin"] = pd.cut(
    result["true_cost_s"],
    bins=bins,
    labels=labels,
    right=False,
)

by_bin = (
    result.groupby(
        "time_bin",
        observed=True,
    )
    .agg(
        pairs=("gap_s", "size"),
        mean_gap_s=("gap_s", "mean"),
        median_gap_s=("gap_s", "median"),
        mean_coverage=("coverage_ratio", "mean"),
        median_coverage=("coverage_ratio", "median"),
    )
    .reset_index()
)

print(
    by_bin.round(4).to_string(index=False)
)

print("\nSaved:", OUTPUT_PATH)
