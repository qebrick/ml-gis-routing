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
    "data/processed/adaptive_alt_landmark_diagnostics.parquet"
)

SEED = 42
N_LANDMARKS = 32
N_CANDIDATES = 32
TOL = 1e-6

rng = np.random.default_rng(SEED)

# --------------------------------------------------
# Graph
# --------------------------------------------------

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

adj = [[] for _ in range(N)]

for u, v, attrs in H.edges(data=True):
    adj[idx[u]].append(
        (idx[v], attrs["weight"])
    )

# --------------------------------------------------
# Reproduce previous landmark selection
# --------------------------------------------------

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

extremes = [
    min(scc_nodes, key=lambda n: lon[n]),
    max(scc_nodes, key=lambda n: lon[n]),
    min(scc_nodes, key=lambda n: lat[n]),
    max(scc_nodes, key=lambda n: lat[n]),
]

landmarks = list(dict.fromkeys(extremes))

# Farthest-point sampling using geographical
# coordinates. This must match the landmark
# selection used in previous experiments.
xy = np.array([
    [lat[n], lon[n]]
    for n in scc_nodes
])

selected_xy = np.array([
    [lat[n], lon[n]]
    for n in landmarks
])

min_dist2 = np.full(
    len(scc_nodes),
    np.inf,
)

for point in selected_xy:
    dist2 = np.sum(
        (xy - point) ** 2,
        axis=1,
    )
    min_dist2 = np.minimum(
        min_dist2, dist2
    )

while len(landmarks) < N_LANDMARKS:
    choice = int(np.argmax(min_dist2))
    landmark = scc_nodes[choice]

    if landmark in landmarks:
        min_dist2[choice] = -1.0
        continue

    landmarks.append(landmark)

    point = xy[choice]
    dist2 = np.sum(
        (xy - point) ** 2,
        axis=1,
    )

    min_dist2 = np.minimum(
        min_dist2, dist2
    )

print("=== LANDMARKS ===")
print("Graph nodes:", N)
print("Graph edges:", H.number_of_edges())
print("SCC nodes:", len(scc))
print("Landmarks:", len(landmarks))
print("First four:", landmarks[:4])

# --------------------------------------------------
# Directed ALT preprocessing
# --------------------------------------------------

forward = np.full(
    (N_LANDMARKS, N),
    np.inf,
)

backward = np.full(
    (N_LANDMARKS, N),
    np.inf,
)

reverse_H = H.reverse(copy=False)

t0 = perf_counter()

for j, landmark in enumerate(landmarks):
    for node, distance in nx.single_source_dijkstra_path_length(
        H,
        landmark,
        weight="weight",
    ).items():
        forward[j, idx[node]] = distance

    for node, distance in nx.single_source_dijkstra_path_length(
        reverse_H,
        landmark,
        weight="weight",
    ).items():
        backward[j, idx[node]] = distance

preprocessing_s = perf_counter() - t0

print(
    "Preprocessing (s):",
    round(preprocessing_s, 3),
)

# --------------------------------------------------
# ALT contribution from each landmark
# --------------------------------------------------

def contributions(source, target):
    """
    ALT directed lower bounds for every landmark
    at the source node.

    h_L(s,t) = max(
        d(L,t)-d(L,s),
        d(s,L)-d(t,L),
        0
    )
    """
    first = (
        forward[:, target]
        - forward[:, source]
    )

    second = (
        backward[:, source]
        - backward[:, target]
    )

    # In disconnected regions inf-inf could
    # produce nan. Treat invalid bounds as 0.
    first = np.where(
        np.isfinite(first),
        first,
        0.0,
    )

    second = np.where(
        np.isfinite(second),
        second,
        0.0,
    )

    return np.maximum.reduce([
        first,
        second,
        np.zeros(N_LANDMARKS),
    ])


def target_heuristic(target, chosen):
    """
    Compute admissible ALT lower bound for all
    graph nodes using only selected landmarks.
    """
    chosen = np.asarray(chosen, dtype=int)

    h = np.zeros(N, dtype=float)

    for j in chosen:
        first = (
            forward[j, target]
            - forward[j]
        )

        second = (
            backward[j]
            - backward[j, target]
        )

        first = np.where(
            np.isfinite(first),
            first,
            0.0,
        )

        second = np.where(
            np.isfinite(second),
            second,
            0.0,
        )

        h = np.maximum(h, first)
        h = np.maximum(h, second)

    h[target] = 0.0

    return h


# --------------------------------------------------
# A* with high-g tie-breaking
# --------------------------------------------------

def astar(source, target, heuristic):
    best = {source: 0.0}
    serial = count()

    heap = [
        (
            float(heuristic[source]),
            0.0,
            next(serial),
            0.0,
            source,
        )
    ]

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
                neighbor,
                np.inf,
            ):
                best[neighbor] = new_g

                heappush(
                    heap,
                    (
                        new_g
                        + float(heuristic[neighbor]),
                        -new_g,
                        next(serial),
                        new_g,
                        neighbor,
                    ),
                )

    raise RuntimeError("No path")


# --------------------------------------------------
# GPS-derived OD
# --------------------------------------------------

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

print("\n=== COVERAGE ===")
print("OD pairs:", len(pairs))
print(
    "Unique targets:",
    len({p[3] for p in pairs}),
)

# --------------------------------------------------
# Candidate subsets
# --------------------------------------------------

candidate_subsets = []

# Four consecutive landmarks from each of
# eight blocks.
for start in range(0, N_LANDMARKS, 4):
    candidate_subsets.append(
        tuple(range(start, start + 4))
    )

# Additional random subsets, deterministic.
while len(candidate_subsets) < N_CANDIDATES:
    subset = tuple(sorted(
        rng.choice(
            N_LANDMARKS,
            size=4,
            replace=False,
        ).tolist()
    ))

    if subset not in candidate_subsets:
        candidate_subsets.append(subset)

print(
    "Oracle candidate subsets:",
    len(candidate_subsets),
)

# --------------------------------------------------
# Benchmark
# --------------------------------------------------

records = []

for pair_id, (
    source,
    target,
    origin,
    destination,
) in enumerate(pairs):

    reference = nx.shortest_path_length(
        H,
        origin,
        destination,
        weight="weight",
    )

    source_scores = contributions(
        source,
        target,
    )

    source_top_4 = tuple(
        np.argsort(-source_scores)[:4].tolist()
    )

    random_4 = tuple(sorted(
        rng.choice(
            N_LANDMARKS,
            size=4,
            replace=False,
        ).tolist()
    ))

    selections = {
        "fixed_4": tuple(range(4)),
        "random_4": random_4,
        "source_top_4": source_top_4,
        "all_32": tuple(range(32)),
    }

    for strategy, chosen in selections.items():
        h = target_heuristic(
            target,
            chosen,
        )

        cost, expanded = astar(
            source,
            target,
            h,
        )

        if abs(cost - reference) > TOL:
            raise RuntimeError(
                f"Optimality failure: {strategy}"
            )

        records.append({
            "pair_id": pair_id,
            "origin": origin,
            "destination": destination,
            "strategy": strategy,
            "expanded": expanded,
            "selected_landmarks": str(chosen),
        })

    # Diagnostic oracle over 32 candidate subsets.
    oracle_results = []

    for candidate_id, chosen in enumerate(
        candidate_subsets
    ):
        h = target_heuristic(
            target,
            chosen,
        )

        cost, expanded = astar(
            source,
            target,
            h,
        )

        if abs(cost - reference) > TOL:
            raise RuntimeError(
                "Oracle candidate nonoptimal"
            )

        oracle_results.append(
            (expanded, candidate_id, chosen)
        )

    best_expanded, best_id, best_subset = min(
        oracle_results,
        key=lambda x: x[0],
    )

    records.append({
        "pair_id": pair_id,
        "origin": origin,
        "destination": destination,
        "strategy": "oracle_4",
        "expanded": best_expanded,
        "selected_landmarks": str(best_subset),
    })

    if (pair_id + 1) % 20 == 0:
        print(
            f"Completed {pair_id + 1}/{len(pairs)}"
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

# --------------------------------------------------
# Summary
# --------------------------------------------------

summary = (
    result.groupby("strategy")
    .agg(
        routes=("pair_id", "size"),
        total_expanded=("expanded", "sum"),
        mean_expanded=("expanded", "mean"),
        median_expanded=("expanded", "median"),
    )
    .reset_index()
)

fixed = int(
    summary.loc[
        summary["strategy"] == "fixed_4",
        "total_expanded",
    ].iloc[0]
)

summary["change_vs_fixed_pct"] = (
    100
    * (
        summary["total_expanded"] / fixed
        - 1
    )
)

print("\n=== STRATEGY SUMMARY ===")
print(
    summary.sort_values(
        "total_expanded"
    ).round(3).to_string(index=False)
)

wide = result.pivot(
    index="pair_id",
    columns="strategy",
    values="expanded",
)

print("\n=== SOURCE_TOP_4 VS FIXED_4 ===")

delta = (
    wide["source_top_4"]
    - wide["fixed_4"]
)

print(
    "Better:", int((delta < 0).sum())
)
print(
    "Worse:", int((delta > 0).sum())
)
print(
    "Equal:", int((delta == 0).sum())
)

print("\n=== ORACLE_4 VS FIXED_4 ===")

delta = (
    wide["oracle_4"]
    - wide["fixed_4"]
)

print(
    "Better:", int((delta < 0).sum())
)
print(
    "Worse:", int((delta > 0).sum())
)
print(
    "Equal:", int((delta == 0).sum())
)

print("\n=== OPTIMALITY ===")
print("All routes matched Dijkstra: True")
print("\nSaved:", OUTPUT_PATH)
