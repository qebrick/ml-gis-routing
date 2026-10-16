from pathlib import Path
from heapq import heappush, heappop
from itertools import count
from time import perf_counter

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold

GRAPH_PATH = Path(
    "data/graphs/porto_drive_conservative_refit.graphml"
)
DIAG_PATH = Path(
    "data/processed/alt_heuristic_diagnostics.parquet"
)
OD_PATH = Path(
    "data/processed/gps_od_route_comparison.parquet"
)
OUTPUT_PATH = Path(
    "data/processed/ml_tiebreak_groupkfold.parquet"
)

N_SPLITS = 5
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

lat = np.array([
    float(G.nodes[node]["y"]) for node in nodes
])

lon = np.array([
    float(G.nodes[node]["x"]) for node in nodes
])

diagnostics = pd.read_parquet(DIAG_PATH)
od = pd.read_parquet(OD_PATH).copy()

# Keep OD pairs whose destination has diagnostics.
available_targets = set(
    diagnostics["target"].astype(int)
)

od = od[
    od["destination"].astype(int).isin(
        available_targets
    )
].copy()

od["destination"] = od["destination"].astype(int)
od["origin"] = od["origin"].astype(int)

def add_features(frame):
    frame = frame.copy()

    ni = frame["node"].map(
        lambda n: idx[int(n)]
    ).to_numpy()

    ti = frame["target"].map(
        lambda n: idx[int(n)]
    ).to_numpy()

    frame["node_lat"] = lat[ni]
    frame["node_lon"] = lon[ni]
    frame["target_lat"] = lat[ti]
    frame["target_lon"] = lon[ti]

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


diagnostics = add_features(diagnostics)

# Split unique destinations, rather than individual
# node-target rows.
unique_targets = np.array(
    sorted(od["destination"].unique())
)

splitter = GroupKFold(n_splits=N_SPLITS)


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


records = []
fold_stats = []

print("=== DATA COVERAGE ===")
print("OD pairs:", len(od))
print("Unique OD targets:", len(unique_targets))
print("Diagnostic rows:", len(diagnostics))

for fold, (train_ti, test_ti) in enumerate(
    splitter.split(
        unique_targets,
        groups=unique_targets,
    ),
    start=1,
):
    train_targets = set(
        unique_targets[train_ti]
    )
    test_targets = set(
        unique_targets[test_ti]
    )

    assert train_targets.isdisjoint(test_targets)

    train = diagnostics[
        ~diagnostics["target"].isin(test_targets)
    ]

    fold_od = od[
        od["destination"].isin(test_targets)
    ]

    model = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="MAE",
        random_seed=SEED,
        verbose=False,
        thread_count=-1,
    )

    t0 = perf_counter()
    model.fit(train[FEATURES], train["gap_s"])
    training_s = perf_counter() - t0

    target_arrays = {}
    inference_ms = 0.0

    for target_node in sorted(
        fold_od["destination"].unique()
    ):
        target_node = int(target_node)
        target_idx = idx[target_node]

        rows = diagnostics[
            diagnostics["target"] == target_node
        ]

        safe = np.zeros(N, dtype=float)

        positions = rows["node"].map(
            lambda n: idx[int(n)]
        ).to_numpy()

        safe[positions] = (
            rows["alt_cost_s"].to_numpy()
        )

        safe[target_idx] = 0.0

        # Full-graph inference; feature preparation
        # is included in the measured time.
        t0 = perf_counter()

        features = pd.DataFrame({
            "alt_cost_s": safe,
            "node_lat": lat,
            "node_lon": lon,
            "target_lat": lat[target_idx],
            "target_lon": lon[target_idx],
        })

        features["delta_lat"] = (
            features["target_lat"]
            - features["node_lat"]
        )
        features["delta_lon"] = (
            features["target_lon"]
            - features["node_lon"]
        )
        features["abs_delta_lat"] = (
            features["delta_lat"].abs()
        )
        features["abs_delta_lon"] = (
            features["delta_lon"].abs()
        )

        predicted_gap = np.maximum(
            model.predict(features[FEATURES]),
            0.0,
        )

        ml = safe + predicted_gap
        ml[target_idx] = 0.0

        inference_ms += (
            perf_counter() - t0
        ) * 1000

        target_arrays[target_node] = (safe, ml)

    for pair_id, row in fold_od.iterrows():
        origin = int(row["origin"])
        destination = int(row["destination"])

        source = idx[origin]
        target = idx[destination]

        safe, ml = target_arrays[destination]

        reference = nx.shortest_path_length(
            H,
            origin,
            destination,
            weight="weight",
        )

        outputs = {}

        for strategy in ["fifo", "high_g", "ml"]:
            t0 = perf_counter()

            cost, expanded = astar(
                source, target, safe, ml, strategy
            )

            runtime_ms = (
                perf_counter() - t0
            ) * 1000

            if abs(cost - reference) > TOL:
                raise RuntimeError(
                    f"Optimality failure: "
                    f"fold={fold}, strategy={strategy}"
                )

            outputs[strategy] = (
                expanded, runtime_ms
            )

        records.append({
            "fold": fold,
            "pair_id": int(pair_id),
            "origin": origin,
            "destination": destination,
            "fifo_expanded": outputs["fifo"][0],
            "high_g_expanded": outputs["high_g"][0],
            "ml_expanded": outputs["ml"][0],
            "fifo_ms": outputs["fifo"][1],
            "high_g_ms": outputs["high_g"][1],
            "ml_ms": outputs["ml"][1],
        })

    fold_stats.append({
        "fold": fold,
        "train_targets": len(train_targets),
        "test_targets": len(test_targets),
        "test_routes": len(fold_od),
        "training_s": training_s,
        "inference_ms": inference_ms,
    })

    print(
        f"Fold {fold}/{N_SPLITS}: "
        f"{len(fold_od)} routes, "
        f"{len(test_targets)} targets, "
        f"training {training_s:.1f}s"
    )

result = pd.DataFrame(records)
stats = pd.DataFrame(fold_stats)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result.to_parquet(OUTPUT_PATH, index=False)

print("\n=== FOLD SUMMARY ===")

fold_summary = (
    result.groupby("fold")
    .agg(
        routes=("pair_id", "size"),
        fifo=("fifo_expanded", "sum"),
        high_g=("high_g_expanded", "sum"),
        ml=("ml_expanded", "sum"),
    )
    .reset_index()
)

fold_summary["ml_vs_high_g_pct"] = (
    100 * (
        fold_summary["ml"]
        / fold_summary["high_g"]
        - 1
    )
)

print(
    fold_summary.round(3).to_string(index=False)
)

print("\n=== POOLED OUT-OF-FOLD RESULTS ===")

totals = {
    strategy: int(
        result[f"{strategy}_expanded"].sum()
    )
    for strategy in ["fifo", "high_g", "ml"]
}

for strategy, total in totals.items():
    print(strategy, "expanded:", total)

print(
    "ML vs FIFO (%):",
    round(
        100 * (
            totals["ml"] / totals["fifo"] - 1
        ),
        3,
    ),
)

print(
    "ML vs HIGH_G (%):",
    round(
        100 * (
            totals["ml"] / totals["high_g"] - 1
        ),
        3,
    ),
)

delta = (
    result["ml_expanded"]
    - result["high_g_expanded"]
)

print("\n=== ML VS HIGH_G BY ROUTE ===")
print("ML better:", int((delta < 0).sum()))
print("ML worse:", int((delta > 0).sum()))
print("Equal:", int((delta == 0).sum()))

print("\n=== COMPUTATIONAL COST ===")
print(
    "Total model training (s):",
    round(stats["training_s"].sum(), 3),
)
print(
    "Total ML inference (ms):",
    round(stats["inference_ms"].sum(), 3),
)

for strategy in ["fifo", "high_g", "ml"]:
    print(
        strategy,
        "single-run search (ms):",
        round(result[f"{strategy}_ms"].sum(), 3),
    )

print("\n=== OPTIMALITY ===")
print("All route costs matched Dijkstra: True")

print("\nSaved:", OUTPUT_PATH)
