from src.paths import ROOT, DATA, benchmark_path, output_path
from pathlib import Path
from time import perf_counter_ns
import hashlib
import heapq
import gc

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeRegressor

GRAPH = ROOT / "data/graphs/porto_drive_conservative_refit.graphml"
LANDMARKS = DATA / "adaptive_alt_landmarks.csv"
TRAIN = DATA / "ml_alt_budget_groupkfold.parquet"
TEST = DATA / "porto_heldout_600_od.parquet"
OUTPUT = output_path("porto_heldout_alt_benchmark.parquet")

WEIGHT = "osm_travel_time_s"
REPEATS = 3
SEED = 20261016

FEATURE_COLS = [
    "air_distance_m", "alt_best", "alt_second",
    "alt_fourth", "alt_eighth", "alt_mean",
    "alt_std", "alt_gap_1_2", "alt_gap_2_4",
    "alt_gap_4_8", "alt_positive_count",
    "abs_delta_lat", "abs_delta_lon",
]

STRATEGIES = [
    "alt2", "alt4", "distance_gate", "ml_tree"
]

def elapsed_ms(start):
    return (perf_counter_ns() - start) / 1e6

def new_model():
    return DecisionTreeRegressor(
        max_depth=2,
        min_samples_leaf=5,
        random_state=42,
    )

def compile_node(tree, node=0):
    left = int(tree.children_left[node])

    if left == -1:
        return ("leaf", float(tree.value[node, 0, 0]))

    return (
        "split",
        int(tree.feature[node]),
        float(tree.threshold[node]),
        compile_node(tree, left),
        compile_node(tree, int(tree.children_right[node])),
    )

def predict_rule(rule, x):
    while rule[0] == "split":
        rule = (
            rule[3]
            if x[rule[1]] <= rule[2]
            else rule[4]
        )
    return rule[1]

print("=== LOAD GRAPH ===")

G = ox.load_graphml(GRAPH)

H = nx.DiGraph()
H.add_nodes_from(G.nodes(data=True))

for u, v, data in G.edges(data=True):
    if u == v:
        continue

    w = float(data[WEIGHT])

    if not H.has_edge(u, v) or w < H[u][v]["weight"]:
        H.add_edge(u, v, weight=w)

nodes = list(H.nodes)
node_idx = {n: i for i, n in enumerate(nodes)}
lookup = {str(n): n for n in nodes}
N = len(nodes)

lat = np.array(
    [float(H.nodes[n]["y"]) for n in nodes]
)
lon = np.array(
    [float(H.nodes[n]["x"]) for n in nodes]
)

adj = [
    [
        (node_idx[v], float(data["weight"]))
        for v, data in H[n].items()
    ]
    for n in nodes
]

print("Nodes:", N)
print("Edges:", H.number_of_edges())

print("\n=== ALT PREPROCESSING ===")

pre_start = perf_counter_ns()

lm_csv = pd.read_csv(LANDMARKS)
landmarks = [
    lookup[str(x)]
    for x in lm_csv["node_id"].tolist()
]

assert len(landmarks) == 32

F = np.full((32, N), np.inf)
B = np.full((32, N), np.inf)

reverse = H.reverse(copy=False)

for j, landmark in enumerate(landmarks):
    forward_dist = nx.single_source_dijkstra_path_length(
        H, landmark, weight="weight"
    )
    backward_dist = nx.single_source_dijkstra_path_length(
        reverse, landmark, weight="weight"
    )

    for n, d in forward_dist.items():
        F[j, node_idx[n]] = d

    for n, d in backward_dist.items():
        B[j, node_idx[n]] = d

preprocessing_ms = elapsed_ms(pre_start)

print("ALT preprocessing (ms):", round(preprocessing_ms, 3))

def haversine(s, t):
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
        12742000
        * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    )

def source_scores(s, t):
    a = np.zeros(32, dtype=np.float64)
    b = np.zeros(32, dtype=np.float64)

    mask = np.isfinite(F[:, s]) & np.isfinite(F[:, t])
    np.subtract(
        F[:, t], F[:, s], out=a, where=mask
    )

    mask = np.isfinite(B[:, s]) & np.isfinite(B[:, t])
    np.subtract(
        B[:, s], B[:, t], out=b, where=mask
    )

    return np.maximum(np.maximum(a, b), 0.0)

def build_heuristic(chosen, target):
    h = np.zeros(N, dtype=np.float64)

    for j in chosen:
        a = np.zeros(N, dtype=np.float64)
        b = np.zeros(N, dtype=np.float64)

        mask = (
            np.isfinite(F[j])
            & np.isfinite(F[j, target])
        )

        np.subtract(
            F[j, target], F[j],
            out=a, where=mask
        )

        mask = (
            np.isfinite(B[j])
            & np.isfinite(B[j, target])
        )

        np.subtract(
            B[j], B[j, target],
            out=b, where=mask
        )

        np.maximum(h, a, out=h)
        np.maximum(h, b, out=h)

    return h

def astar(s, t, h):
    g = np.full(N, np.inf)
    g[s] = 0.0

    heap = [(float(h[s]), -0.0, s)]
    expanded = 0

    while heap:
        _, neg_g, u = heapq.heappop(heap)
        current = -neg_g

        if current > g[u] + 1e-9:
            continue

        expanded += 1

        if u == t:
            return current, expanded

        for v, weight in adj[u]:
            candidate = current + weight

            if candidate < g[v] - 1e-9:
                g[v] = candidate
                heapq.heappush(
                    heap,
                    (
                        candidate + float(h[v]),
                        -candidate,
                        v,
                    )
                )

    return np.inf, expanded

def make_features(s, t, scores, distance):
    ordered = np.sort(scores)[::-1]

    return np.array([
        distance,
        ordered[0],
        ordered[1],
        ordered[3],
        ordered[7],
        ordered.mean(),
        ordered.std(),
        ordered[0] - ordered[1],
        ordered[1] - ordered[3],
        ordered[3] - ordered[7],
        float(np.count_nonzero(ordered > 0)),
        abs(lat[t] - lat[s]),
        abs(lon[t] - lon[s]),
    ], dtype=np.float64)

print("\n=== TRAIN FINAL ML MODEL ===")

train = (
    pd.read_parquet(TRAIN)
    .sort_values("pair_id")
    .reset_index(drop=True)
)

X = train[FEATURE_COLS].to_numpy(dtype=np.float64)
y = train["saved_expansions"].to_numpy(dtype=np.float64)
groups = train["destination"].astype(str).to_numpy()

saved_time = (
    train["alt2_time_ms"].to_numpy()
    - train["alt4_time_ms"].to_numpy()
)

# Cross-fitted predictions on the training set
# for threshold selection. No held-out OD used.
oof = np.full(len(train), np.nan)

for tr, va in GroupKFold(n_splits=5).split(
    X, y, groups
):
    m = new_model()
    m.fit(X[tr], y[tr])
    oof[va] = m.predict(X[va])

assert np.isfinite(oof).all()

candidates = [0, 10, 25, 50, 100, 150, 200, 300, 500]

gains = [
    float(np.where(
        oof >= threshold, saved_time, 0.0
    ).sum())
    for threshold in candidates
]

ml_threshold = candidates[int(np.argmax(gains))]

final_model = new_model()
final_model.fit(X, y)

rule = compile_node(final_model.tree_)

# Distance threshold fixed on original GPS OD.
# The training parquet contains the original
# air-distance feature for those 142 requests.
distance_threshold = float(
    train["air_distance_m"].quantile(0.75)
)

print("Training OD:", len(train))
print("Training destinations:", len(set(groups)))
print("ML threshold:", ml_threshold)
print("Distance threshold (m):", round(distance_threshold, 3))
print("Tree nodes:", final_model.tree_.node_count)
print("Compiled tree:", rule)

print("\n=== VERIFY HELD-OUT SAMPLE ===")

test = (
    pd.read_parquet(TEST)
    .sort_values("pair_id")
    .reset_index(drop=True)
)

assert len(test) == 600

content = "\n".join(
    f"{r.origin},{r.destination}"
    for r in test.itertuples()
)

digest = hashlib.sha256(
    content.encode("utf-8")
).hexdigest()

expected_digest = (
    "d93bc12a37bb62b8a22a4fae55a6c5bd3bd863"
    "aa1a705f4b413cd47204d14b1e"
)

assert digest == expected_digest, (
    "Held-out OD checksum mismatch"
)

pairs = [
    (
        node_idx[lookup[str(r.origin)]],
        node_idx[lookup[str(r.destination)]],
    )
    for r in test.itertuples()
]

print("Held-out OD:", len(pairs))
print("SHA256:", digest)

# Dijkstra distances are outside the timed region.
# They are used exclusively for optimality checks.
print("\n=== DIJKSTRA REFERENCE ===")

optimal = np.array([
    nx.shortest_path_length(
        H, nodes[s], nodes[t], weight="weight"
    )
    for s, t in pairs
], dtype=np.float64)

print("Reference paths:", len(optimal))

# Warm-up.
for i in range(min(20, len(pairs))):
    s, t = pairs[i]
    scores = source_scores(s, t)
    chosen = np.argsort(-scores, kind="stable")[:2]
    h = build_heuristic(chosen, t)
    cost, _ = astar(s, t, h)

    assert np.isclose(
        cost, optimal[i], rtol=1e-8, atol=1e-6
    )

print("\n=== END-TO-END BENCHMARK ===")

rng = np.random.default_rng(SEED)
records = []

gc.collect()
gc.disable()

try:
    for repeat in range(REPEATS):
        for i in rng.permutation(len(pairs)):
            s, t = pairs[i]

            for strategy in rng.permutation(STRATEGIES):
                start = perf_counter_ns()

                # Every strategy computes source scores,
                # because all use source-top-k selection.
                scores = source_scores(s, t)

                decision_start = perf_counter_ns()

                if strategy == "alt2":
                    k = 2

                elif strategy == "alt4":
                    k = 4

                elif strategy == "distance_gate":
                    d = haversine(s, t)
                    k = (
                        4 if d >= distance_threshold
                        else 2
                    )

                else:
                    d = haversine(s, t)
                    features = make_features(
                        s, t, scores, d
                    )
                    prediction = predict_rule(
                        rule, features
                    )
                    k = (
                        4 if prediction >= ml_threshold
                        else 2
                    )

                decision_ms = elapsed_ms(decision_start)

                # Rank landmarks after strategy decision.
                chosen = np.argsort(
                    -scores, kind="stable"
                )[:k]

                selection_ms = elapsed_ms(start)

                preparation_start = perf_counter_ns()
                h = build_heuristic(chosen, t)
                preparation_ms = elapsed_ms(
                    preparation_start
                )

                search_start = perf_counter_ns()
                cost, expanded = astar(s, t, h)
                search_ms = elapsed_ms(search_start)

                total_ms = elapsed_ms(start)

                if not np.isclose(
                    cost,
                    optimal[i],
                    rtol=1e-8,
                    atol=1e-6,
                ):
                    raise AssertionError(
                        f"Non-optimal path: "
                        f"pair={i}, strategy={strategy}, "
                        f"cost={cost}, optimal={optimal[i]}"
                    )

                records.append({
                    "repeat": repeat,
                    "pair_id": int(test.iloc[i]["pair_id"]),
                    "destination": str(
                        test.iloc[i]["destination"]
                    ),
                    "distance_bin": str(
                        test.iloc[i]["distance_bin"]
                    ),
                    "strategy": strategy,
                    "k": int(k),
                    "expanded": int(expanded),
                    "decision_ms": decision_ms,
                    "selection_ms": selection_ms,
                    "preparation_ms": preparation_ms,
                    "search_ms": search_ms,
                    "total_ms": total_ms,
                })

        print(
            f"Completed repeat {repeat + 1}/{REPEATS}",
            flush=True,
        )
finally:
    gc.enable()

result = pd.DataFrame(records)

OUTPUT.parent.mkdir(
    parents=True, exist_ok=True
)
result.to_parquet(OUTPUT, index=False)

print("\n=== HELD-OUT SUMMARY ===")

summary = (
    result.groupby(["repeat", "strategy"])
    ["total_ms"]
    .sum()
    .unstack()
)

print("Per-repeat total time (ms):")
print(summary.round(3).to_string())

print("\nMedian total time (ms):")
print(summary.median().round(3).to_string())

print("\n=== EXPANSIONS ===")

expanded = (
    result[result["repeat"] == 0]
    .groupby("strategy")["expanded"]
    .sum()
)

print(expanded.to_string())

print("\n=== CHOSEN K ===")

print(
    result[result["repeat"] == 0]
    .groupby(["strategy", "k"])
    .size()
    .to_string()
)

print("\n=== BY DISTANCE BIN ===")

bin_summary = (
    result.groupby([
        "repeat", "distance_bin", "strategy"
    ])["total_ms"]
    .sum()
    .reset_index()
    .groupby(["distance_bin", "strategy"])
    ["total_ms"]
    .median()
    .unstack()
)

print(bin_summary.round(3).to_string())

print("\n=== PAIRED DELTAS VS ALT-2 ===")

wide = result.pivot(
    index=["repeat", "pair_id", "destination"],
    columns="strategy",
    values="total_ms",
).reset_index()

for strategy in [
    "alt4", "distance_gate", "ml_tree"
]:
    delta = wide[strategy] - wide["alt2"]

    print(
        f"{strategy}: "
        f"mean={delta.mean():+.6f} ms/OD, "
        f"faster={int((delta < 0).sum())}, "
        f"slower={int((delta > 0).sum())}"
    )

print("\n=== OPTIMALITY ===")
print("All routes optimal: True")
print("Saved:", OUTPUT)
