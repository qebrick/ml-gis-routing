from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

GRAPH = Path(
    "data/graphs/porto_drive_conservative_refit.graphml"
)
LANDMARKS = Path(
    "data/processed/adaptive_alt_landmarks.csv"
)
BENCHMARK = Path(
    "data/processed/adaptive_alt_end_to_end_benchmark.parquet"
)
OD_PATH = Path(
    "data/processed/gps_od_route_comparison.parquet"
)
OUTPUT = Path(
    "data/processed/ml_alt_budget_groupkfold.parquet"
)

WEIGHT = "osm_travel_time_s"
SEED = 42

# ----------------------------------------------
# Graph and ALT distances
# ----------------------------------------------

G = ox.load_graphml(GRAPH)

H = nx.DiGraph()
H.add_nodes_from(G.nodes(data=True))

for u, v, data in G.edges(data=True):
    w = float(data[WEIGHT])
    if not H.has_edge(u, v) or w < H[u][v][WEIGHT]:
        H.add_edge(u, v, **{WEIGHT: w})

nodes = list(H.nodes)
idx = {n: i for i, n in enumerate(nodes)}
lookup = {str(n): n for n in nodes}

lm_ids = pd.read_csv(LANDMARKS)["node_id"].tolist()
landmarks = [lookup[str(x)] for x in lm_ids]

assert len(landmarks) == 32

F = np.full((32, len(nodes)), np.inf)
B = np.full((32, len(nodes)), np.inf)

R = H.reverse(copy=False)

for j, lm in enumerate(landmarks):
    for n, d in nx.single_source_dijkstra_path_length(
        H, lm, weight=WEIGHT
    ).items():
        F[j, idx[n]] = d

    for n, d in nx.single_source_dijkstra_path_length(
        R, lm, weight=WEIGHT
    ).items():
        B[j, idx[n]] = d

lat = np.array([float(H.nodes[n]["y"]) for n in nodes])
lon = np.array([float(H.nodes[n]["x"]) for n in nodes])

def air_distance(s, t):
    p1, p2 = np.radians([lat[s], lat[t]])
    dp = p2 - p1
    dl = np.radians(lon[t] - lon[s])
    a = (
        np.sin(dp / 2)**2
        + np.cos(p1)*np.cos(p2)*np.sin(dl / 2)**2
    )
    return float(
        12742000 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    )

def alt_scores(s, t):
    a = np.zeros(32)
    b = np.zeros(32)

    valid = np.isfinite(F[:, s]) & np.isfinite(F[:, t])
    np.subtract(F[:, t], F[:, s], out=a, where=valid)

    valid = np.isfinite(B[:, s]) & np.isfinite(B[:, t])
    np.subtract(B[:, s], B[:, t], out=b, where=valid)

    return np.maximum(np.maximum(a, b), 0.0)

# ----------------------------------------------
# Build pre-search features
# ----------------------------------------------

od = pd.read_parquet(OD_PATH).reset_index(drop=True)

rows = []

for pair_id, row in enumerate(od.itertuples()):
    s = idx[lookup[str(row.origin)]]
    t = idx[lookup[str(row.destination)]]

    scores = np.sort(alt_scores(s, t))[::-1]
    distance = air_distance(s, t)

    rows.append({
        "pair_id": pair_id,
        "destination": str(row.destination),
        "air_distance_m": distance,
        "alt_best": scores[0],
        "alt_second": scores[1],
        "alt_fourth": scores[3],
        "alt_eighth": scores[7],
        "alt_mean": scores.mean(),
        "alt_std": scores.std(),
        "alt_gap_1_2": scores[0] - scores[1],
        "alt_gap_2_4": scores[1] - scores[3],
        "alt_gap_4_8": scores[3] - scores[7],
        "alt_positive_count": int((scores > 0).sum()),
        "abs_delta_lat": abs(lat[t] - lat[s]),
        "abs_delta_lon": abs(lon[t] - lon[s]),
    })

features = pd.DataFrame(rows)

# ----------------------------------------------
# Aggregate benchmark to OD level
# ----------------------------------------------

bench = pd.read_parquet(BENCHMARK)

base = bench[
    bench["strategy"].isin([
        "adaptive_alt_2",
        "adaptive_alt_4",
        "distance_gated_alt",
    ])
]

timings = base.pivot_table(
    index="pair_id",
    columns="strategy",
    values="total_ms",
    aggfunc="mean",
)

expanded = base.pivot_table(
    index="pair_id",
    columns="strategy",
    values="expanded",
    aggfunc="first",
)

features["saved_expansions"] = (
    expanded["adaptive_alt_2"].to_numpy()
    - expanded["adaptive_alt_4"].to_numpy()
)

# Positive means ALT-4 is faster.
features["saved_time_ms"] = (
    timings["adaptive_alt_2"].to_numpy()
    - timings["adaptive_alt_4"].to_numpy()
)

# Check that pivot ordering matches pair_id.
assert np.array_equal(
    timings.index.to_numpy(),
    features["pair_id"].to_numpy()
)

feature_cols = [
    "air_distance_m",
    "alt_best",
    "alt_second",
    "alt_fourth",
    "alt_eighth",
    "alt_mean",
    "alt_std",
    "alt_gap_1_2",
    "alt_gap_2_4",
    "alt_gap_4_8",
    "alt_positive_count",
    "abs_delta_lat",
    "abs_delta_lon",
]

X = features[feature_cols].to_numpy()
y = features["saved_expansions"].to_numpy()
groups = features["destination"].to_numpy()

# ----------------------------------------------
# GroupKFold: train-only threshold selection
# ----------------------------------------------

cv = GroupKFold(n_splits=5)

pred = np.full(len(features), np.nan)
chosen = np.zeros(len(features), dtype=int)
folds = np.zeros(len(features), dtype=int)
thresholds = np.zeros(len(features))

candidate_thresholds = [
    0, 10, 25, 50, 100, 150, 200, 300, 500
]

for fold, (train, test) in enumerate(
    cv.split(X, y, groups), start=1
):
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=5,
        random_state=SEED,
        n_jobs=-1,
    )

    model.fit(X[train], y[train])

    train_pred = model.predict(X[train])
    test_pred = model.predict(X[test])

    # This threshold is chosen only on training OD.
    # Training predictions are in-sample, so this
    # is exploratory rather than fully nested CV.
    best_threshold = None
    best_gain = -np.inf

    for threshold in candidate_thresholds:
        use4 = train_pred >= threshold

        gain = np.where(
            use4,
            features.iloc[train]["saved_time_ms"].to_numpy(),
            0.0,
        ).sum()

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    pred[test] = test_pred
    chosen[test] = np.where(
        test_pred >= best_threshold, 4, 2
    )
    folds[test] = fold
    thresholds[test] = best_threshold

features["fold"] = folds
features["predicted_saved_expansions"] = pred
features["ml_k"] = chosen
features["train_selected_threshold"] = thresholds

features["ml_time_ms"] = np.where(
    chosen == 4,
    timings["adaptive_alt_4"].to_numpy(),
    timings["adaptive_alt_2"].to_numpy(),
)

features["alt2_time_ms"] = (
    timings["adaptive_alt_2"].to_numpy()
)
features["alt4_time_ms"] = (
    timings["adaptive_alt_4"].to_numpy()
)
features["distance_gate_time_ms"] = (
    timings["distance_gated_alt"].to_numpy()
)

features["ml_expanded"] = np.where(
    chosen == 4,
    expanded["adaptive_alt_4"].to_numpy(),
    expanded["adaptive_alt_2"].to_numpy(),
)

# ----------------------------------------------
# Results
# ----------------------------------------------

print("=== GROUPKFOLD COVERAGE ===")
print("OD:", len(features))
print("Unique destinations:", features["destination"].nunique())
print("ML chose k=4:", int((features["ml_k"] == 4).sum()))

print("\n=== FOLD THRESHOLDS ===")
print(
    features.groupby("fold")
    .agg(
        routes=("pair_id", "size"),
        threshold=("train_selected_threshold", "first"),
        k4=("ml_k", lambda s: int((s == 4).sum())),
    )
    .to_string()
)

print("\n=== OFFLINE RUNTIME COMPARISON ===")

for col in [
    "alt2_time_ms",
    "alt4_time_ms",
    "distance_gate_time_ms",
    "ml_time_ms",
]:
    print(
        f"{col:24s}",
        f"{features[col].sum():.4f} ms"
    )

print("\n=== ML VS ALT-2 ===")
delta = features["ml_time_ms"] - features["alt2_time_ms"]

print("Mean delta per OD (ms):", round(delta.mean(), 6))
print("ML faster OD:", int((delta < 0).sum()))
print("ML slower OD:", int((delta > 0).sum()))
print("ML equal OD:", int((delta == 0).sum()))

print("\n=== EXPANSIONS ===")
print("ALT-2:", int(expanded["adaptive_alt_2"].sum()))
print("ALT-4:", int(expanded["adaptive_alt_4"].sum()))
print("ML:", int(features["ml_expanded"].sum()))

print("\n=== PREDICTION DIAGNOSTICS ===")
print(
    "Spearman correlation:",
    round(
        features[
            "predicted_saved_expansions"
        ].corr(
            features["saved_expansions"],
            method="spearman",
        ),
        4,
    ),
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
features.to_parquet(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
