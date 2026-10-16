from pathlib import Path
from time import perf_counter_ns
import gc

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeRegressor

DATA = Path("data/processed/ml_alt_budget_groupkfold.parquet")
VALID = Path("data/processed/ml_alt_compact_tree_validation.parquet")
OUTPUT = Path("data/processed/ml_alt_compiled_rule_benchmark.parquet")

cols = [
    "air_distance_m", "alt_best", "alt_second",
    "alt_fourth", "alt_eighth", "alt_mean",
    "alt_std", "alt_gap_1_2", "alt_gap_2_4",
    "alt_gap_4_8", "alt_positive_count",
    "abs_delta_lat", "abs_delta_lon",
]

df = pd.read_parquet(DATA).sort_values("pair_id").reset_index(drop=True)
valid = pd.read_parquet(VALID)
valid = valid[valid["depth"] == 2].set_index("pair_id")
valid = valid.loc[df["pair_id"]].reset_index()

X = df[cols].to_numpy(dtype=np.float64)
y = df["saved_expansions"].to_numpy(dtype=np.float64)
groups = df["destination"].astype(str).to_numpy()
saved_time = (
    df["alt2_time_ms"].to_numpy()
    - df["alt4_time_ms"].to_numpy()
)

def new_model():
    return DecisionTreeRegressor(
        max_depth=2, min_samples_leaf=5, random_state=42
    )

def compile_tree(model):
    t = model.tree_

    def compile_node(node):
        left = int(t.children_left[node])
        if left == -1:
            return ("leaf", float(t.value[node, 0, 0]))

        return (
            "split",
            int(t.feature[node]),
            float(t.threshold[node]),
            compile_node(left),
            compile_node(int(t.children_right[node])),
        )

    return compile_node(0)

def predict_compiled(rule, x):
    # Generic compiled structure: no sklearn or NumPy tree access.
    while rule[0] == "split":
        rule = rule[3] if x[rule[1]] <= rule[2] else rule[4]
    return rule[1]

def predict_raw(model, x):
    t = model.tree_
    node = 0
    while t.children_left[node] != -1:
        node = (
            t.children_left[node]
            if x[t.feature[node]] <= t.threshold[node]
            else t.children_right[node]
        )
    return float(t.value[node, 0, 0])

models = {}
rules = {}
thresholds = {}

for fold, (train, test) in enumerate(
    GroupKFold(n_splits=5).split(X, y, groups), 1
):
    oof = np.full(len(train), np.nan)

    for a, b in GroupKFold(n_splits=4).split(
        X[train], y[train], groups[train]
    ):
        m = new_model()
        m.fit(X[train[a]], y[train[a]])
        oof[b] = m.predict(X[train[b]])

    candidates = [0, 10, 25, 50, 100, 150, 200, 300, 500]
    threshold = max(
        candidates,
        key=lambda z: np.where(
            oof >= z, saved_time[train], 0.0
        ).sum()
    )

    # Reproduce previous tie-breaking: first threshold wins.
    gains = [
        np.where(oof >= z, saved_time[train], 0.0).sum()
        for z in candidates
    ]
    threshold = candidates[int(np.argmax(gains))]

    model = new_model()
    model.fit(X[train], y[train])

    models[fold] = model
    rules[fold] = compile_tree(model)
    thresholds[fold] = threshold

    for i in test:
        direct = predict_compiled(rules[fold], X[i])
        original = float(model.predict(X[i:i+1])[0])

        assert np.isclose(direct, original)

        chosen = 4 if direct >= threshold else 2
        assert chosen == int(valid.iloc[i]["chosen_k"])

    print(
        f"Fold {fold}: nodes={model.tree_.node_count}, "
        f"threshold={threshold}, verified={len(test)}"
    )

print("\n=== COMPILED RULES ===")
for fold in sorted(rules):
    print(f"Fold {fold}: {rules[fold]}")

# Fixed pre-existing air-distance values; only decision cost
# is benchmarked. Same threshold for all methods for timing.
air_threshold = 2900.0
rng = np.random.default_rng(42)

records = []

# Warm up before measuring.
for i in range(len(df)):
    fold = int(valid.iloc[i]["fold"])
    predict_compiled(rules[fold], X[i])
    predict_raw(models[fold], X[i])

gc.collect()
gc.disable()

try:
    for repeat in range(100):
        for i in rng.permutation(len(df)):
            fold = int(valid.iloc[i]["fold"])
            x = X[i]
            model = models[fold]
            rule = rules[fold]
            threshold = thresholds[fold]

            methods = rng.permutation([
                "distance_rule",
                "compiled_rule",
                "raw_tree",
            ])

            for method in methods:
                start = perf_counter_ns()

                if method == "distance_rule":
                    k = 4 if x[0] >= air_threshold else 2
                elif method == "compiled_rule":
                    k = (
                        4 if predict_compiled(rule, x) >= threshold
                        else 2
                    )
                else:
                    k = (
                        4 if predict_raw(model, x) >= threshold
                        else 2
                    )

                elapsed_us = (perf_counter_ns() - start) / 1000

                records.append({
                    "repeat": repeat,
                    "pair_id": int(df.iloc[i]["pair_id"]),
                    "method": method,
                    "elapsed_us": elapsed_us,
                    "chosen_k": k,
                })
finally:
    gc.enable()

result = pd.DataFrame(records)

print("\n=== LATENCY ===")
summary = result.groupby("method")["elapsed_us"].agg(
    median_us="median",
    mean_us="mean",
    p95_us=lambda s: s.quantile(0.95),
)
print(summary.round(4).to_string())

print("\n=== BREAK EVEN ===")
incremental = (
    summary.loc["compiled_rule", "median_us"]
    - summary.loc["distance_rule", "median_us"]
)

print(f"Incremental median latency: {incremental:+.4f} us/OD")
print(
    "Estimated incremental cost for 142 OD: "
    f"{incremental * 142 / 1000:.4f} ms"
)

offline_advantage = (
    df["distance_gate_time_ms"]
    - valid["ml_time_ms"]
).sum()

print(
    "Offline advantage vs distance gate: "
    f"{offline_advantage:.4f} ms"
)

print(
    "Estimated remaining advantage: "
    f"{offline_advantage - incremental * 142 / 1000:+.4f} ms"
)

compiled = result[result["method"] == "compiled_rule"]
raw = result[result["method"] == "raw_tree"]

compiled_k = compiled.sort_values(
    ["repeat", "pair_id"]
)["chosen_k"].to_numpy()

raw_k = raw.sort_values(
    ["repeat", "pair_id"]
)["chosen_k"].to_numpy()

assert np.array_equal(compiled_k, raw_k)
print("\nCompiled and raw decisions identical: True")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_parquet(OUTPUT, index=False)
print("Saved:", OUTPUT)
