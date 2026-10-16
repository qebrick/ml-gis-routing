from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

DATA = Path(
    "data/processed/ml_alt_budget_groupkfold.parquet"
)
OUTPUT = Path(
    "data/processed/ml_alt_budget_inference_cost.parquet"
)

df = pd.read_parquet(DATA).sort_values("pair_id").reset_index(drop=True)

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

X = df[feature_cols].to_numpy(dtype=np.float64)
y = df["saved_expansions"].to_numpy(dtype=np.float64)

rng = np.random.default_rng(42)
records = []
all_predictions = np.full(len(df), np.nan)

print("=== SINGLE-QUERY RF INFERENCE ===")

for fold in sorted(df["fold"].unique()):
    train = np.flatnonzero(df["fold"].to_numpy() != fold)
    test = np.flatnonzero(df["fold"].to_numpy() == fold)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X[train], y[train])

    # Verify the retrained model reproduces saved decisions.
    threshold = float(
        df.iloc[test]["train_selected_threshold"].iloc[0]
    )
    predicted = model.predict(X[test])
    expected_k = np.where(predicted >= threshold, 4, 2)

    saved_k = df.iloc[test]["ml_k"].to_numpy()
    if not np.array_equal(expected_k, saved_k):
        raise AssertionError(
            f"Fold {fold}: retrained decisions differ "
            "from saved ML results"
        )

    all_predictions[test] = predicted

    # Warm up both prediction paths.
    for i in test[:min(5, len(test))]:
        model.predict(X[i:i + 1])
    model.predict(X[test])

    # Repeated single-query calls in randomized order.
    for repeat in range(20):
        for i in rng.permutation(test):
            start = perf_counter_ns()
            score = float(model.predict(X[i:i + 1])[0])
            elapsed_ms = (perf_counter_ns() - start) / 1e6

            records.append({
                "fold": int(fold),
                "pair_id": int(df.iloc[i]["pair_id"]),
                "repeat": repeat,
                "inference_ms": elapsed_ms,
                "predicted_score": score,
            })

    # Batch inference is not representative of a
    # single route request, but gives useful context.
    batch_samples = []
    for _ in range(20):
        start = perf_counter_ns()
        model.predict(X[test])
        batch_samples.append(
            (perf_counter_ns() - start) / 1e6
        )

    print(
        f"Fold {fold}: "
        f"test OD={len(test)}, "
        f"median batch={np.median(batch_samples):.4f} ms, "
        f"per OD={np.median(batch_samples)/len(test)*1000:.2f} us"
    )

timings = pd.DataFrame(records)

per_od = (
    timings.groupby("pair_id", as_index=False)
    ["inference_ms"].median()
)

result = df[[
    "pair_id",
    "destination",
    "fold",
    "ml_k",
    "alt2_time_ms",
    "alt4_time_ms",
    "distance_gate_time_ms",
    "ml_time_ms",
]].merge(
    per_od,
    on="pair_id",
    validate="one_to_one",
)

result["ml_with_inference_ms"] = (
    result["ml_time_ms"] + result["inference_ms"]
)

print("\n=== SINGLE-QUERY INFERENCE COST ===")
print(
    "Median per OD (us):",
    round(timings["inference_ms"].median() * 1000, 3)
)
print(
    "Mean per OD (us):",
    round(timings["inference_ms"].mean() * 1000, 3)
)
print(
    "P95 per OD (us):",
    round(timings["inference_ms"].quantile(.95) * 1000, 3)
)

print("\n=== BREAK-EVEN ANALYSIS ===")

offline_gain_vs_distance = (
    result["distance_gate_time_ms"]
    - result["ml_time_ms"]
).sum()

offline_gain_vs_alt2 = (
    result["alt2_time_ms"]
    - result["ml_time_ms"]
).sum()

inference_cost = result["inference_ms"].sum()

print(
    "Offline saving vs distance gate (ms):",
    round(offline_gain_vs_distance, 4)
)
print(
    "Offline saving vs ALT-2 (ms):",
    round(offline_gain_vs_alt2, 4)
)
print(
    "RF inference cost for 142 OD (ms):",
    round(inference_cost, 4)
)

print(
    "Net saving vs distance gate, "
    "excluding feature computation (ms):",
    round(offline_gain_vs_distance - inference_cost, 4)
)

print(
    "Net saving vs ALT-2, "
    "excluding feature computation (ms):",
    round(offline_gain_vs_alt2 - inference_cost, 4)
)

print("\n=== IMPORTANT ===")
print(
    "These figures exclude computation of 32 ALT "
    "features per OD, so they are optimistic for ML."
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_parquet(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
