import numpy as np
import pandas as pd

PATH = "data/processed/ml_predictions_pilot.parquet"

df = pd.read_parquet(PATH)

df["baseline_abs_error"] = (
    df["target_speed_kmh"] - df["baseline_speed_kmh"]
).abs()

df["catboost_abs_error"] = (
    df["target_speed_kmh"] - df["catboost_speed_kmh"]
).abs()

df["length_bin"] = pd.cut(
    df["edge_length_m"],
    bins=[0, 50, 100, 200, 400, 1000, np.inf],
    labels=[
        "0-50",
        "50-100",
        "100-200",
        "200-400",
        "400-1000",
        "1000+",
    ],
    right=False,
)

def summarize(group):
    return pd.Series({
        "rows": len(group),
        "trips": group["trip_id"].nunique(),
        "baseline_mae": group["baseline_abs_error"].mean(),
        "catboost_mae": group["catboost_abs_error"].mean(),
        "improvement_pct": (
            100
            * (
                group["baseline_abs_error"].mean()
                - group["catboost_abs_error"].mean()
            )
            / max(group["baseline_abs_error"].mean(), 1e-9)
        ),
    })

print("=== ERROR BY EDGE LENGTH ===")

by_length = (
    df.groupby("length_bin", observed=True)
    .apply(summarize, include_groups=False)
)

print(by_length.round(2).to_string())

print("\n=== ERROR BY HIGHWAY ===")

by_highway = (
    df.groupby("highway")
    .apply(summarize, include_groups=False)
    .sort_values("rows", ascending=False)
)

print(by_highway.round(2).to_string())

print("\n=== TEST COVERAGE ===")
print("Test rows:", len(df))
print("Test trips:", df["trip_id"].nunique())
print(
    "Test unique edges:",
    df[["edge_u", "edge_v", "edge_key"]]
    .drop_duplicates()
    .shape[0],
)

print("\n=== EDGE OVERLAP WITH TRAIN ===")

all_data = pd.read_parquet(
    "data/processed/ml_training_data.parquet"
)

test_trips = set(df["trip_id"])
train = all_data[
    ~all_data["trip_id"].isin(test_trips)
]

train_edges = set(
    zip(
        train["edge_u"],
        train["edge_v"],
        train["edge_key"],
    )
)

test_edges = list(
    zip(
        df["edge_u"],
        df["edge_v"],
        df["edge_key"],
    )
)

seen = np.array([
    edge in train_edges
    for edge in test_edges
])

print("Test rows on edges seen in training:", int(seen.sum()))
print("Test rows on unseen edges:", int((~seen).sum()))

for name, mask in [
    ("Seen edges", seen),
    ("Unseen edges", ~seen),
]:
    subset = df.loc[mask]

    if len(subset) == 0:
        continue

    print(
        f"{name:15s} | "
        f"rows={len(subset):3d} | "
        f"baseline MAE={subset['baseline_abs_error'].mean():.3f} | "
        f"CatBoost MAE={subset['catboost_abs_error'].mean():.3f}"
    )
