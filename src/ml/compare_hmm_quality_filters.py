from pathlib import Path

import pandas as pd

INPUT_PATH = Path(
    "data/processed/hmm_backtracking_diagnostics.parquet"
)

df = pd.read_parquet(INPUT_PATH).copy()

scenarios = {
    "all_clean_intervals": pd.Series(
        True, index=df.index
    ),
    "no_long_reverse_100m": (
        df["max_reverse_pair_m"] < 100
    ),
    "no_reverse_50m": (
        df["max_reverse_pair_m"] < 50
    ),
    "no_immediate_reverse": (
        df["reverse_pairs"] == 0
    ),
    "moving_5_to_90": (
        df["target_speed_kmh"].between(5, 90)
    ),
    "moving_no_long_reverse": (
        df["target_speed_kmh"].between(5, 90)
        & (df["max_reverse_pair_m"] < 100)
    ),
    "moving_no_reverse": (
        df["target_speed_kmh"].between(5, 90)
        & (df["reverse_pairs"] == 0)
    ),
}

rows = []

for name, mask in scenarios.items():
    subset = df.loc[mask]

    rows.append({
        "scenario": name,
        "intervals": len(subset),
        "trips": subset["trip_id"].nunique(),
        "share_pct": 100 * len(subset) / len(df),
        "median_speed_kmh": (
            subset["target_speed_kmh"].median()
        ),
        "osm_mae_s": (
            subset["absolute_time_error_s"].mean()
        ),
        "osm_bias_s": (
            subset["time_error_s"].mean()
        ),
    })

result = pd.DataFrame(rows)

print("=== FILTER SENSITIVITY ===")
print(
    result.round(2).to_string(index=False)
)

print("\n=== REMOVED BY LONG REVERSE ===")
removed = df.loc[
    df["max_reverse_pair_m"] >= 100
]

print("Intervals:", len(removed))
print("Trips:", removed["trip_id"].nunique())

print("\n=== REMOVED BY SPEED ===")
speed_removed = df.loc[
    ~df["target_speed_kmh"].between(5, 90)
]

print("Intervals:", len(speed_removed))
print("Trips:", speed_removed["trip_id"].nunique())
