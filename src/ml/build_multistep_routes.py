from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path(
    "data/processed/hmm_backtracking_diagnostics.parquet"
)
OUTPUT_PATH = Path(
    "data/processed/ml_multistep_routes.parquet"
)

df = pd.read_parquet(INPUT_PATH).copy()

# Exclude suspicious immediate reversals >= 100 m.
df = df.loc[
    df["max_reverse_pair_m"] < 100
].copy()

# Keep original HMM fragment identifiers to prevent
# windows from crossing map-matching discontinuities.
required = [
    "trip_id",
    "point_index",
    "hmm_fragment_id",
    "interval_seconds",
    "route_distance_m",
    "osm_time_s",
]

missing = set(required) - set(df.columns)
if missing:
    raise RuntimeError(
        f"Missing columns: {sorted(missing)}"
    )

df = df.sort_values(
    ["trip_id", "hmm_fragment_id", "point_index"]
).copy()

records = []

for (trip_id, fragment_id), group in df.groupby(
    ["trip_id", "hmm_fragment_id"],
    sort=False,
):
    group = group.sort_values("point_index").reset_index(
        drop=True
    )

    for window_size in [4, 8]:
        for start in range(
            0,
            len(group) - window_size + 1,
            window_size,
        ):
            window = group.iloc[
                start:start + window_size
            ]

            # Do not join intervals across removed
            # observations or missing GPS indices.
            indices = window["point_index"].to_numpy()

            if not np.all(np.diff(indices) == 1):
                continue

            observed_time_s = (
                window["interval_seconds"].sum()
            )
            osm_time_s = window["osm_time_s"].sum()
            route_distance_m = (
                window["route_distance_m"].sum()
            )

            if (
                observed_time_s <= 0
                or osm_time_s <= 0
                or route_distance_m <= 0
            ):
                continue

            records.append({
                "trip_id": trip_id,
                "hmm_fragment_id": fragment_id,
                "window_size": window_size,
                "start_point_index": int(indices[0]),
                "end_point_index": int(indices[-1]),
                "observed_time_s": observed_time_s,
                "osm_time_s": osm_time_s,
                "route_distance_m": route_distance_m,
                "mean_observed_speed_kmh": (
                    3.6 * route_distance_m
                    / observed_time_s
                ),
                "time_ratio": (
                    observed_time_s / osm_time_s
                ),
            })

result = pd.DataFrame(records)

if result.empty:
    raise RuntimeError(
        "No continuous multistep routes were produced."
    )

result["osm_error_s"] = (
    result["osm_time_s"]
    - result["observed_time_s"]
)

result["osm_abs_error_s"] = (
    result["osm_error_s"].abs()
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result.to_parquet(OUTPUT_PATH, index=False)

print("=== MULTISTEP ROUTES ===")
print("Routes:", len(result))
print("Trips:", result["trip_id"].nunique())

print("\n=== BY WINDOW SIZE ===")

summary = (
    result.groupby("window_size")
    .agg(
        routes=("trip_id", "size"),
        trips=("trip_id", "nunique"),
        median_distance_m=(
            "route_distance_m", "median"
        ),
        median_observed_s=(
            "observed_time_s", "median"
        ),
        median_osm_s=(
            "osm_time_s", "median"
        ),
        osm_mae_s=("osm_abs_error_s", "mean"),
        osm_bias_s=("osm_error_s", "mean"),
    )
)

print(summary.round(2).to_string())

print("\n=== SPEED DISTRIBUTION ===")

print(
    result.groupby("window_size")[
        "mean_observed_speed_kmh"
    ]
    .describe()
    .round(2)
    .to_string()
)

print("\n=== TIME RATIO ===")

print(
    result.groupby("window_size")[
        "time_ratio"
    ]
    .describe()
    .round(2)
    .to_string()
)

print("\nSaved:", OUTPUT_PATH)
