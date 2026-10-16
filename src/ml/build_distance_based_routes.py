from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path(
    "data/processed/hmm_backtracking_diagnostics.parquet"
)
OUTPUT_PATH = Path(
    "data/processed/ml_distance_based_routes.parquet"
)

df = pd.read_parquet(INPUT_PATH).copy()

df = df.loc[
    df["max_reverse_pair_m"] < 100
].copy()

df = df.sort_values(
    ["trip_id", "hmm_fragment_id", "point_index"]
)

records = []

for (trip_id, fragment_id), group in df.groupby(
    ["trip_id", "hmm_fragment_id"],
    sort=False,
):
    group = group.sort_values(
        "point_index"
    ).reset_index(drop=True)

    for target_distance_m in [300, 500, 800]:
        start = 0

        while start < len(group):
            total_distance = 0.0
            total_osm_time = 0.0
            total_observed_time = 0.0

            indices = []
            end = start

            while end < len(group):
                row = group.iloc[end]

                if indices:
                    if row["point_index"] != indices[-1] + 1:
                        break

                indices.append(
                    int(row["point_index"])
                )

                total_distance += row["route_distance_m"]
                total_osm_time += row["osm_time_s"]
                total_observed_time += row["interval_seconds"]

                end += 1

                if total_distance >= target_distance_m:
                    break

            if (
                total_distance >= target_distance_m
                and total_osm_time > 0
                and total_observed_time > 0
            ):
                records.append({
                    "trip_id": trip_id,
                    "hmm_fragment_id": fragment_id,
                    "target_distance_m": target_distance_m,
                    "start_point_index": indices[0],
                    "end_point_index": indices[-1],
                    "interval_count": len(indices),
                    "route_distance_m": total_distance,
                    "observed_time_s": total_observed_time,
                    "osm_time_s": total_osm_time,
                    "time_ratio": (
                        total_observed_time / total_osm_time
                    ),
                    "mean_speed_kmh": (
                        3.6 * total_distance
                        / total_observed_time
                    ),
                })

                start = end
            else:
                # Avoid creating a route across a gap.
                start = max(end, start + 1)

result = pd.DataFrame(records)

if result.empty:
    raise RuntimeError(
        "No distance-based routes were created."
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

result.to_parquet(
    OUTPUT_PATH,
    index=False,
)

print("=== DISTANCE-BASED ROUTES ===")
print("Routes:", len(result))
print("Trips:", result["trip_id"].nunique())

print("\n=== BY TARGET DISTANCE ===")

summary = (
    result.groupby("target_distance_m")
    .agg(
        routes=("trip_id", "size"),
        trips=("trip_id", "nunique"),
        median_intervals=("interval_count", "median"),
        median_distance_m=("route_distance_m", "median"),
        median_observed_s=("observed_time_s", "median"),
        median_osm_s=("osm_time_s", "median"),
        osm_mae_s=("osm_abs_error_s", "mean"),
        osm_bias_s=("osm_error_s", "mean"),
    )
)

print(summary.round(2).to_string())

print("\n=== OBSERVED TIME DISTRIBUTION ===")

print(
    result.groupby("target_distance_m")[
        "observed_time_s"
    ]
    .describe()
    .round(2)
    .to_string()
)

print("\n=== SPEED DISTRIBUTION ===")

print(
    result.groupby("target_distance_m")[
        "mean_speed_kmh"
    ]
    .describe()
    .round(2)
    .to_string()
)

print("\nSaved:", OUTPUT_PATH)
