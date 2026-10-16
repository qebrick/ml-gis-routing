from pathlib import Path
import ast
import re

import numpy as np
import osmnx as ox
import pandas as pd

GRAPH_PATH = Path("data/graphs/porto_drive.graphml")
SAMPLES_PATH = Path(
    "data/processed/single_edge_samples_pilot.parquet"
)
INTERVALS_PATH = Path(
    "data/processed/gps_intervals.parquet"
)
OUTPUT_PATH = Path(
    "data/processed/ml_training_data.parquet"
)


def parse_numeric(value):
    if value is None:
        return np.nan

    if isinstance(value, (list, tuple)):
        values = [parse_numeric(x) for x in value]
        values = [x for x in values if pd.notna(x)]
        return min(values) if values else np.nan

    text = str(value).strip()

    if text.startswith("[") and text.endswith("]"):
        try:
            return parse_numeric(ast.literal_eval(text))
        except (ValueError, SyntaxError):
            return np.nan

    matches = re.findall(r"\d+(?:\.\d+)?", text)

    if not matches:
        return np.nan

    return float(matches[0])


def parse_highway(value):
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else "unknown"

    text = str(value)

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list) and parsed:
                return str(parsed[0])
        except (ValueError, SyntaxError):
            pass

    return text if text != "None" else "unknown"


G = ox.load_graphml(GRAPH_PATH)

edge_rows = []

for u, v, k, d in G.edges(keys=True, data=True):
    edge_rows.append({
        "edge_u": u,
        "edge_v": v,
        "edge_key": k,
        "edge_length_m": float(d["length"]),
        "highway": parse_highway(d.get("highway")),
        "maxspeed_kmh": parse_numeric(d.get("maxspeed")),
        "lanes": parse_numeric(d.get("lanes")),
        "oneway": str(d.get("oneway", "unknown")),
    })

edges = pd.DataFrame(edge_rows)

samples = pd.read_parquet(SAMPLES_PATH)

data = samples.merge(
    edges,
    on=["edge_u", "edge_v", "edge_key"],
    how="left",
    validate="many_to_one",
)

if data["edge_length_m"].isna().any():
    raise RuntimeError("Some matched edges are absent from graph.")

intervals = pd.read_parquet(
    INTERVALS_PATH,
    columns=["trip_id", "point_index", "timestamp_utc"],
)

# In traversal records, point_index denotes the END of
# the GPS interval, so the original interval index is i-1.
data["interval_point_index"] = data["point_index"] - 1

data = data.merge(
    intervals.rename(
        columns={"point_index": "interval_point_index"}
    ),
    on=["trip_id", "interval_point_index"],
    how="left",
    validate="many_to_one",
)

if data["timestamp_utc"].isna().any():
    raise RuntimeError("Missing timestamps for training samples.")

timestamps = pd.to_datetime(
    data["timestamp_utc"],
    utc=True,
)

# Porto uses Europe/Lisbon local time, including DST.
local_time = timestamps.dt.tz_convert("Europe/Lisbon")

data["hour"] = local_time.dt.hour
data["day_of_week"] = local_time.dt.dayofweek

data["has_maxspeed"] = data["maxspeed_kmh"].notna().astype(int)
data["has_lanes"] = data["lanes"].notna().astype(int)

data["target_speed_kmh"] = data["observed_speed_kmh"]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
data.to_parquet(OUTPUT_PATH, index=False)

print("=== ML TRAINING DATA ===")
print("Rows:", len(data))
print("Trips:", data["trip_id"].nunique())
print(
    "Unique directed edges:",
    data[["edge_u", "edge_v", "edge_key"]]
    .drop_duplicates()
    .shape[0],
)

print("\nMissing features:")
print(
    data[
        [
            "edge_length_m",
            "highway",
            "maxspeed_kmh",
            "lanes",
            "oneway",
            "hour",
            "day_of_week",
            "target_speed_kmh",
        ]
    ].isna().sum().to_string()
)

print("\nTarget speed (km/h):")
print(
    data["target_speed_kmh"]
    .describe(percentiles=[0.5, 0.9, 0.95])
    .to_string()
)

print("\nSaved:", OUTPUT_PATH)
