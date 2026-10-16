from pathlib import Path
import ast

import numpy as np
import pandas as pd
import osmnx as ox
from catboost import CatBoostRegressor

GRAPH_PATH = Path("data/graphs/porto_drive_routing.graphml")
MODEL_PATH = Path("models/catboost_speed_pilot.cbm")
OUTPUT_PATH = Path("data/graphs/porto_drive_ml_weighted.graphml")
DIAGNOSTICS_PATH = Path("data/processed/ml_edge_predictions.parquet")

HOUR = 13
DAY_OF_WEEK = 1

MIN_TRAIN_LENGTH = 30.59
MAX_TRAIN_LENGTH = 1948.75

MIN_SPEED_KMH = 5.0
MAX_SPEED_KMH = 90.0


def first_value(value):
    if isinstance(value, (list, tuple)):
        return value[0] if value else None

    if value is None:
        return None

    text = str(value).strip()

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return parsed[0] if parsed else None
        except (ValueError, SyntaxError):
            pass

    return value


def parse_number(value):
    value = first_value(value)

    if value is None:
        return np.nan

    text = str(value).strip().lower()

    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_maxspeed(value):
    value = first_value(value)

    if value is None:
        return np.nan

    text = str(value).strip().lower()

    if "mph" in text:
        try:
            return float(text.replace("mph", "").strip()) * 1.609344
        except ValueError:
            return np.nan

    return parse_number(value)


def parse_oneway(value):
    if isinstance(value, bool):
        return str(value)

    text = str(value).strip().lower()

    if text in {"true", "1", "yes"}:
        return "True"

    if text in {"false", "0", "no"}:
        return "False"

    raise ValueError(f"Unexpected oneway value: {value!r}")


G = ox.load_graphml(GRAPH_PATH)

model = CatBoostRegressor()
model.load_model(str(MODEL_PATH))

records = []
edge_ids = []

for u, v, k, d in G.edges(keys=True, data=True):
    length_m = float(d["length"])
    osm_speed_kmh = float(d["osm_speed_kmh"])

    highway = first_value(d.get("highway"))
    highway = str(highway) if highway is not None else "unknown"

    maxspeed = parse_maxspeed(d.get("maxspeed"))
    lanes = parse_number(d.get("lanes"))

    records.append({
        "edge_length_m": length_m,
        "highway": highway,
        "maxspeed_kmh": maxspeed,
        "lanes": lanes,
        "oneway": parse_oneway(d.get("oneway")),
        "hour": HOUR,
        "day_of_week": DAY_OF_WEEK,
        "has_maxspeed": int(np.isfinite(maxspeed)),
        "has_lanes": int(np.isfinite(lanes)),
        "osm_speed_kmh": osm_speed_kmh,
    })

    edge_ids.append((u, v, k))

df = pd.DataFrame(records)

features = model.feature_names_
X = df[features].copy()

predictions = np.asarray(model.predict(X), dtype=float)

if not np.isfinite(predictions).all():
    raise RuntimeError("Model produced non-finite predictions.")

df["raw_ml_speed_kmh"] = predictions

df["clipped_ml_speed_kmh"] = np.clip(
    predictions,
    MIN_SPEED_KMH,
    MAX_SPEED_KMH,
)

df["outside_train_length"] = (
    (df["edge_length_m"] < MIN_TRAIN_LENGTH)
    | (df["edge_length_m"] > MAX_TRAIN_LENGTH)
)

df["routing_speed_kmh"] = np.where(
    df["outside_train_length"],
    df["osm_speed_kmh"],
    df["clipped_ml_speed_kmh"],
)

df["ml_travel_time_s"] = (
    3.6 * df["edge_length_m"] / df["routing_speed_kmh"]
)

if not np.isfinite(df["ml_travel_time_s"]).all():
    raise RuntimeError("Invalid ML travel times.")

if (df["ml_travel_time_s"] <= 0).any():
    raise RuntimeError("Non-positive ML travel times.")

for i, (u, v, k) in enumerate(edge_ids):
    d = G[u][v][k]

    d["ml_speed_kmh"] = float(df.iloc[i]["routing_speed_kmh"])
    d["ml_travel_time_s"] = float(df.iloc[i]["ml_travel_time_s"])

df.insert(0, "edge_u", [u for u, v, k in edge_ids])
df.insert(1, "edge_v", [v for u, v, k in edge_ids])
df.insert(2, "edge_key", [k for u, v, k in edge_ids])

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
DIAGNOSTICS_PATH.parent.mkdir(parents=True, exist_ok=True)

ox.save_graphml(G, OUTPUT_PATH)
df.to_parquet(DIAGNOSTICS_PATH, index=False)

print("=== ML GRAPH DIAGNOSTICS ===")
print("Edges:", len(df))
print("Hour:", HOUR)
print("Day of week:", DAY_OF_WEEK)

print("\nOutside training length range:")
print(int(df["outside_train_length"].sum()))

print("\nPredictions below minimum speed:")
print(int((df["raw_ml_speed_kmh"] < MIN_SPEED_KMH).sum()))

print("\nPredictions above maximum speed:")
print(int((df["raw_ml_speed_kmh"] > MAX_SPEED_KMH).sum()))

print("\n=== SPEED DISTRIBUTION ===")
print(
    df[
        [
            "osm_speed_kmh",
            "raw_ml_speed_kmh",
            "routing_speed_kmh",
        ]
    ].describe(
        percentiles=[0.1, 0.5, 0.9, 0.95]
    ).round(2).to_string()
)

print("\n=== ROUTING SPEED BY HIGHWAY ===")
print(
    df.groupby("highway")["routing_speed_kmh"]
    .agg(["count", "median", "mean"])
    .sort_values("count", ascending=False)
    .head(15)
    .round(2)
    .to_string()
)

print("\nSaved graph:", OUTPUT_PATH)
print("Saved diagnostics:", DIAGNOSTICS_PATH)
