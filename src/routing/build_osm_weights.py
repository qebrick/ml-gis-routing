from pathlib import Path
import ast

import osmnx as ox
import pandas as pd

GRAPH_PATH = Path("data/graphs/porto_drive.graphml")
OUTPUT_PATH = Path("data/graphs/porto_drive_osm_weighted.graphml")

SPEEDS = {
    "motorway": 90.0,
    "trunk": 70.0,
    "primary": 50.0,
    "secondary": 40.0,
    "tertiary": 35.0,
    "residential": 25.0,
    "living_street": 15.0,
    "unclassified": 25.0,
    "service": 15.0,
    "busway": 25.0,
}


def normalize_highway(value):
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

    return text


def baseline_speed(highway):
    highway = normalize_highway(highway)

    if highway.endswith("_link"):
        base = highway.removesuffix("_link")
        return 0.7 * SPEEDS.get(base, 25.0)

    return SPEEDS.get(highway, 25.0)


G = ox.load_graphml(GRAPH_PATH)

records = []

for u, v, k, d in G.edges(keys=True, data=True):
    length_m = float(d["length"])
    speed_kmh = baseline_speed(d.get("highway"))

    travel_time_s = 3.6 * length_m / speed_kmh

    d["osm_speed_kmh"] = float(speed_kmh)
    d["osm_travel_time_s"] = float(travel_time_s)

    records.append({
        "edge_u": u,
        "edge_v": v,
        "edge_key": k,
        "highway": normalize_highway(d.get("highway")),
        "length_m": length_m,
        "speed_kmh": speed_kmh,
        "travel_time_s": travel_time_s,
    })

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
ox.save_graphml(G, OUTPUT_PATH)

edges = pd.DataFrame(records)

print("=== OSM BASELINE WEIGHTS ===")
print("Edges:", len(edges))

print("\nSpeed distribution (km/h):")
print(
    edges["speed_kmh"]
    .describe(percentiles=[0.1, 0.5, 0.9, 0.95])
    .to_string()
)

print("\nTravel time distribution (seconds):")
print(
    edges["travel_time_s"]
    .describe(percentiles=[0.1, 0.5, 0.9, 0.95])
    .to_string()
)

print("\nNon-positive travel times:")
print(int((edges["travel_time_s"] <= 0).sum()))

print("\nSaved:", OUTPUT_PATH)
