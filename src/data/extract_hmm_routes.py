from pathlib import Path
from functools import lru_cache

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

GRAPH_PATH = Path("data/graphs/porto_drive.graphml")
MATCHED_PATH = Path("data/processed/hmm_pilot.parquet")
INTERVALS_PATH = Path("data/processed/gps_intervals.parquet")
OUTPUT_PATH = Path("data/processed/hmm_edge_traversals_pilot.parquet")

G = ox.load_graphml(GRAPH_PATH)
matched = pd.read_parquet(MATCHED_PATH)
intervals = pd.read_parquet(
    INTERVALS_PATH,
    columns=[
        "trip_id",
        "point_index",
        "distance_m",
        "interval_seconds",
    ],
)

interval_lookup = {
    (r.trip_id, int(r.point_index)): (
        float(r.distance_m),
        float(r.interval_seconds),
    )
    for r in intervals.itertuples(index=False)
}


def edge_length(u, v, k):
    return float(G[u][v][k]["length"])


@lru_cache(maxsize=50000)
def middle_path(source, target):
    if source == target:
        return []

    try:
        nodes = nx.shortest_path(
            G,
            source=source,
            target=target,
            weight="length",
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

    edges = []

    for u, v in zip(nodes[:-1], nodes[1:]):
        # MultiDiGraph may have parallel directed edges.
        # Select the minimum-length parallel edge.
        k = min(
            G[u][v],
            key=lambda key: float(G[u][v][key]["length"]),
        )
        edges.append((u, v, k))

    return edges


def extract_parts(a, b):
    u1, v1, k1 = a["edge_u"], a["edge_v"], a["edge_key"]
    u2, v2, k2 = b["edge_u"], b["edge_v"], b["edge_key"]

    f1 = float(a["fraction"])
    f2 = float(b["fraction"])

    l1 = edge_length(u1, v1, k1)
    l2 = edge_length(u2, v2, k2)

    same_edge = (u1, v1, k1) == (u2, v2, k2)

    if same_edge and f2 >= f1:
        distance = (f2 - f1) * l1
        return [(u1, v1, k1, distance)] if distance > 0 else []

    path = middle_path(v1, u2)

    if path is None:
        return None

    parts = []

    first_distance = (1.0 - f1) * l1
    if first_distance > 0:
        parts.append((u1, v1, k1, first_distance))

    for u, v, k in path:
        parts.append((u, v, k, edge_length(u, v, k)))

    last_distance = f2 * l2
    if last_distance > 0:
        parts.append((u2, v2, k2, last_distance))

    return parts


records = []
transition_count = 0
failed_count = 0
distance_mismatches = 0

matched = matched.sort_values(
    ["hmm_fragment_id", "point_index"]
)

for fragment_id, group in matched.groupby(
    "hmm_fragment_id",
    sort=False,
):
    rows = group.to_dict("records")

    for a, b in zip(rows[:-1], rows[1:]):
        if int(b["point_index"]) != int(a["point_index"]) + 1:
            raise RuntimeError(
                f"Non-consecutive points in fragment {fragment_id}"
            )

        key = (a["trip_id"], int(a["point_index"]))

        if key not in interval_lookup:
            raise RuntimeError(f"Missing GPS interval: {key}")

        gps_distance, dt = interval_lookup[key]
        parts = extract_parts(a, b)

        if parts is None:
            failed_count += 1
            continue

        route_distance = sum(p[3] for p in parts)
        expected = float(b["route_distance_from_previous_m"])

        if not np.isclose(
            route_distance,
            expected,
            rtol=1e-5,
            atol=0.05,
        ):
            distance_mismatches += 1

        transition_count += 1

        for order, (u, v, k, traversed_m) in enumerate(parts):
            records.append({
                "hmm_fragment_id": fragment_id,
                "segment_id": int(a["segment_id"]),
                "trip_id": a["trip_id"],
                "point_index": int(b["point_index"]),
                "edge_order": order,
                "edge_u": u,
                "edge_v": v,
                "edge_key": k,
                "traversed_length_m": traversed_m,
                "route_distance_m": route_distance,
                "gps_displacement_m": gps_distance,
                "interval_seconds": dt,
                "route_speed_kmh": route_distance / dt * 3.6,
            })

output = pd.DataFrame(records)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
output.to_parquet(OUTPUT_PATH, index=False)

print("=== ROUTE EXTRACTION ===")
print("Matched GPS points:", len(matched))
print("HMM fragments:", matched["hmm_fragment_id"].nunique())
print("Transitions processed:", transition_count)
print("Unreachable transitions:", failed_count)
print("Distance mismatches:", distance_mismatches)
print("Edge traversal records:", len(output))

if not output.empty:
    print(
        "Unique directed edges:",
        output[["edge_u", "edge_v", "edge_key"]]
        .drop_duplicates()
        .shape[0],
    )

    print("\nRoute speed (km/h), one value per transition:")
    transition_speeds = output.drop_duplicates(
        ["hmm_fragment_id", "point_index"]
    )["route_speed_kmh"]

    print(
        transition_speeds.describe(
            percentiles=[0.5, 0.9, 0.95, 0.99]
        ).to_string()
    )

print("\nSaved:", OUTPUT_PATH)
