from pathlib import Path
from functools import lru_cache

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
from pyproj import Transformer

GRAPH_PATH = Path("data/graphs/porto_drive.graphml")
INPUT_PATH = Path("data/processed/map_matching_segments_v2.parquet")
OUTPUT_PATH = Path("data/processed/hmm_pilot.parquet")

N_SEGMENTS = 121
N_CANDIDATES = 4
SEARCH_RADIUS_M = 60.0
GPS_SIGMA_M = 15.0
TRANSITION_BETA_M = 50.0
MAX_ROUTE_SPEED_KMH = 130.0

print("Loading graph...")
G = ox.load_graphml(GRAPH_PATH)
Gp = ox.project_graph(G)

print("Preparing edge geometries...")

edge_ids = []
geometries = []
edge_lengths = []
node_xy = {
    n: (float(d["x"]), float(d["y"]))
    for n, d in Gp.nodes(data=True)
}

for u, v, k, data in Gp.edges(keys=True, data=True):
    geom = data.get("geometry")

    if geom is None:
        geom = LineString([node_xy[u], node_xy[v]])

    if geom.length <= 0:
        continue

    edge_ids.append((u, v, k))
    geometries.append(geom)
    edge_lengths.append(float(data["length"]))

tree = STRtree(geometries)

transformer = Transformer.from_crs(
    "EPSG:4326",
    Gp.graph["crs"],
    always_xy=True,
)

@lru_cache(maxsize=50000)
def shortest_length(source, target):
    if source == target:
        return 0.0

    try:
        return float(
            nx.shortest_path_length(
                Gp,
                source=source,
                target=target,
                weight="length",
            )
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float("inf")


def get_candidates(lon, lat):
    x, y = transformer.transform(float(lon), float(lat))
    p = Point(x, y)

    indices = tree.query(p.buffer(SEARCH_RADIUS_M))
    candidates = []

    for idx in indices:
        idx = int(idx)
        geom = geometries[idx]
        distance = p.distance(geom)

        if distance > SEARCH_RADIUS_M:
            continue

        u, v, k = edge_ids[idx]

        # Determine the direction of the geometry.
        start = Point(node_xy[u])
        end = Point(node_xy[v])

        forward = (
            geom.coords[0][0] - start.x
        ) ** 2 + (
            geom.coords[0][1] - start.y
        ) ** 2 <= (
            geom.coords[-1][0] - start.x
        ) ** 2 + (
            geom.coords[-1][1] - start.y
        ) ** 2

        # Geometry may be stored in reverse direction.
        projected = geom.project(p)
        fraction = projected / geom.length

        if not forward:
            fraction = 1.0 - fraction

        fraction = float(np.clip(fraction, 0.0, 1.0))

        candidates.append({
            "u": u,
            "v": v,
            "key": k,
            "distance_m": float(distance),
            "fraction": fraction,
            "edge_length_m": edge_lengths[idx],
        })

    candidates.sort(key=lambda c: c["distance_m"])
    return candidates[:N_CANDIDATES]


def route_distance(a, b):
    # Positions are measured along directed edges.
    la = a["edge_length_m"]
    lb = b["edge_length_m"]

    if (a["u"], a["v"], a["key"]) == (
        b["u"], b["v"], b["key"]
    ):
        if b["fraction"] >= a["fraction"]:
            return (b["fraction"] - a["fraction"]) * la

    # Complete the first edge, travel through the network,
    # then traverse the initial portion of the second edge.
    middle = shortest_length(a["v"], b["u"])

    if not np.isfinite(middle):
        return float("inf")

    return (
        (1.0 - a["fraction"]) * la
        + middle
        + b["fraction"] * lb
    )


def emission_cost(c):
    return 0.5 * (c["distance_m"] / GPS_SIGMA_M) ** 2


def transition_cost(a, b, displacement_m, dt_seconds):
    distance = route_distance(a, b)

    if not np.isfinite(distance):
        return float("inf"), distance

    speed_kmh = distance / dt_seconds * 3.6

    # Hard diagnostic feasibility constraint for this pilot.
    if speed_kmh > MAX_ROUTE_SPEED_KMH:
        return float("inf"), distance

    cost = abs(distance - displacement_m) / TRANSITION_BETA_M
    return cost, distance


intervals = pd.read_parquet(
    "data/processed/gps_intervals.parquet",
    columns=[
        "trip_id",
        "point_index",
        "distance_m",
        "interval_seconds",
    ],
)

interval_lookup = {
    (row.trip_id, int(row.point_index)):
    (float(row.distance_m), float(row.interval_seconds))
    for row in intervals.itertuples(index=False)
}

points = pd.read_parquet(INPUT_PATH)
points = points.sort_values(
    ["segment_id", "point_index"]
).reset_index(drop=True)

segment_ids = (
    points.groupby("segment_id")
    .size()
    .sort_values(ascending=False)
    .head(N_SEGMENTS)
    .index
)

results = []
successful_segments = 0
failed_segments = 0

print(f"Processing {len(segment_ids)} segments...")

for segment_id in segment_ids:
    group = points[
        points["segment_id"] == segment_id
    ].sort_values("point_index")

    rows = list(group.itertuples(index=False))
    candidate_lists = [
        get_candidates(row.lon, row.lat)
        for row in rows
    ]

    if any(not c for c in candidate_lists):
        print(f"Segment {segment_id}: no candidates for some points")
        failed_segments += 1
        continue

    fragment_start = 0
    fragment_number = 0
    saved_fragments = 0
    discarded_fragments = 0
    split_count = 0


    def save_fragment(start, end, final_costs, pointers_history):
        global successful_segments

        fragment_length = end - start

        if fragment_length < 5:
            return False

        if not any(np.isfinite(final_costs)):
            return False

        state = int(np.argmin(final_costs))
        states = [state]

        for pointers in reversed(pointers_history):
            state = pointers[state]

            if state < 0:
                raise RuntimeError("Invalid Viterbi backpointer.")

            states.append(state)

        states.reverse()

        if len(states) != fragment_length:
            raise RuntimeError("Viterbi fragment length mismatch.")

        for local_i, state in enumerate(states):
            i = start + local_i
            row = rows[i]
            c = candidate_lists[i][state]

            matched_route_distance = np.nan

            if local_i > 0:
                previous = candidate_lists[i - 1][states[local_i - 1]]
                matched_route_distance = route_distance(previous, c)

            results.append({
                "segment_id": int(segment_id),
                "hmm_fragment_id": (
                    f"{segment_id}_{fragment_number}"
                ),
                "trip_id": row.trip_id,
                "point_index": int(row.point_index),
                "edge_u": c["u"],
                "edge_v": c["v"],
                "edge_key": c["key"],
                "fraction": c["fraction"],
                "distance_to_edge_m": c["distance_m"],
                "route_distance_from_previous_m": matched_route_distance,
            })

        return True


    costs = [
        emission_cost(c)
        for c in candidate_lists[0]
    ]

    backpointers = []

    for i in range(1, len(rows)):
        prev_row = rows[i - 1]
        key = (prev_row.trip_id, int(prev_row.point_index))

        interval = interval_lookup.get(key)

        if interval is None:
            raise RuntimeError(
                f"Missing verified interval inside segment: {key}"
            )

        displacement, dt = interval

        if dt <= 0:
            raise RuntimeError(f"Invalid interval duration: {key}")

        new_costs = []
        pointers = []

        for current in candidate_lists[i]:
            best_cost = float("inf")
            best_prev = -1

            for j, previous in enumerate(candidate_lists[i - 1]):
                if not np.isfinite(costs[j]):
                    continue

                tcost, _ = transition_cost(
                    previous, current, displacement, dt
                )

                if not np.isfinite(tcost):
                    continue

                score = costs[j] + tcost + emission_cost(current)

                if score < best_cost:
                    best_cost = score
                    best_prev = j

            new_costs.append(best_cost)
            pointers.append(best_prev)

        if not any(np.isfinite(new_costs)):
            # Finish the valid fragment before the failed transition.
            saved = save_fragment(
                fragment_start,
                i,
                costs,
                backpointers,
            )

            if saved:
                saved_fragments += 1
            else:
                discarded_fragments += 1

            split_count += 1
            fragment_number += 1

            # Restart at the current GPS observation.
            fragment_start = i
            costs = [
                emission_cost(c)
                for c in candidate_lists[i]
            ]
            backpointers = []

            continue

        costs = new_costs
        backpointers.append(pointers)

    # Save the final fragment.
    saved = save_fragment(
        fragment_start,
        len(rows),
        costs,
        backpointers,
    )

    if saved:
        saved_fragments += 1
    else:
        discarded_fragments += 1

    if saved_fragments:
        successful_segments += 1
    else:
        failed_segments += 1

    print(
        f"Segment {segment_id}: "
        f"saved fragments={saved_fragments}, "
        f"discarded fragments={discarded_fragments}, "
        f"splits={split_count}"
    )
    print(
        f"Segment {segment_id}: "
        f"matched {len(rows)} points"
    )

output = pd.DataFrame(results)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
output.to_parquet(OUTPUT_PATH, index=False)

print("\n=== HMM PILOT RESULTS ===")
print("Successful segments:", successful_segments)
print("Failed segments:", failed_segments)
print("Matched points:", len(output))
if not output.empty:
    print(
        "Saved HMM fragments:",
        output["hmm_fragment_id"].nunique()
    )

    print(
        "Original segments represented:",
        output["segment_id"].nunique()
    )

    print(
        "Retained points (%):",
        round(
            100 * len(output)
            / len(points[points["segment_id"].isin(segment_ids)]),
            2,
        )
    )

    fragment_lengths = output.groupby(
        "hmm_fragment_id"
    ).size()

    print("\nHMM fragment lengths:")
    print(fragment_lengths.describe().to_string())
print("Saved:", OUTPUT_PATH)

if not output.empty:
    print("\nDistance to matched edge (m):")
    print(
        output["distance_to_edge_m"]
        .describe(percentiles=[0.5, 0.9, 0.95, 0.99])
        .to_string()
    )

    print("\nMatched route distance between GPS points (m):")
    print(
        output["route_distance_from_previous_m"]
        .dropna()
        .describe(percentiles=[0.5, 0.9, 0.95, 0.99])
        .to_string()
    )
