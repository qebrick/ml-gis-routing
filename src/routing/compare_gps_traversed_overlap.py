from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

GRAPH_PATH = Path(
    "data/graphs/porto_drive_conservative_refit.graphml"
)
OD_PATH = Path(
    "data/processed/gps_od_route_comparison.parquet"
)
TRAVERSALS_PATH = Path(
    "data/processed/hmm_edge_traversals_pilot.parquet"
)
OUTPUT_PATH = Path(
    "data/processed/gps_traversed_overlap_comparison.parquet"
)

G = ox.load_graphml(GRAPH_PATH)
od = pd.read_parquet(OD_PATH)
tr = pd.read_parquet(TRAVERSALS_PATH)

OBJECTIVES = {
    "osm": "osm_travel_time_s",
    "class": "class_travel_time_s",
    "conservative": "conservative_travel_time_s",
}


def edge_id(u, v, key):
    return int(u), int(v), int(key)


def weight_fn(attribute):
    def weight(u, v, parallel_edges):
        return min(
            float(attrs[attribute])
            for attrs in parallel_edges.values()
        )
    return weight


def route_edge_set(origin, destination, attribute):
    path = nx.shortest_path(
        G,
        source=int(origin),
        target=int(destination),
        weight=weight_fn(attribute),
    )

    selected = set()

    for u, v in zip(path[:-1], path[1:]):
        key, _ = min(
            G[u][v].items(),
            key=lambda item: float(item[1][attribute]),
        )
        selected.add(edge_id(u, v, key))

    return selected


# Build a fragment lookup once instead of repeatedly
# filtering the complete traversal table.
fragments = {}

for (trip_id, fragment_id), part in tr.groupby(
    ["trip_id", "hmm_fragment_id"],
    sort=False,
):
    valid = part.loc[
        np.isfinite(part["traversed_length_m"])
        & (part["traversed_length_m"] > 0)
    ]

    fragments[(str(trip_id), str(fragment_id))] = [
        (
            edge_id(row.edge_u, row.edge_v, row.edge_key),
            float(row.traversed_length_m),
        )
        for row in valid.itertuples(index=False)
    ]


rows = []
missing_fragments = 0
missing_edges = 0

for pair in od.itertuples(index=False):
    fragment_key = (
        str(pair.trip_id),
        str(pair.hmm_fragment_id),
    )

    traversals = fragments.get(fragment_key)

    if not traversals:
        missing_fragments += 1
        continue

    # Keep traversal occurrences, including repetitions.
    valid_traversals = []

    for edge, length in traversals:
        if G.has_edge(*edge):
            valid_traversals.append((edge, length))
        else:
            missing_edges += 1

    total_length = sum(
        length for _, length in valid_traversals
    )

    if total_length <= 0:
        continue

    recalls = {}

    for name, attribute in OBJECTIVES.items():
        route_edges = route_edge_set(
            pair.origin,
            pair.destination,
            attribute,
        )

        matched_length = sum(
            length
            for edge, length in valid_traversals
            if edge in route_edges
        )

        recalls[name] = matched_length / total_length

    rows.append({
        "trip_id": str(pair.trip_id),
        "hmm_fragment_id": str(pair.hmm_fragment_id),
        "gps_traversed_m": total_length,
        "gps_traversal_count": len(valid_traversals),
        "class_changed": bool(pair.class_changed),
        "conservative_changed": bool(
            pair.conservative_changed
        ),
        "recall_osm": recalls["osm"],
        "recall_class": recalls["class"],
        "recall_conservative": recalls["conservative"],
        "delta_class": (
            recalls["class"] - recalls["osm"]
        ),
        "delta_conservative": (
            recalls["conservative"] - recalls["osm"]
        ),
    })


result = pd.DataFrame(rows)

if result.empty:
    raise RuntimeError(
        "No valid traversal-based comparisons"
    )

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)
result.to_parquet(OUTPUT_PATH, index=False)

print("=== COVERAGE ===")
print("OD pairs:", len(result))
print("Unique trips:", result["trip_id"].nunique())
print("Missing fragments:", missing_fragments)
print("Missing traversal edges:", missing_edges)

print("\n=== TRAVERSAL-WEIGHTED RECALL ===")

columns = [
    "recall_osm",
    "recall_class",
    "recall_conservative",
    "delta_class",
    "delta_conservative",
]

print(
    result[columns]
    .agg(["mean", "median", "std"])
    .round(4)
    .to_string()
)

print("\n=== CHANGED ROUTES ONLY ===")

for model in ["class", "conservative"]:
    subset = result.loc[result[f"{model}_changed"]]
    delta = subset[f"delta_{model}"]

    print(f"\n{model.upper()}")
    print("Routes:", len(subset))

    if subset.empty:
        continue

    print("Mean delta:", round(delta.mean(), 4))
    print("Median delta:", round(delta.median(), 4))
    print("Better:", int((delta > 1e-9).sum()))
    print("Worse:", int((delta < -1e-9).sum()))
    print(
        "Equal:",
        int((delta.abs() <= 1e-9).sum()),
    )

print("\n=== CONSERVATIVE BY TRIP ===")

trip_delta = (
    result.groupby("trip_id")["delta_conservative"]
    .mean()
)

print("Trips:", len(trip_delta))
print("Mean trip delta:", round(trip_delta.mean(), 4))
print(
    "Positive trips:",
    int((trip_delta > 1e-9).sum()),
)
print(
    "Negative trips:",
    int((trip_delta < -1e-9).sum()),
)

print("\n=== PREVIOUS VS NEW METRIC ===")

previous_path = Path(
    "data/processed/gps_path_overlap_comparison.parquet"
)

if previous_path.exists():
    previous = pd.read_parquet(previous_path)

    merged = result.merge(
        previous[
            [
                "trip_id",
                "hmm_fragment_id",
                "delta_class",
                "delta_conservative",
            ]
        ],
        on=["trip_id", "hmm_fragment_id"],
        suffixes=("_new", "_old"),
        validate="one_to_one",
    )

    print("Comparable OD pairs:", len(merged))

    for model in ["class", "conservative"]:
        print(
            f"{model}:",
            "old mean delta =",
            round(
                merged[f"delta_{model}_old"].mean(),
                4,
            ),
            "new mean delta =",
            round(
                merged[f"delta_{model}_new"].mean(),
                4,
            ),
        )

print("\nSaved:", OUTPUT_PATH)
