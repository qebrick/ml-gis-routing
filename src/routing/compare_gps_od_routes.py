from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

GRAPH_PATH = Path(
    "data/graphs/porto_drive_conservative_refit.graphml"
)
HMM_PATH = Path(
    "data/processed/hmm_pilot.parquet"
)
OUTPUT_PATH = Path(
    "data/processed/gps_od_route_comparison.parquet"
)

G = ox.load_graphml(GRAPH_PATH)
hmm = pd.read_parquet(HMM_PATH)

OBJECTIVES = {
    "osm": "osm_travel_time_s",
    "class": "class_travel_time_s",
    "conservative": "conservative_travel_time_s",
}

MIN_DISTANCE_M = 300.0


def weight_fn(attribute):
    def weight(u, v, parallel_edges):
        return min(
            float(attrs[attribute])
            for attrs in parallel_edges.values()
        )
    return weight


def evaluate_path(path, attribute):
    distance = 0.0
    residential_length = 0.0
    selected_edges = []
    costs = {
        name: 0.0
        for name in OBJECTIVES
    }

    for u, v in zip(path[:-1], path[1:]):
        key, attrs = min(
            G[u][v].items(),
            key=lambda item: float(
                item[1][attribute]
            ),
        )

        selected_edges.append((u, v, key))

        length = float(attrs["length"])
        distance += length

        if attrs["class_road_group"] == "residential":
            residential_length += length

        for name, cost_attribute in OBJECTIVES.items():
            costs[name] += float(
                attrs[cost_attribute]
            )

    return {
        "edges": tuple(selected_edges),
        "distance_m": distance,
        "residential_share": (
            residential_length / distance
            if distance > 0 else 0.0
        ),
        **{
            f"time_{name}_s": value
            for name, value in costs.items()
        },
    }


# Segment boundaries are approximate:
# first edge_u -> last edge_v.
# Use HMM fragments to avoid connecting
# across map-matching restarts.
group_columns = [
    "trip_id",
    "hmm_fragment_id",
]

missing = [
    col for col in group_columns
    if col not in hmm.columns
]

if missing:
    raise ValueError(
        f"Missing HMM columns: {missing}"
    )

hmm = hmm.sort_values(
    group_columns + ["point_index"]
)

records = []
skipped = {
    "short": 0,
    "unreachable": 0,
    "invalid": 0,
}

for group_key, group in hmm.groupby(
    group_columns,
    sort=False,
):
    if len(group) < 2:
        continue

    first = group.iloc[0]
    last = group.iloc[-1]

    origin = int(first["edge_u"])
    destination = int(last["edge_v"])

    if origin == destination:
        skipped["invalid"] += 1
        continue

    routes = {}

    try:
        for name, attribute in OBJECTIVES.items():
            path = nx.shortest_path(
                G,
                origin,
                destination,
                weight=weight_fn(attribute),
            )

            routes[name] = evaluate_path(
                path, attribute
            )

    except (nx.NetworkXNoPath, nx.NodeNotFound):
        skipped["unreachable"] += 1
        continue

    osm = routes["osm"]

    if osm["distance_m"] < MIN_DISTANCE_M:
        skipped["short"] += 1
        continue

    class_route = routes["class"]
    conservative = routes["conservative"]

    record = {
        "trip_id": group_key[0],
        "hmm_fragment_id": group_key[1],
        "origin": origin,
        "destination": destination,
        "osm_distance_m": osm["distance_m"],
        "class_distance_m": class_route["distance_m"],
        "conservative_distance_m": (
            conservative["distance_m"]
        ),
        "class_changed": (
            class_route["edges"] != osm["edges"]
        ),
        "conservative_changed": (
            conservative["edges"] != osm["edges"]
        ),
        "osm_residential_share": (
            osm["residential_share"]
        ),
        "class_residential_share": (
            class_route["residential_share"]
        ),
        "conservative_residential_share": (
            conservative["residential_share"]
        ),
    }

    for model, route in routes.items():
        for objective in OBJECTIVES:
            record[
                f"{model}_route_{objective}_time_s"
            ] = route[f"time_{objective}_s"]

    for model in ["class", "conservative"]:
        route = routes[model]

        record[f"{model}_distance_change_pct"] = (
            100
            * (
                route["distance_m"]
                / osm["distance_m"]
                - 1
            )
        )

        record[
            f"{model}_residential_change_pp"
        ] = (
            100
            * (
                route["residential_share"]
                - osm["residential_share"]
            )
        )

        record[
            f"{model}_osm_time_change_pct"
        ] = (
            100
            * (
                route["time_osm_s"]
                / osm["time_osm_s"]
                - 1
            )
        )

        record[
            f"{model}_objective_gain_pct"
        ] = (
            100
            * (
                osm[f"time_{model}_s"]
                - route[f"time_{model}_s"]
            )
            / osm[f"time_{model}_s"]
        )

    records.append(record)

result = pd.DataFrame(records)

if result.empty:
    raise RuntimeError(
        "No eligible GPS-derived OD pairs"
    )

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)
result.to_parquet(OUTPUT_PATH, index=False)

print("=== GPS OD COVERAGE ===")
print("Eligible OD pairs:", len(result))
print("Unique trips:", result["trip_id"].nunique())
print("Skipped:", skipped)

print("\n=== ROUTE CHANGE COMPARISON ===")

for model in ["class", "conservative"]:
    changed = result[f"{model}_changed"]
    subset = result.loc[changed]

    print(f"\n{model.upper()}")
    print(
        "Changed routes:",
        int(changed.sum()),
        "/",
        len(result),
    )

    print(
        "Changed share (%):",
        round(100 * changed.mean(), 2),
    )

    print(
        "Mean residential change (all), pp:",
        round(
            result[
                f"{model}_residential_change_pp"
            ].mean(),
            3,
        ),
    )

    print(
        "Mean residential change (changed), pp:",
        round(
            subset[
                f"{model}_residential_change_pp"
            ].mean(),
            3,
        ) if len(subset) else np.nan,
    )

    print(
        "Mean distance change (changed), %:",
        round(
            subset[
                f"{model}_distance_change_pct"
            ].mean(),
            3,
        ) if len(subset) else np.nan,
    )

    print(
        "Mean OSM time change (changed), %:",
        round(
            subset[
                f"{model}_osm_time_change_pct"
            ].mean(),
            3,
        ) if len(subset) else np.nan,
    )

    print(
        "Mean own-objective gain (all), %:",
        round(
            result[
                f"{model}_objective_gain_pct"
            ].mean(),
            3,
        ),
    )

print("\n=== SWITCHING PATTERNS ===")

print(
    pd.crosstab(
        result["class_changed"],
        result["conservative_changed"],
        rownames=["Class changed"],
        colnames=["Conservative changed"],
    ).to_string()
)

print("\n=== OBJECTIVE CONSISTENCY ===")

for model in ["class", "conservative"]:
    violations = (
        result[f"{model}_route_{model}_time_s"]
        >
        result[f"osm_route_{model}_time_s"] + 1e-6
    ).sum()

    print(
        f"{model} violations:",
        int(violations),
    )

print("\nSaved:", OUTPUT_PATH)
