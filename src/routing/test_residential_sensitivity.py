from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

GRAPH_PATH = Path(
    "data/graphs/porto_drive_class_weighted.graphml"
)
PAIRS_PATH = Path(
    "data/processed/class_astar_comparison.parquet"
)
OUTPUT_PATH = Path(
    "data/processed/residential_sensitivity.parquet"
)

G = ox.load_graphml(GRAPH_PATH)
pairs = pd.read_parquet(PAIRS_PATH)

def weight_function(attribute):
    def weight(u, v, parallel_edges):
        return min(
            float(attrs[attribute])
            for attrs in parallel_edges.values()
        )
    return weight

# Conservative scenario: remove residential discount.
for _, _, _, attrs in G.edges(keys=True, data=True):
    if attrs["class_road_group"] == "residential":
        attrs["conservative_time_s"] = float(
            attrs["osm_travel_time_s"]
        )
    else:
        attrs["conservative_time_s"] = float(
            attrs["class_travel_time_s"]
        )

objectives = {
    "osm": "osm_travel_time_s",
    "class": "class_travel_time_s",
    "conservative": "conservative_time_s",
}

def route_metrics(path, objective):
    length = 0.0
    residential_length = 0.0
    costs = {name: 0.0 for name in objectives}

    selected_edges = []

    for u, v in zip(path[:-1], path[1:]):
        key, attrs = min(
            G[u][v].items(),
            key=lambda item: float(
                item[1][objective]
            ),
        )

        selected_edges.append((u, v, key))

        edge_length = float(attrs["length"])
        length += edge_length

        if attrs["class_road_group"] == "residential":
            residential_length += edge_length

        for name, attribute in objectives.items():
            costs[name] += float(attrs[attribute])

    return {
        "edges": tuple(selected_edges),
        "distance_m": length,
        "residential_share": (
            residential_length / length
            if length > 0 else 0.0
        ),
        **{
            f"{name}_time_s": value
            for name, value in costs.items()
        },
    }

records = []

for pair in pairs.itertuples(index=False):
    routes = {}

    for name, attribute in objectives.items():
        path = nx.shortest_path(
            G,
            source=pair.origin,
            target=pair.destination,
            weight=weight_function(attribute),
        )

        routes[name] = route_metrics(
            path, attribute
        )

    osm = routes["osm"]
    original = routes["class"]
    conservative = routes["conservative"]

    records.append({
        "pair_id": pair.pair_id,
        "class_changed": (
            original["edges"] != osm["edges"]
        ),
        "conservative_changed": (
            conservative["edges"] != osm["edges"]
        ),
        "class_residential_delta_pp": (
            100 * (
                original["residential_share"]
                - osm["residential_share"]
            )
        ),
        "conservative_residential_delta_pp": (
            100 * (
                conservative["residential_share"]
                - osm["residential_share"]
            )
        ),
        "class_distance_change_pct": (
            100 * (
                original["distance_m"]
                / osm["distance_m"] - 1
            )
        ),
        "conservative_distance_change_pct": (
            100 * (
                conservative["distance_m"]
                / osm["distance_m"] - 1
            )
        ),
        "class_osm_time_change_pct": (
            100 * (
                original["osm_time_s"]
                / osm["osm_time_s"] - 1
            )
        ),
        "conservative_osm_time_change_pct": (
            100 * (
                conservative["osm_time_s"]
                / osm["osm_time_s"] - 1
            )
        ),
        "class_objective_gain_pct": (
            100 * (
                osm["class_time_s"]
                - original["class_time_s"]
            ) / osm["class_time_s"]
        ),
        "conservative_objective_gain_pct": (
            100 * (
                osm["conservative_time_s"]
                - conservative["conservative_time_s"]
            ) / osm["conservative_time_s"]
        ),
    })

result = pd.DataFrame(records)

OUTPUT_PATH.parent.mkdir(
    parents=True, exist_ok=True
)
result.to_parquet(OUTPUT_PATH, index=False)

print("=== ROUTE CHANGE COMPARISON ===")
print("OD pairs:", len(result))

for scenario in ["class", "conservative"]:
    changed = result[f"{scenario}_changed"]
    subset = result.loc[changed]

    print(f"\n{scenario.upper()}")
    print("Changed routes:", int(changed.sum()))
    print(
        "Mean residential change (all), pp:",
        round(
            result[
                f"{scenario}_residential_delta_pp"
            ].mean(), 3
        )
    )

    if len(subset):
        print(
            "Mean residential change (changed), pp:",
            round(
                subset[
                    f"{scenario}_residential_delta_pp"
                ].mean(), 3
            )
        )
        print(
            "Mean distance change (changed), %:",
            round(
                subset[
                    f"{scenario}_distance_change_pct"
                ].mean(), 3
            )
        )
        print(
            "Mean OSM time change (changed), %:",
            round(
                subset[
                    f"{scenario}_osm_time_change_pct"
                ].mean(), 3
            )
        )

    print(
        "Mean own-objective gain (all), %:",
        round(
            result[
                f"{scenario}_objective_gain_pct"
            ].mean(), 3
        )
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

print("\nSaved:", OUTPUT_PATH)
