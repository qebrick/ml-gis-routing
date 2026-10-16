from pathlib import Path

import pandas as pd
import osmnx as ox

VALIDATION_PATH = Path(
    "data/processed/osm_interval_validation.parquet"
)
TRAVERSALS_PATH = Path(
    "data/processed/hmm_edge_traversals_pilot.parquet"
)
GRAPH_PATH = Path(
    "data/graphs/porto_drive_osm_weighted.graphml"
)

df = pd.read_parquet(VALIDATION_PATH)
traversals = pd.read_parquet(TRAVERSALS_PATH)
G = ox.load_graphml(GRAPH_PATH)

df["osm_time_ratio"] = df["osm_time_s"] / df["observed_time_s"]
outliers = df.nlargest(10, "osm_time_ratio")

print("=== OUTLIER ROUTE COMPOSITION ===")

for _, row in outliers.iterrows():
    trip_id = row["trip_id"]
    point_index = row["point_index"]

    route = traversals.loc[
        (traversals["trip_id"] == trip_id)
        & (traversals["point_index"] == point_index)
    ].sort_values("edge_order")

    print("\n" + "=" * 65)
    print("Trip:", trip_id)
    print("Point index:", point_index)
    print("Route distance:", round(row["route_distance_m"], 2))
    print("HMM speed:", round(row["target_speed_kmh"], 2))
    print("OSM time:", round(row["osm_time_s"], 2))
    print("Traversals:", len(route))

    for _, edge in route.iterrows():
        u = edge["edge_u"]
        v = edge["edge_v"]
        k = edge["edge_key"]

        if not G.has_edge(u, v, k):
            print("MISSING EDGE:", u, v, k)
            continue

        attrs = G[u][v][k]

        print(
            f"  {u} -> {v} [{k}] "
            f"length={edge['traversed_length_m']:.1f}m "
            f"highway={attrs.get('highway')} "
            f"osm_speed={attrs.get('osm_speed_kmh')}km/h "
            f"maxspeed={attrs.get('maxspeed')}"
        )

