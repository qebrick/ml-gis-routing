from pathlib import Path

import osmnx as ox

INPUT_PATH = Path(
    "data/graphs/porto_drive_osm_weighted.graphml"
)
OUTPUT_PATH = Path(
    "data/graphs/porto_drive_routing.graphml"
)

G = ox.load_graphml(INPUT_PATH)

print("=== ROUTING GRAPH PREPARATION ===")
print("Original nodes:", G.number_of_nodes())
print("Original edges:", G.number_of_edges())

self_loops = [
    (u, v, k)
    for u, v, k in G.edges(keys=True)
    if u == v
]

print("Self-loops:", len(self_loops))

G.remove_edges_from(self_loops)

invalid_weights = [
    (u, v, k)
    for u, v, k, d in G.edges(keys=True, data=True)
    if float(d["osm_travel_time_s"]) <= 0
]

if invalid_weights:
    raise RuntimeError(
        f"Invalid routing weights: {len(invalid_weights)}"
    )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
ox.save_graphml(G, OUTPUT_PATH)

print("Routing nodes:", G.number_of_nodes())
print("Routing edges:", G.number_of_edges())
print("Invalid weights:", len(invalid_weights))
print("Saved:", OUTPUT_PATH)
