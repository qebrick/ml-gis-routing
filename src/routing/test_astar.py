import math
import random

import networkx as nx
import osmnx as ox

GRAPH_PATH = "data/graphs/porto_drive_routing.graphml"

WEIGHT = "osm_travel_time_s"
MAX_SPEED_KMH = 90.0
EARTH_RADIUS_M = 6371008.8
RANDOM_SEED = 42
N_TESTS = 20

G = ox.load_graphml(GRAPH_PATH)
# GraphML may restore custom edge attributes as strings.
# NetworkX requires numeric weights for Dijkstra and A*.
for u, v, k, data in G.edges(keys=True, data=True):
    data[WEIGHT] = float(data[WEIGHT])

def haversine_m(lat1, lon1, lat2, lon2):
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return (
        2
        * EARTH_RADIUS_M
        * math.asin(min(1.0, math.sqrt(a)))
    )


def heuristic(u, target):
    a = G.nodes[u]
    b = G.nodes[target]

    distance_m = haversine_m(
        float(a["y"]),
        float(a["x"]),
        float(b["y"]),
        float(b["x"]),
    )

    return 3.6 * distance_m / MAX_SPEED_KMH


def route_cost(path):
    total = 0.0

    for u, v in zip(path[:-1], path[1:]):
        # MultiDiGraph: choose the cheapest parallel edge.
        total += min(
            float(d[WEIGHT])
            for d in G[u][v].values()
        )

    return total


print("=== HEURISTIC VALIDATION ===")

violations = []

for u, v, k, d in G.edges(keys=True, data=True):
    a = G.nodes[u]
    b = G.nodes[v]

    straight_m = haversine_m(
        float(a["y"]),
        float(a["x"]),
        float(b["y"]),
        float(b["x"]),
    )

    length_m = float(d["length"])

    if length_m + 0.1 < straight_m:
        violations.append(
            (u, v, k, length_m, straight_m)
        )

print(
    "Edges shorter than straight-line distance:",
    len(violations),
)

if violations:
    print("Examples:", violations[:5])
    print(
        "Using zero heuristic to guarantee correctness."
    )

    def safe_heuristic(u, target):
        return 0.0

else:
    safe_heuristic = heuristic


print("\n=== A* VS DIJKSTRA ===")

random.seed(RANDOM_SEED)

nodes = list(G.nodes())
completed = 0
attempts = 0
max_attempts = N_TESTS * 20

while completed < N_TESTS and attempts < max_attempts:
    attempts += 1

    source, target = random.sample(nodes, 2)

    try:
        dijkstra_path = nx.shortest_path(
            G,
            source,
            target,
            weight=WEIGHT,
            method="dijkstra",
        )

        astar_path = nx.astar_path(
            G,
            source,
            target,
            heuristic=safe_heuristic,
            weight=WEIGHT,
        )

    except nx.NetworkXNoPath:
        continue

    dijkstra_cost = route_cost(dijkstra_path)
    astar_cost = route_cost(astar_path)

    difference = abs(dijkstra_cost - astar_cost)

    if difference > 1e-6:
        raise RuntimeError(
            f"A* mismatch: {difference:.6f} seconds"
        )

    completed += 1

    print(
        f"Test {completed:2d}: "
        f"Dijkstra={dijkstra_cost:8.2f}s | "
        f"A*={astar_cost:8.2f}s | "
        f"difference={difference:.9f}s"
    )

print("\n=== RESULT ===")
print("Successful tests:", completed)
print("Attempts:", attempts)

if completed < N_TESTS:
    raise RuntimeError(
        "Not enough reachable source-target pairs."
    )

print("A* and Dijkstra agree on all tested pairs.")
