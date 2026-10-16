from pathlib import Path
import hashlib

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

GRAPH = Path("data/graphs/porto_drive_conservative_refit.graphml")
GPS = Path("data/processed/gps_od_route_comparison.parquet")
OUT = Path("data/processed/porto_heldout_600_od.parquet")

SEED = 20261016
PER_BIN = 200
MAX_ATTEMPTS = 300000

rng = np.random.default_rng(SEED)

print("=== LOADING ===")
G = ox.load_graphml(GRAPH)

# Build a directed graph with minimum travel-time
# parallel edges, consistent with previous benchmarks.
H = nx.DiGraph()
H.add_nodes_from(G.nodes(data=True))

for u, v, data in G.edges(data=True):
    if u == v:
        continue
    w = float(data["osm_travel_time_s"])
    if not H.has_edge(u, v) or w < H[u][v]["weight"]:
        H.add_edge(u, v, weight=w)

nodes = list(H.nodes)
lookup = {str(n): n for n in nodes}

lat = np.array(
    [float(H.nodes[n]["y"]) for n in nodes],
    dtype=np.float64
)
lon = np.array(
    [float(H.nodes[n]["x"]) for n in nodes],
    dtype=np.float64
)

node_index = {n: i for i, n in enumerate(nodes)}

def haversine(i, j):
    p1 = np.radians(lat[i])
    p2 = np.radians(lat[j])
    dp = p2 - p1
    dl = np.radians(lon[j] - lon[i])
    a = (
        np.sin(dp / 2.0) ** 2
        + np.cos(p1) * np.cos(p2)
        * np.sin(dl / 2.0) ** 2
    )
    return float(
        12742000.0
        * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    )

gps = pd.read_parquet(GPS)
original_pairs = set()
gps_distances = []

for row in gps.itertuples():
    s = lookup[str(row.origin)]
    t = lookup[str(row.destination)]
    original_pairs.add((str(s), str(t)))
    gps_distances.append(
        haversine(node_index[s], node_index[t])
    )

q1, q2 = np.quantile(gps_distances, [1 / 3, 2 / 3])

def distance_bin(d):
    if d <= q1:
        return 0
    if d <= q2:
        return 1
    return 2

print("Graph nodes:", H.number_of_nodes())
print("Graph edges:", H.number_of_edges())
print("GPS OD:", len(gps))
print("GPS distance terciles (m):", round(q1, 2), round(q2, 2))

# Directed strongly connected components:
# sampling within the same SCC guarantees reachability.
sccs = [
    list(c)
    for c in nx.strongly_connected_components(H)
    if len(c) >= 2
]
sccs.sort(key=len, reverse=True)

largest = sccs[0]
largest_indices = np.array(
    [node_index[n] for n in largest],
    dtype=np.int64
)

print("Largest SCC nodes:", len(largest))
print(
    "Largest SCC share:",
    round(len(largest) / H.number_of_nodes(), 4)
)

# Restrict to largest SCC to avoid a strong
# reachability-selection bias from tiny components.
counts = [0, 0, 0]
selected = set()
rows = []

attempts = 0

while min(counts) < PER_BIN and attempts < MAX_ATTEMPTS:
    attempts += 1

    i, j = rng.choice(
        largest_indices,
        size=2,
        replace=False,
    )

    s = nodes[int(i)]
    t = nodes[int(j)]

    key = (str(s), str(t))

    if key in original_pairs or key in selected:
        continue

    d = haversine(int(i), int(j))
    b = distance_bin(d)

    if counts[b] >= PER_BIN:
        continue

    selected.add(key)
    counts[b] += 1

    rows.append({
        "origin": str(s),
        "destination": str(t),
        "air_distance_m": d,
        "distance_bin": [
            "short", "medium", "long"
        ][b],
        "distance_bin_id": b,
        "sample_seed": SEED,
    })

if counts != [PER_BIN, PER_BIN, PER_BIN]:
    raise RuntimeError(
        f"Insufficient sampled pairs: {counts}; "
        f"attempts={attempts}"
    )

test = pd.DataFrame(rows)
test = test.sort_values(
    ["distance_bin_id", "origin", "destination"]
).reset_index(drop=True)

test.insert(0, "pair_id", np.arange(len(test)))

# Explicitly verify directed reachability.
# We already sampled from the same SCC, but
# validate it to catch graph-processing mistakes.
for row in test.itertuples():
    s = lookup[row.origin]
    t = lookup[row.destination]
    if not nx.has_path(H, s, t):
        raise AssertionError(
            f"Unreachable OD: {s} -> {t}"
        )

assert len(test) == 600
assert test[["origin", "destination"]].duplicated().sum() == 0
assert not any(
    (row.origin, row.destination) in original_pairs
    for row in test.itertuples()
)

OUT.parent.mkdir(parents=True, exist_ok=True)
test.to_parquet(OUT, index=False)

# Hash the actual OD content, not parquet metadata.
content = "\n".join(
    f"{r.origin},{r.destination}"
    for r in test.itertuples()
)
digest = hashlib.sha256(
    content.encode("utf-8")
).hexdigest()

print("\n=== HELD-OUT SAMPLE ===")
print("Total OD:", len(test))
print("Unique origins:", test["origin"].nunique())
print("Unique destinations:", test["destination"].nunique())
print("Attempts:", attempts)
print("Overlap with GPS OD:", 0)
print("Unreachable OD:", 0)

print("\n=== DISTANCE BINS ===")
print(
    test.groupby("distance_bin", sort=False)
    ["air_distance_m"]
    .agg(["count", "min", "median", "max"])
    .round(2)
    .to_string()
)

print("\n=== FROZEN TEST SET ===")
print("SHA256:", digest)
print("Saved:", OUT)
