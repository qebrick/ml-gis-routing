import osmnx as ox
import pandas as pd

GRAPH_PATH = "data/graphs/porto_drive.graphml"

G = ox.load_graphml(GRAPH_PATH)

fields = [
    "length",
    "highway",
    "maxspeed",
    "lanes",
    "oneway",
    "name",
    "bridge",
    "tunnel",
    "junction",
]

records = []

for u, v, k, data in G.edges(keys=True, data=True):
    row = {
        "edge_u": u,
        "edge_v": v,
        "edge_key": k,
    }

    for field in fields:
        row[field] = data.get(field)

    records.append(row)

edges = pd.DataFrame(records)

print("=== EDGE FEATURE COVERAGE ===")
print("Total directed edges:", len(edges))

for field in fields:
    non_null = edges[field].notna().sum()
    coverage = non_null / len(edges) * 100

    print(
        f"{field:12s}: "
        f"{non_null:5d} / {len(edges)} "
        f"({coverage:.1f}%)"
    )

print("\n=== HIGHWAY VALUES ===")
print(
    edges["highway"]
    .astype(str)
    .value_counts()
    .head(15)
    .to_string()
)

print("\n=== MAXSPEED VALUES ===")
print(
    edges["maxspeed"]
    .astype(str)
    .value_counts()
    .head(15)
    .to_string()
)

print("\n=== LANES VALUES ===")
print(
    edges["lanes"]
    .astype(str)
    .value_counts()
    .head(15)
    .to_string()
)

print("\n=== ONEWAY VALUES ===")
print(
    edges["oneway"]
    .astype(str)
    .value_counts()
    .to_string()
)
