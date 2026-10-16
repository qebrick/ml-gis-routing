import osmnx as ox
import pandas as pd

PATH = "data/graphs/porto_drive_osm_weighted.graphml"

G = ox.load_graphml(PATH)

rows = []

for u, v, k, d in G.edges(keys=True, data=True):
    length = float(d["length"])

    if length <= 0:
        rows.append({
            "u": u,
            "v": v,
            "key": k,
            "length_m": length,
            "highway": str(d.get("highway")),
            "osmid": str(d.get("osmid")),
            "self_loop": u == v,
            "has_geometry": "geometry" in d,
        })

df = pd.DataFrame(rows)

print("=== ZERO-LENGTH EDGES ===")
print("Count:", len(df))

if len(df):
    print("\nSelf-loops:", int(df["self_loop"].sum()))

    print("\nHighway distribution:")
    print(df["highway"].value_counts().to_string())

    print("\nExamples:")
    print(df.head(20).to_string(index=False))
