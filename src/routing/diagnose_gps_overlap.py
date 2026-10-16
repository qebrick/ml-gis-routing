from pathlib import Path

import osmnx as ox
import pandas as pd

G = ox.load_graphml(
    "data/graphs/porto_drive_conservative_refit.graphml"
)

od = pd.read_parquet(
    "data/processed/gps_od_route_comparison.parquet"
)

tr = pd.read_parquet(
    "data/processed/hmm_edge_traversals_pilot.parquet"
)

print("=== DATA TYPES ===")

for name, df in [("OD", od), ("TRAVERSALS", tr)]:
    print(f"\n{name}")
    for col in [
        "trip_id",
        "hmm_fragment_id",
        "edge_u",
        "edge_v",
        "edge_key",
    ]:
        if col in df.columns:
            print(
                f"{col}: {df[col].dtype}; "
                f"example={repr(df[col].iloc[0])}"
            )

print("\n=== GRAPH EDGE TYPES ===")

u, v, key = next(iter(G.edges(keys=True)))

print("u:", repr(u), type(u).__name__)
print("v:", repr(v), type(v).__name__)
print("key:", repr(key), type(key).__name__)

print("\n=== FRAGMENT MATCHING ===")

od_pairs = set(
    zip(
        od["trip_id"].astype(str),
        od["hmm_fragment_id"].astype(str),
    )
)

tr_pairs = set(
    zip(
        tr["trip_id"].astype(str),
        tr["hmm_fragment_id"].astype(str),
    )
)

print("OD fragments:", len(od_pairs))
print("Traversal fragments:", len(tr_pairs))
print("Common fragments:", len(od_pairs & tr_pairs))

print("\n=== EDGE MATCHING ===")

sample = tr.loc[
    tr["traversed_length_m"] > 0
].head(20)

direct_matches = 0
numeric_matches = 0
string_matches = 0

for row in sample.itertuples(index=False):
    a, b, k = row.edge_u, row.edge_v, row.edge_key

    if G.has_edge(a, b, k):
        direct_matches += 1

    try:
        if G.has_edge(int(a), int(b), int(k)):
            numeric_matches += 1
    except (ValueError, TypeError):
        pass

    if G.has_edge(str(a), str(b), str(k)):
        string_matches += 1

print("Sample edges:", len(sample))
print("Direct matches:", direct_matches)
print("Numeric matches:", numeric_matches)
print("String matches:", string_matches)

print("\n=== FIRST OD FRAGMENT ===")

first = od.iloc[0]

part = tr.loc[
    (tr["trip_id"].astype(str) == str(first["trip_id"]))
    & (
        tr["hmm_fragment_id"].astype(str)
        == str(first["hmm_fragment_id"])
    )
]

print("OD trip:", repr(first["trip_id"]))
print("OD fragment:", repr(first["hmm_fragment_id"]))
print("Matching traversal rows:", len(part))

if not part.empty:
    print(
        part[
            [
                "edge_u",
                "edge_v",
                "edge_key",
                "traversed_length_m",
            ]
        ].head(5).to_string(index=False)
    )
