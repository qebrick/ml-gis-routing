from pathlib import Path

import networkx as nx
import osmnx as ox
import pandas as pd

GRAPH_PATH = Path("data/graphs/porto_drive.graphml")
INPUT_PATH = Path("data/processed/map_matching_clean.parquet")
OUTPUT_PATH = Path("data/processed/edge_transition_diagnostics.parquet")

G = ox.load_graphml(GRAPH_PATH)
df = pd.read_parquet(INPUT_PATH)
df = df.sort_values(["segment_id", "point_index"]).reset_index(drop=True)

records = []

for segment_id, group in df.groupby("segment_id", sort=False):
    rows = list(group.itertuples(index=False))

    for first, second in zip(rows[:-1], rows[1:]):
        u1, v1 = first.edge_u, first.edge_v
        u2, v2 = second.edge_u, second.edge_v

        same_edge = (
            u1 == u2
            and v1 == v2
            and first.edge_key == second.edge_key
        )

        # Directed connectivity between endpoints.
        try:
            path_length = nx.shortest_path_length(
                G,
                source=v1,
                target=u2,
                weight="length",
            )
            reachable = True
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path_length = None
            reachable = False

        records.append({
            "segment_id": segment_id,
            "trip_id": first.trip_id,
            "point_index_1": first.point_index,
            "point_index_2": second.point_index,
            "same_edge": same_edge,
            "reachable": reachable,
            "endpoint_path_length_m": path_length,
        })

result = pd.DataFrame(records)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result.to_parquet(OUTPUT_PATH, index=False)

print("=== EDGE TRANSITION DIAGNOSTICS ===")
print("Transitions:", len(result))
print("Same-edge transitions:", int(result["same_edge"].sum()))
print(
    "Same-edge share:",
    f"{result['same_edge'].mean() * 100:.2f}%",
)
print("Reachable transitions:", int(result["reachable"].sum()))
print(
    "Reachable share:",
    f"{result['reachable'].mean() * 100:.2f}%",
)
print(
    "Unreachable transitions:",
    int((~result["reachable"]).sum()),
)

print("\nEndpoint path length statistics (m):")
print(
    result["endpoint_path_length_m"]
    .dropna()
    .describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    .to_string()
)

print("\nSaved:", OUTPUT_PATH)
