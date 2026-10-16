import numpy as np
import osmnx as ox
import pandas as pd

DATA_PATH = "data/processed/ml_training_data.parquet"
GRAPH_PATH = "data/graphs/porto_drive.graphml"

data = pd.read_parquet(DATA_PATH)
G = ox.load_graphml(GRAPH_PATH)

graph_rows = []

for u, v, k, d in G.edges(keys=True, data=True):
    graph_rows.append({
        "edge_u": u,
        "edge_v": v,
        "edge_key": k,
        "edge_length_m": float(d["length"]),
        "highway": str(d.get("highway", "unknown")),
    })

graph = pd.DataFrame(graph_rows)

sampled_edges = data[
    ["edge_u", "edge_v", "edge_key"]
].drop_duplicates()

graph["sampled"] = (
    graph.merge(
        sampled_edges.assign(sampled=True),
        on=["edge_u", "edge_v", "edge_key"],
        how="left",
    )["sampled"]
    .fillna(False)
    .to_numpy()
)

print("=== SAMPLING BIAS ===")
print("Total graph edges:", len(graph))
print("Sampled unique edges:", int(graph["sampled"].sum()))

print("\n=== EDGE LENGTH DISTRIBUTION ===")

for name, values in [
    ("All graph edges", graph["edge_length_m"]),
    ("Sampled edges", graph.loc[graph["sampled"], "edge_length_m"]),
    ("Training rows", data["edge_length_m"]),
]:
    print(
        f"{name:18s} | "
        f"median={values.median():7.1f} m | "
        f"p90={values.quantile(0.9):7.1f} m | "
        f"mean={values.mean():7.1f} m"
    )

print("\n=== COVERAGE BY EDGE LENGTH ===")

bins = [0, 25, 50, 100, 200, 400, 1000, np.inf]
labels = [
    "0-25", "25-50", "50-100", "100-200",
    "200-400", "400-1000", "1000+"
]

graph["length_bin"] = pd.cut(
    graph["edge_length_m"],
    bins=bins,
    labels=labels,
    right=False,
)

coverage = graph.groupby(
    "length_bin",
    observed=True,
).agg(
    graph_edges=("sampled", "size"),
    sampled_edges=("sampled", "sum"),
)

coverage["coverage_pct"] = (
    100 * coverage["sampled_edges"]
    / coverage["graph_edges"]
)

print(coverage.to_string())

print("\n=== HIGHWAY DISTRIBUTION ===")

graph["highway_group"] = graph["highway"].str.replace(
    r"^\['([^']+)'.*$",
    r"\1",
    regex=True,
)

data["highway_group"] = data["highway"].astype(str)

all_share = (
    graph["highway_group"]
    .value_counts(normalize=True)
    .rename("graph_share")
)

sample_share = (
    data["highway_group"]
    .value_counts(normalize=True)
    .rename("sample_share")
)

comparison = pd.concat(
    [all_share, sample_share],
    axis=1,
).fillna(0)

comparison["graph_pct"] = 100 * comparison["graph_share"]
comparison["sample_pct"] = 100 * comparison["sample_share"]

print(
    comparison[
        ["graph_pct", "sample_pct"]
    ]
    .sort_values("sample_pct", ascending=False)
    .head(15)
    .round(2)
    .to_string()
)

print("\n=== PARTIAL EDGE FRACTION ===")

fraction = (
    data["traversed_length_m"]
    / data["edge_length_m"]
)

print(
    fraction.describe(
        percentiles=[0.1, 0.5, 0.9, 0.95]
    ).to_string()
)

print(
    "Fraction > 1:",
    int((fraction > 1.001).sum()),
)

print(
    "Fraction >= 0.8:",
    int((fraction >= 0.8).sum()),
)
