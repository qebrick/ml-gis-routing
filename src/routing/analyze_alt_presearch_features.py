from pathlib import Path

import numpy as np
import osmnx as ox
import pandas as pd

GRAPH_PATH = Path(
    "data/graphs/porto_drive_conservative_refit.graphml"
)
INPUT_PATH = Path(
    "data/processed/adaptive_alt_ml_opportunity.parquet"
)
OUTPUT_PATH = Path(
    "data/processed/adaptive_alt_presearch_features.parquet"
)

G = ox.load_graphml(GRAPH_PATH)
df = pd.read_parquet(INPUT_PATH).copy()

# Normalize IDs without assuming GraphML node type.
lookup = {str(n): n for n in G.nodes}

def coords(series):
    ids = [lookup[str(n)] for n in series]
    lat = np.array([
        float(G.nodes[n]["y"]) for n in ids
    ])
    lon = np.array([
        float(G.nodes[n]["x"]) for n in ids
    ])
    return lat, lon

slat, slon = coords(df["origin"])
tlat, tlon = coords(df["destination"])

# Haversine straight-line distance.
r = 6371000.0

p1 = np.radians(slat)
p2 = np.radians(tlat)
dp = p2 - p1
dl = np.radians(tlon - slon)

a = (
    np.sin(dp / 2) ** 2
    + np.cos(p1) * np.cos(p2)
    * np.sin(dl / 2) ** 2
)

df["air_distance_m"] = (
    2 * r * np.arcsin(
        np.sqrt(np.clip(a, 0.0, 1.0))
    )
)

df["abs_delta_lat"] = np.abs(tlat - slat)
df["abs_delta_lon"] = np.abs(tlon - slon)

# Corrected candidate oracle: source-top-4 is
# included as an additional candidate.
df["oracle_plus_4"] = df[
    ["oracle_4", "source_top_4"]
].min(axis=1)

df["avoidable_expansions"] = (
    df["source_top_4"]
    - df["oracle_plus_4"]
)

df["avoidable_pct"] = (
    100 * df["avoidable_expansions"]
    / df["source_top_4"].clip(lower=1)
)

# Labels are diagnostic only; they must never
# be used as pre-search input features.
df["hard_route"] = (
    df["source_top_4"] >=
    df["source_top_4"].quantile(0.75)
)

df["large_opportunity"] = (
    df["avoidable_expansions"] >= 20
)

print("=== CORRECTED ORACLE ===")
print("Routes:", len(df))
print(
    "Source-top-4 expansions:",
    int(df["source_top_4"].sum())
)
print(
    "Oracle-plus expansions:",
    int(df["oracle_plus_4"].sum())
)
print(
    "Potential reduction (%):",
    round(
        100 * (
            1 - df["oracle_plus_4"].sum()
            / df["source_top_4"].sum()
        ),
        3
    )
)
print(
    "Routes with avoidable expansions:",
    int((df["avoidable_expansions"] > 0).sum())
)

print("\n=== PRE-SEARCH FEATURE ASSOCIATIONS ===")

for feature in [
    "air_distance_m",
    "abs_delta_lat",
    "abs_delta_lon",
]:
    rho_hard = df[feature].corr(
        df["source_top_4"],
        method="spearman"
    )
    rho_gain = df[feature].corr(
        df["avoidable_expansions"],
        method="spearman"
    )

    print(
        f"{feature:20s} "
        f"rho(expanded)={rho_hard:+.3f} "
        f"rho(avoidable)={rho_gain:+.3f}"
    )

print("\n=== BY AIR-DISTANCE QUARTILE ===")

df["distance_quartile"] = pd.qcut(
    df["air_distance_m"].rank(method="first"),
    q=4,
    labels=["Q1", "Q2", "Q3", "Q4"]
)

summary = (
    df.groupby(
        "distance_quartile",
        observed=True
    )
    .agg(
        routes=("pair_id", "size"),
        mean_air_m=("air_distance_m", "mean"),
        source_expanded=("source_top_4", "sum"),
        oracle_plus_expanded=("oracle_plus_4", "sum"),
        avoidable=("avoidable_expansions", "sum"),
        large_opportunities=("large_opportunity", "sum"),
    )
    .reset_index()
)

summary["potential_pct"] = (
    100 * summary["avoidable"]
    / summary["source_expanded"]
)

print(
    summary.round(2).to_string(index=False)
)

print("\n=== DISTANCE-ONLY GATE DIAGNOSTIC ===")

# Diagnostic screening only, not an independently
# validated policy. A final threshold must be
# chosen inside training folds.
threshold = df["air_distance_m"].quantile(0.75)

selected = df["air_distance_m"] >= threshold

total_gain = df["avoidable_expansions"].sum()
captured = df.loc[
    selected, "avoidable_expansions"
].sum()

print(
    "Distance threshold (m):",
    round(threshold, 1)
)
print(
    "Selected routes:",
    int(selected.sum())
)
print(
    "Share of routes selected (%):",
    round(100 * selected.mean(), 2)
)
print(
    "Share of potential captured (%):",
    round(
        100 * captured / total_gain
        if total_gain else 0.0,
        2
    )
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_parquet(OUTPUT_PATH, index=False)

print("\nSaved:", OUTPUT_PATH)
