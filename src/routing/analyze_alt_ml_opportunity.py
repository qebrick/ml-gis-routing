from pathlib import Path
import numpy as np
import pandas as pd

INPUT = Path(
    "data/processed/adaptive_alt_landmark_diagnostics.parquet"
)
OUTPUT = Path(
    "data/processed/adaptive_alt_ml_opportunity.parquet"
)

df = pd.read_parquet(INPUT)

wide = df.pivot(
    index="pair_id",
    columns="strategy",
    values="expanded",
).reset_index()

od = (
    df.drop_duplicates("pair_id")
    [["pair_id", "origin", "destination"]]
)

result = wide.merge(
    od,
    on="pair_id",
    validate="one_to_one",
)

result["source_gain"] = (
    result["fixed_4"] - result["source_top_4"]
)

result["oracle_gain"] = (
    result["source_top_4"] - result["oracle_4"]
)

result["oracle_gain_pct"] = (
    100 * result["oracle_gain"]
    / result["source_top_4"].clip(lower=1)
)

result["remaining_fraction"] = (
    result["oracle_gain"]
    / result["source_top_4"].clip(lower=1)
)

# Route difficulty proxy: expansions with fixed ALT-4.
result["difficulty_group"] = pd.qcut(
    result["fixed_4"].rank(method="first"),
    q=4,
    labels=[
        "Q1 easiest",
        "Q2",
        "Q3",
        "Q4 hardest",
    ],
)

print("=== OVERALL OPPORTUNITY ===")
print("Routes:", len(result))
print(
    "Source-top-4 expansions:",
    int(result["source_top_4"].sum()),
)
print(
    "Oracle-4 expansions:",
    int(result["oracle_4"].sum()),
)
print(
    "Potential additional reduction (%):",
    round(
        100 * (
            1 - result["oracle_4"].sum()
            / result["source_top_4"].sum()
        ),
        3,
    ),
)

print("\n=== ROUTE-LEVEL OPPORTUNITY ===")
print(
    "Oracle better:",
    int((result["oracle_gain"] > 0).sum()),
)
print(
    "Oracle equal:",
    int((result["oracle_gain"] == 0).sum()),
)
print(
    "Oracle worse:",
    int((result["oracle_gain"] < 0).sum()),
)

print("\n=== BY ROUTE DIFFICULTY ===")

grouped = (
    result.groupby(
        "difficulty_group",
        observed=True,
    )
    .agg(
        routes=("pair_id", "size"),
        fixed_expanded=("fixed_4", "sum"),
        source_expanded=("source_top_4", "sum"),
        oracle_expanded=("oracle_4", "sum"),
        remaining_gain=("oracle_gain", "sum"),
    )
    .reset_index()
)

grouped["potential_reduction_pct"] = (
    100 * grouped["remaining_gain"]
    / grouped["source_expanded"]
)

print(grouped.round(3).to_string(index=False))

print("\n=== CONCENTRATION OF POTENTIAL ===")

positive = (
    result["oracle_gain"]
    .clip(lower=0)
    .sort_values(ascending=False)
    .to_numpy()
)

total = positive.sum()

for n in [5, 10, 20, 40]:
    share = (
        100 * positive[:n].sum() / total
        if total else 0.0
    )
    print(
        f"Top {n} routes: "
        f"{share:.2f}% of remaining gain"
    )

print("\n=== TOP 10 OPPORTUNITIES ===")

columns = [
    "pair_id",
    "origin",
    "destination",
    "fixed_4",
    "source_top_4",
    "oracle_4",
    "oracle_gain",
    "oracle_gain_pct",
]

print(
    result.sort_values(
        "oracle_gain",
        ascending=False,
    )[columns]
    .head(10)
    .round(3)
    .to_string(index=False)
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)
result.to_parquet(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
