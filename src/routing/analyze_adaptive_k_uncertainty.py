from pathlib import Path
import numpy as np
import pandas as pd

PATH = Path(
    "data/processed/adaptive_alt_precomputed_benchmark.parquet"
)

df = pd.read_parquet(PATH)

# Attach destination to each OD, using the same
# ordering as the benchmark.
od = pd.read_parquet(
    "data/processed/gps_od_route_comparison.parquet"
).reset_index(drop=True)

target_map = dict(enumerate(
    od["destination"].astype(str)
))

df["target"] = df["pair_id"].map(target_map)

wide = df[df["k"].isin([2, 4])].pivot(
    index=["repeat", "pair_id", "target"],
    columns="k",
    values="total_ms",
).reset_index()

wide["delta_ms"] = wide[4] - wide[2]

print("=== PAIRED K4 VS K2 ===")
print("Paired observations:", len(wide))
print("Mean difference per OD (ms):",
      round(wide["delta_ms"].mean(), 5))
print("Median difference per OD (ms):",
      round(wide["delta_ms"].median(), 5))
print("K4 faster observations:",
      int((wide["delta_ms"] < 0).sum()))
print("K2 faster observations:",
      int((wide["delta_ms"] > 0).sum()))

print("\n=== BY REPEAT ===")
repeat = wide.groupby("repeat")["delta_ms"].sum()
print(repeat.round(3).to_string())
print(
    "K4 faster repeats:",
    int((repeat < 0).sum()),
    "of",
    len(repeat),
)

# First average repetitions within each OD.
per_od = (
    wide.groupby(["pair_id", "target"], as_index=False)
    ["delta_ms"].mean()
)

# Cluster bootstrap by destination.
groups = [
    part["delta_ms"].to_numpy()
    for _, part in per_od.groupby("target")
]

rng = np.random.default_rng(42)
boot = np.empty(10000)

for b in range(len(boot)):
    selected = rng.integers(
        0, len(groups), size=len(groups)
    )
    values = np.concatenate(
        [groups[j] for j in selected]
    )
    boot[b] = values.mean()

lo, hi = np.percentile(boot, [2.5, 97.5])

print("\n=== TARGET-CLUSTER BOOTSTRAP ===")
print(
    "Mean delta per OD (ms):",
    round(per_od["delta_ms"].mean(), 5),
)
print(
    "95% bootstrap CI (ms):",
    f"[{lo:.5f}, {hi:.5f}]",
)
print(
    "Probability delta < 0 in bootstrap:",
    round(float((boot < 0).mean()), 4),
)

print("\nInterpretation:")
print("Negative delta favors k=4.")
print("Positive delta favors k=2.")
