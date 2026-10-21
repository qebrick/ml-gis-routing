from src.paths import ROOT, DATA, benchmark_path, output_path
from pathlib import Path

import numpy as np
import pandas as pd

BASE = DATA
BENCH = benchmark_path()
TEST = BASE / "porto_heldout_600_od.parquet"
TRAIN = BASE / "ml_alt_budget_groupkfold.parquet"

OUT = output_path("porto_alt_workload_scenarios.csv")
REPORT = output_path("porto_alt_workload_scenarios.md")

B = 5000
SEED = 20261016
rng = np.random.default_rng(SEED)

bench = pd.read_parquet(BENCH)
test = pd.read_parquet(TEST)
train = pd.read_parquet(TRAIN)

# Reconstruct the original GPS distance terciles.
q1, q2 = np.quantile(
    train["air_distance_m"].to_numpy(),
    [1 / 3, 2 / 3],
)

print("=== DISTANCE BOUNDARIES ===")
print(f"Short/medium: {q1:.3f} m")
print(f"Medium/long: {q2:.3f} m")

# Validate the frozen held-out distance classification.
expected_bins = np.where(
    test["air_distance_m"] <= q1,
    "short",
    np.where(
        test["air_distance_m"] <= q2,
        "medium",
        "long",
    ),
)

assert np.array_equal(
    expected_bins,
    test["distance_bin"].to_numpy(),
)

# Verify that the benchmark contains all 3 repeats
# for both strategies and every OD.
subset = bench[
    bench["strategy"].isin(["alt2", "distance_gate"])
].copy()

assert len(subset) == 600 * 3 * 2
assert not subset.duplicated(
    ["repeat", "pair_id", "strategy"]
).any()

wide = subset.pivot(
    index=["repeat", "pair_id"],
    columns="strategy",
    values="total_ms",
).reset_index()

assert len(wide) == 1800
assert wide[["alt2", "distance_gate"]].notna().all().all()

wide["delta_us"] = (
    wide["distance_gate"] - wide["alt2"]
) * 1000.0

# Repeated measurements of the same OD are averaged.
per_od = (
    wide.groupby("pair_id", as_index=False)["delta_us"]
    .mean()
    .merge(
        test[["pair_id", "destination", "distance_bin"]],
        on="pair_id",
        validate="one_to_one",
    )
)

assert len(per_od) == 600

bins = ["short", "medium", "long"]
bin_values = {}
bin_clusters = {}

for name in bins:
    frame = per_od[
        per_od["distance_bin"] == name
    ].reset_index(drop=True)

    assert len(frame) == 200

    bin_values[name] = frame["delta_us"].to_numpy()

    # Resample destinations within each distance bin.
    # This preserves the intended workload proportions.
    bin_clusters[name] = [
        g.index.to_numpy()
        for _, g in frame.groupby("destination", sort=True)
    ]

means = np.array([
    bin_values[name].mean() for name in bins
])

print("\n=== OBSERVED STRATUM EFFECTS ===")
for name, value in zip(bins, means):
    print(f"{name:6s}: {value:+.3f} us/OD")

# One shared set of stratified bootstrap replicates
# is reused across all workload scenarios.
boot = np.empty((B, 3), dtype=np.float64)

for b in range(B):
    for j, name in enumerate(bins):
        groups = bin_clusters[name]
        choices = rng.integers(
            0, len(groups), size=len(groups)
        )
        indices = np.concatenate([
            groups[i] for i in choices
        ])
        boot[b, j] = bin_values[name][indices].mean()

def scenario(name, p_short, p_medium, p_long):
    p = np.array(
        [p_short, p_medium, p_long],
        dtype=np.float64,
    )
    assert np.isclose(p.sum(), 1.0)

    estimate = float(p @ means)
    samples = boot @ p
    low, high = np.quantile(samples, [0.025, 0.975])

    return {
        "scenario": name,
        "short_pct": 100 * p_short,
        "medium_pct": 100 * p_medium,
        "long_pct": 100 * p_long,
        "delta_us_per_od": estimate,
        "ci95_low_us": float(low),
        "ci95_high_us": float(high),
        "prob_gate_faster_bootstrap": float(
            np.mean(samples < 0)
        ),
    }

scenarios = [
    scenario("long_05", 0.475, 0.475, 0.05),
    scenario("long_10", 0.45, 0.45, 0.10),
    scenario("long_15", 0.425, 0.425, 0.15),
    scenario("long_20", 0.40, 0.40, 0.20),
    scenario("balanced_3_bins", 1/3, 1/3, 1/3),
    scenario("long_50", 0.25, 0.25, 0.50),
]

result = pd.DataFrame(scenarios)

print("\n=== WORKLOAD SCENARIOS ===")
print(result.round(3).to_string(index=False))

# Break-even fraction of long OD, assuming that
# short and medium have equal shares of the remainder.
nonlong = (means[0] + means[1]) / 2.0
long = means[2]

def break_even(nonlong_effect, long_effect):
    denominator = nonlong_effect - long_effect
    if denominator <= 0:
        return np.nan
    return nonlong_effect / denominator

point = break_even(nonlong, long)

nonlong_boot = (boot[:, 0] + boot[:, 1]) / 2.0
denom = nonlong_boot - boot[:, 2]

thresholds = np.full(B, np.nan)
valid = (
    (nonlong_boot > 0)
    & (boot[:, 2] < 0)
    & (denom > 0)
)
thresholds[valid] = nonlong_boot[valid] / denom[valid]

print("\n=== BREAK-EVEN LONG SHARE ===")
print(f"Point estimate: {100 * point:.3f}%")
print(
    "Valid bootstrap replicates:",
    int(valid.sum()),
    "/",
    B,
)

if valid.all():
    low, high = np.quantile(
        thresholds, [0.025, 0.975]
    )
    print(
        f"95% bootstrap interval: "
        f"[{100*low:.3f}%, {100*high:.3f}%]"
    )
else:
    print(
        "Some bootstrap replicates have no interior "
        "break-even point; inspect before interpretation."
    )

# GPS distance terciles were used to define bins.
# Consequently the original GPS sample is expected
# to be approximately one-third in each bin.
gps_d = train["air_distance_m"].to_numpy()
gps_counts = np.array([
    np.count_nonzero(gps_d <= q1),
    np.count_nonzero((gps_d > q1) & (gps_d <= q2)),
    np.count_nonzero(gps_d > q2),
])

gps_weights = gps_counts / gps_counts.sum()

gps_scenario = scenario(
    "original_gps_distance_mix",
    *gps_weights,
)

result = pd.concat(
    [result, pd.DataFrame([gps_scenario])],
    ignore_index=True,
)

print("\n=== ORIGINAL GPS DISTANCE MIX ===")
print("Counts:", gps_counts.tolist())
print("Weights:", np.round(gps_weights, 4).tolist())
print(
    "Estimated gate - ALT2:",
    f"{gps_scenario['delta_us_per_od']:+.3f} us/OD",
)

print(
    "Note: this is a reweighted synthetic-OD estimate, "
    "not a new GPS benchmark."
)

OUT.parent.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

result.to_csv(OUT, index=False)

lines = [
    "# Adaptive ALT: workload sensitivity",
    "",
    "Все результаты основаны на фиксированном тесте из 600 OD.",
    "Отрицательная разница означает ускорение distance-gate.",
    "",
    "## Сценарии",
    "",
    "```",
    result.round(3).to_string(index=False),
    "```",
    "",
    "## Точка безубыточности",
    "",
    f"Оценка доли long: {100*point:.3f}%.",
]

if valid.all():
    lines.append(
        f"95% bootstrap CI: "
        f"[{100*low:.3f}%, {100*high:.3f}%]."
    )

lines += [
    "",
    "## Ограничения",
    "",
    "- Это сценарный анализ, не новый независимый тест.",
    "- Условные распределения OD внутри групп расстояний",
    "  считаются такими же, как в синтетической выборке.",
    "- Доли short и medium при расчёте break-even равны.",
    "- Bootstrap выполняется по destination внутри групп.",
    "- Доли GPS по терцилям почти равны по определению;",
    "  они не характеризуют реальный рабочий поток запросов.",
    "- Никакие пороги маршрутизатора здесь не настраиваются.",
]

REPORT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print("\n=== SAVED ===")
print(OUT)
print(REPORT)
