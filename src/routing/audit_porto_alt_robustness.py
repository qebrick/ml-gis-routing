from src.paths import ROOT, DATA, benchmark_path, output_path
from pathlib import Path
import numpy as np
import pandas as pd

BASE = DATA
INPUT = benchmark_path()
OUTPUT = output_path("porto_alt_robustness_audit.csv")
REPORT = output_path("porto_alt_robustness_audit.md")

df = pd.read_parquet(INPUT)

required = {
    "repeat", "pair_id", "destination",
    "distance_bin", "strategy", "total_ms"
}
assert required.issubset(df.columns)

strategies = {
    "alt2", "alt4", "distance_gate", "ml_tree"
}
assert set(df["strategy"]) == strategies
assert len(df) == 600 * 3 * 4
assert not df.duplicated(
    ["repeat", "pair_id", "strategy"]
).any()

wide = df.pivot(
    index=["repeat", "pair_id", "destination", "distance_bin"],
    columns="strategy",
    values="total_ms",
).reset_index()

assert len(wide) == 1800
assert wide[["alt2", "alt4", "distance_gate", "ml_tree"]].notna().all().all()

wide["gate_delta_ms"] = wide["distance_gate"] - wide["alt2"]
wide["ml_delta_ms"] = wide["ml_tree"] - wide["alt2"]

per_od = (
    wide.groupby(
        ["pair_id", "destination", "distance_bin"],
        as_index=False
    )
    .agg(
        gate_delta_ms=("gate_delta_ms", "mean"),
        ml_delta_ms=("ml_delta_ms", "mean"),
    )
)

assert len(per_od) == 600

print("=== PER-REPEAT EFFECT ===")
repeat_summary = (
    wide.groupby("repeat")
    .agg(
        gate_delta_total_ms=("gate_delta_ms", "sum"),
        gate_delta_mean_us=("gate_delta_ms", lambda x: 1000*x.mean()),
        ml_delta_total_ms=("ml_delta_ms", "sum"),
        ml_delta_mean_us=("ml_delta_ms", lambda x: 1000*x.mean()),
    )
)
print(repeat_summary.round(4).to_string())

print("\n=== CONCENTRATION OF GATE SAVINGS ===")

# Positive savings: ALT-2 is slower than gate.
per_od["gate_saving_ms"] = -per_od["gate_delta_ms"]

positive = per_od[
    per_od["gate_saving_ms"] > 0
].sort_values("gate_saving_ms", ascending=False)

gross_positive = positive["gate_saving_ms"].sum()
net_saving = per_od["gate_saving_ms"].sum()

print("Routes where gate is faster:", len(positive))
print("Routes where gate is slower:", int(
    (per_od["gate_saving_ms"] < 0).sum()
))
print("Gross positive savings (ms):", round(gross_positive, 4))
print("Net savings (ms):", round(net_saving, 4))

for n in [1, 3, 5, 10, 20]:
    top = positive.head(n)["gate_saving_ms"].sum()
    print(
        f"Top {n:2d}: saving={top:.4f} ms, "
        f"share_of_gross={100*top/gross_positive:.2f}%, "
        f"share_of_net={100*top/net_saving:.2f}%"
    )

print("\n=== LEAVE-TOP-OUT SENSITIVITY ===")

rows = []

for n in [0, 1, 3, 5, 10, 20]:
    excluded = set(positive.head(n)["pair_id"])
    subset = per_od[
        ~per_od["pair_id"].isin(excluded)
    ]

    mean_us = subset["gate_delta_ms"].mean() * 1000
    total_ms = subset["gate_delta_ms"].sum()

    rows.append({
        "analysis": "leave_top_out",
        "group": f"remove_top_{n}",
        "n_od": len(subset),
        "mean_delta_us": mean_us,
        "total_delta_ms": total_ms,
        "ci95_low_us": np.nan,
        "ci95_high_us": np.nan,
    })

    print(
        f"Remove top {n:2d}: "
        f"n={len(subset)}, "
        f"mean={mean_us:+.3f} us/OD, "
        f"total={total_ms:+.3f} ms"
    )

print("\n=== STRATIFIED DESTINATION BOOTSTRAP ===")

rng = np.random.default_rng(20261016)
B = 5000

for distance_bin in ["short", "medium", "long"]:
    subset = per_od[
        per_od["distance_bin"] == distance_bin
    ].reset_index(drop=True)

    groups = [
        g.index.to_numpy()
        for _, g in subset.groupby(
            "destination", sort=True
        )
    ]

    values = subset["gate_delta_ms"].to_numpy()
    estimates = np.empty(B)

    for b in range(B):
        picks = rng.integers(
            0, len(groups), size=len(groups)
        )
        indices = np.concatenate([
            groups[j] for j in picks
        ])
        estimates[b] = values[indices].mean() * 1000

    low, high = np.quantile(
        estimates, [0.025, 0.975]
    )
    mean_us = values.mean() * 1000

    rows.append({
        "analysis": "distance_bin_bootstrap",
        "group": distance_bin,
        "n_od": len(subset),
        "mean_delta_us": mean_us,
        "total_delta_ms": values.sum(),
        "ci95_low_us": low,
        "ci95_high_us": high,
    })

    print(
        f"{distance_bin:6s}: "
        f"mean={mean_us:+.3f} us/OD, "
        f"95% CI=[{low:+.3f}, {high:+.3f}]"
    )

print("\n=== TOP 10 MOST INFLUENTIAL OD ===")

top10 = (
    positive.head(10)[
        ["pair_id", "distance_bin", "gate_saving_ms"]
    ]
    .copy()
)

print(top10.round(4).to_string(index=False))

result = pd.DataFrame(rows)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

result.to_csv(OUTPUT, index=False)

report_lines = [
    "# Adaptive ALT: аудит устойчивости",
    "",
    "Разности определены как distance-gate минус ALT-2.",
    "Отрицательные значения означают преимущество gate.",
    "",
    "## Повторные измерения",
    "",
    "```",
    repeat_summary.round(4).to_string(),
    "```",
    "",
    "## Концентрация выигрыша",
    "",
    f"- Положительная экономия: {gross_positive:.4f} мс",
    f"- Чистая экономия: {net_saving:.4f} мс",
    f"- Маршрутов с ускорением: {len(positive)}",
    "",
    "## Чувствительность к исключению наиболее выгодных OD",
    "",
    "```",
    result[result["analysis"] == "leave_top_out"]
    .round(4).to_string(index=False),
    "```",
    "",
    "## Эффект по группам расстояний",
    "",
    "```",
    result[result["analysis"] == "distance_bin_bootstrap"]
    .round(4).to_string(index=False),
    "```",
    "",
    "## Ограничения",
    "",
    "- Исключение top-N выполнено после просмотра результатов;",
    "  это диагностика влияния наблюдений, не новая оценка качества.",
    "- Bootstrap по destination отражает вариацию состава",
    "  синтетической выборки, но не перенос на другие города.",
    "- Три повтора не считаются независимыми OD.",
]

REPORT.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8"
)

print("\n=== SAVED ===")
print(OUTPUT)
print(REPORT)
