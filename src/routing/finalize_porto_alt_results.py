from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import networkx as nx
import sklearn

BASE = Path("data/processed")
BENCH = BASE / "porto_heldout_alt_benchmark.parquet"
TEST = BASE / "porto_heldout_600_od.parquet"

SUMMARY_OUT = BASE / "porto_alt_final_summary.csv"
META_OUT = BASE / "porto_alt_reproducibility.json"
REPORT_OUT = Path("reports/porto_adaptive_alt_results.md")

STRATEGIES = ["alt2", "alt4", "distance_gate", "ml_tree"]
REPEATS = 3
N_OD = 600
SEED = 20261016

bench = pd.read_parquet(BENCH)
test = pd.read_parquet(TEST).sort_values("pair_id").reset_index(drop=True)

# 1. Frozen held-out set
assert len(test) == N_OD
assert test["pair_id"].is_unique
assert test[["origin", "destination"]].duplicated().sum() == 0

content = "\n".join(
    f"{r.origin},{r.destination}" for r in test.itertuples()
)
digest = hashlib.sha256(content.encode("utf-8")).hexdigest()

EXPECTED_SHA = (
    "d93bc12a37bb62b8a22a4fae55a6c5bd3bd863"
    "aa1a705f4b413cd47204d14b1e"
)
assert digest == EXPECTED_SHA

# 2. Benchmark completeness
expected_rows = N_OD * REPEATS * len(STRATEGIES)
assert len(bench) == expected_rows
assert set(bench["strategy"]) == set(STRATEGIES)
assert set(bench["repeat"]) == set(range(REPEATS))
assert bench["pair_id"].nunique() == N_OD

keys = ["repeat", "pair_id", "strategy"]
assert not bench.duplicated(keys).any()

coverage = bench.groupby(["pair_id", "strategy"]).size()
assert (coverage == REPEATS).all()

assert bench["total_ms"].gt(0).all()
assert bench["expanded"].ge(1).all()
assert bench["k"].isin([2, 4]).all()

# Decisions and expansion counts must not vary by repeat.
assert (
    bench.groupby(["pair_id", "strategy"])["k"]
    .nunique().eq(1).all()
)
assert (
    bench.groupby(["pair_id", "strategy"])["expanded"]
    .nunique().eq(1).all()
)

# 3. Summary based on repeat totals
repeat_totals = (
    bench.groupby(["repeat", "strategy"])["total_ms"]
    .sum()
    .unstack()
    .reindex(columns=STRATEGIES)
)

medians = repeat_totals.median()
baseline = float(medians["alt2"])

first = bench[bench["repeat"] == 0]

expanded = (
    first.groupby("strategy")["expanded"]
    .sum()
    .reindex(STRATEGIES)
)

k4 = (
    first[first["k"] == 4]
    .groupby("strategy")
    .size()
    .reindex(STRATEGIES, fill_value=0)
)

summary = pd.DataFrame({
    "strategy": STRATEGIES,
    "median_total_600_od_ms": [
        float(medians[s]) for s in STRATEGIES
    ],
    "change_vs_alt2_pct": [
        100 * (float(medians[s]) / baseline - 1)
        for s in STRATEGIES
    ],
    "expanded_nodes": [
        int(expanded[s]) for s in STRATEGIES
    ],
    "expansion_reduction_vs_alt2_pct": [
        100 * (1 - float(expanded[s]) / float(expanded["alt2"]))
        for s in STRATEGIES
    ],
    "k4_routes": [
        int(k4[s]) for s in STRATEGIES
    ],
})

# 4. Per-OD paired differences averaged over repeats
wide = bench.pivot(
    index=["repeat", "pair_id", "destination"],
    columns="strategy",
    values="total_ms",
).reset_index()

per_od = (
    wide.groupby(["pair_id", "destination"], as_index=False)
    [STRATEGIES].mean()
)

comparisons = {
    "distance_gate_minus_alt2": ("distance_gate", "alt2"),
    "ml_tree_minus_alt2": ("ml_tree", "alt2"),
    "ml_tree_minus_distance_gate": ("ml_tree", "distance_gate"),
    "alt4_minus_alt2": ("alt4", "alt2"),
}

# Bootstrap entire destination clusters.
groups = [
    group.index.to_numpy()
    for _, group in per_od.groupby("destination", sort=True)
]

rng = np.random.default_rng(42)
B = 5000

# Shared bootstrap samples for every comparison.
bootstrap_indices = [
    np.concatenate([
        groups[j]
        for j in rng.integers(0, len(groups), size=len(groups))
    ])
    for _ in range(B)
]

comparison_rows = []

for name, (a, b) in comparisons.items():
    delta = (
        per_od[a].to_numpy() - per_od[b].to_numpy()
    )
    samples = np.array([
        delta[idx].mean()
        for idx in bootstrap_indices
    ])
    low, high = np.quantile(samples, [0.025, 0.975])

    comparison_rows.append({
        "comparison": name,
        "mean_delta_us_per_od": float(delta.mean() * 1000),
        "ci95_low_us": float(low * 1000),
        "ci95_high_us": float(high * 1000),
    })

comparison_df = pd.DataFrame(comparison_rows)

# 5. Distance-stratified totals
bin_totals = (
    bench.groupby(
        ["repeat", "distance_bin", "strategy"]
    )["total_ms"]
    .sum()
    .reset_index()
    .groupby(["distance_bin", "strategy"])["total_ms"]
    .median()
    .unstack()
    .reindex(columns=STRATEGIES)
)

# 6. Timing component means
components = (
    bench.groupby("strategy")[
        ["selection_ms", "preparation_ms", "search_ms", "total_ms"]
    ]
    .mean()
    .reindex(STRATEGIES)
    * 1000
)

# selection_ms already includes decision_ms.
# Therefore decision_ms must NOT be added separately.

# 7. Metadata
metadata = {
    "dataset": "Porto OSMnx directed drive graph",
    "heldout_od_count": N_OD,
    "heldout_sha256": digest,
    "distance_bins": {
        str(k): int(v)
        for k, v in test["distance_bin"].value_counts().items()
    },
    "benchmark_repeats": REPEATS,
    "strategies": STRATEGIES,
    "sampling_seed": SEED,
    "bootstrap_seed": 42,
    "bootstrap_iterations": B,
    "bootstrap_cluster": "destination",
    "training_od_count": 142,
    "distance_gate_threshold_m": 2914.709,
    "ml_tree_depth": 2,
    "ml_tree_threshold_saved_expansions": 25,
    "ml_tree_training_data": "original 142 GPS OD only",
    "test_set_used_for_tuning": False,
    "notes": [
        "Synthetic OD balanced across three geographic-distance bins",
        "All 600 OD lie in the largest strongly connected component",
        "Three repeats measure timing variability, not new OD samples",
        "ALT preprocessing excluded from per-query timing",
        "Reported inference costs depend on Python/NumPy implementation",
        "All 2400 strategy-route combinations verified against Dijkstra",
    ],
    "environment": {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "networkx": nx.__version__,
        "scikit_learn": sklearn.__version__,
    },
}

# 8. Write outputs
SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)

summary.to_csv(SUMMARY_OUT, index=False)

META_OUT.write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

def markdown_table(frame, digits=3):
    f = frame.copy()
    for c in f.columns:
        if pd.api.types.is_float_dtype(f[c]):
            f[c] = f[c].map(lambda x: f"{x:.{digits}f}")

    headers = list(f.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in f.itertuples(index=False, name=None):
        lines.append(
            "| " + " | ".join(str(x) for x in row) + " |"
        )
    return "\n".join(lines)

report = f"""# Adaptive ALT routing in Porto: результаты эксперимента

## Постановка задачи

Исследуется выбор числа ALT-landmarks для A* на ориентированном
дорожном графе Porto с весами `osm_travel_time_s`.

Сравниваются четыре стратегии: ALT-2, ALT-4,
distance-gate и ML-tree.

## Экспериментальный протокол

- Исходная обучающая выборка: 142 GPS OD.
- Независимая тестовая выборка: 600 новых синтетических OD.
- Short / medium / long: по 200 OD.
- Тестовые пары не совпадают с исходными GPS OD.
- Все тестовые пары достижимы.
- Три рандомизированных повтора.
- ML-tree: глубина 2, порог 25.
- Distance-gate: порог 2914,709 м.
- Параметры зафиксированы до оценки на новых OD.
- Все маршруты проверены на оптимальность против Dijkstra.
- ALT preprocessing: 679,277 мс в одном измерении, исключено
  из времени отдельных запросов.
- SHA-256 тестовой выборки: `{digest}`.

## End-to-end результаты

Медиана суммы времени обработки 600 OD по трём повторам.

{markdown_table(summary, 3)}

## Парные различия времени

Разности рассчитаны для каждого OD с усреднением трёх повторов.
95% доверительные интервалы получены bootstrap по destination
(5000 итераций). Единицы: мкс/OD.

{markdown_table(comparison_df, 3)}

## Результаты по расстояниям

Медиана суммарного времени на 200 OD в каждой группе, мс.

{markdown_table(bin_totals.reset_index(), 3)}

## Разложение времени

Среднее время на OD, мкс.
`selection_ms` уже включает `decision_ms`.

{markdown_table(components.reset_index(), 3)}

## Интерпретация

Distance-gate сокращает полное время маршрутизации
относительно ALT-2 на сбалансированной тестовой выборке.
ML-tree сокращает число раскрытий, но уступает distance-gate
по полному времени из-за дополнительных накладных расходов.

## Ограничения

1. OD синтетические и сбалансированы по расстоянию; результаты
   не являются прямой оценкой задержки на реальном потоке такси.
2. Проверка выполнена на том же дорожном графе Porto.
3. Выводы о задержке зависят от реализации на Python/NumPy
   и аппаратной среды.
4. Исходные 142 GPS OD использовались для разработки и выбора
   стратегий; независимая проверка выполнена на 600 новых OD.
5. Доверительные интервалы характеризуют вариацию состава OD
   при выбранной схеме bootstrap, а не перенос на другие города.
"""

REPORT_OUT.write_text(report, encoding="utf-8")

print("=== VALIDATION ===")
print("Held-out SHA256:", digest)
print("Benchmark rows:", len(bench))
print("Unique OD:", bench["pair_id"].nunique())
print("Complete repeat coverage: True")
print("Deterministic k across repeats: True")
print("Deterministic expansions across repeats: True")

print("\n=== FINAL SUMMARY ===")
print(summary.round(3).to_string(index=False))

print("\n=== PAIRED BOOTSTRAP ===")
print(comparison_df.round(3).to_string(index=False))

print("\n=== OUTPUT FILES ===")
print(SUMMARY_OUT)
print(META_OUT)
print(REPORT_OUT)
