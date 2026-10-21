"""Read-only integrity checks for the published article inputs and measurements."""
import hashlib
import json

import numpy as np
import pandas as pd

from src.paths import DATA

STRATEGIES = ("alt2", "alt4", "distance_gate", "ml_tree")
HELDOUT_SHA = "d93bc12a37bb62b8a22a4fae55a6c5bd3bd863aa1a705f4b413cd47204d14b1e"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_measurements(bench, od):
    """Validate the full Cartesian design and correspondence to the frozen OD set."""
    keys = ["repeat", "pair_id", "strategy"]
    expected = pd.MultiIndex.from_product([range(3), od.pair_id, STRATEGIES], names=keys)
    require(not bench.duplicated(keys).any(), "Duplicate measurement keys")
    actual = pd.MultiIndex.from_frame(bench[keys])
    require(len(actual) == len(expected) and actual.difference(expected).empty
            and expected.difference(actual).empty, "Incomplete or foreign measurement keys")
    merged = bench.merge(od[["pair_id", "destination", "distance_bin"]], on="pair_id",
                         suffixes=("", "_od"), validate="many_to_one")
    for col in ("destination", "distance_bin"):
        require(merged[col].astype(str).eq(merged[col + "_od"].astype(str)).all(),
                f"Benchmark {col} does not match OD")
    timing = ["decision_ms", "selection_ms", "preparation_ms", "search_ms", "total_ms"]
    values = bench[timing].to_numpy(dtype=float)
    require(np.isfinite(values).all() and (values >= 0).all() and bench.total_ms.gt(0).all(),
            "Invalid timing values")
    require(bench.selection_ms.ge(bench.decision_ms - 1e-9).all(), "Decision time exceeds selection")
    components = bench[["selection_ms", "preparation_ms", "search_ms"]].sum(axis=1)
    require(bench.total_ms.ge(components - 1e-6).all(), "Total time excludes measured components")
    require(bench.k.isin([2, 4]).all(), "Invalid landmark budget")
    require(bench.loc[bench.strategy == "alt2", "k"].eq(2).all(), "ALT-2 budget mismatch")
    require(bench.loc[bench.strategy == "alt4", "k"].eq(4).all(), "ALT-4 budget mismatch")
    require(bench.expanded.ge(1).all() and np.isfinite(bench.expanded).all(), "Invalid expansion count")
    for col in ("k", "expanded"):
        require(bench.groupby(["pair_id", "strategy"])[col].nunique().eq(1).all(),
                f"Non-deterministic {col}")


def verify_published():
    od = pd.read_parquet(DATA / "porto_heldout_600_od.parquet").sort_values("pair_id")
    bench = pd.read_parquet(DATA / "porto_heldout_alt_benchmark.parquet")
    train = pd.read_parquet(DATA / "ml_alt_budget_groupkfold.parquet")
    gps = pd.read_parquet(DATA / "gps_od_route_comparison.parquet")
    require(len(od) == 600 and od.pair_id.is_unique, "Expected 600 unique pair IDs")
    require(not od.duplicated(["origin", "destination"]).any(), "Duplicate OD pairs")
    content = "\n".join(f"{r.origin},{r.destination}" for r in od.itertuples())
    require(hashlib.sha256(content.encode()).hexdigest() == HELDOUT_SHA, "Held-out checksum mismatch")
    require(od.distance_bin.value_counts().to_dict() == {"short": 200, "medium": 200, "long": 200},
            "Unbalanced held-out distance bins")
    pairs = lambda frame: set(map(tuple, frame[["origin", "destination"]].astype(str).to_numpy()))
    require(not pairs(od) & pairs(gps), "Training/test OD overlap")
    require(len(train) == len(gps) == 142 and train.destination.nunique() == 124, "Training sample mismatch")
    validate_measurements(bench, od)
    medians = bench.groupby(["repeat", "strategy"]).total_ms.sum().unstack().median()
    summary = pd.read_csv(DATA / "porto_alt_final_summary.csv").set_index("strategy")
    first = bench[bench.repeat == 0]
    for strategy in STRATEGIES:
        row = summary.loc[strategy]
        require(np.isclose(medians[strategy], row.median_total_600_od_ms, rtol=0, atol=1e-8),
                f"Published time mismatch: {strategy}")
        require(first.loc[first.strategy == strategy, "expanded"].sum() == row.expanded_nodes,
                f"Published expansion mismatch: {strategy}")
        require(len(first[(first.strategy == strategy) & (first.k == 4)]) == row.k4_routes,
                f"Published budget count mismatch: {strategy}")
    metadata = json.loads((DATA / "porto_alt_reproducibility.json").read_text(encoding="utf-8-sig"))
    require(metadata["heldout_sha256"] == HELDOUT_SHA, "Metadata checksum mismatch")
    print("PASS: 600 disjoint OD, 142 training OD, 7200 measurements; checksum and published summary match.")
    print("Saved measurements checked; routing runtime was not remeasured.")


if __name__ == "__main__":
    verify_published()
