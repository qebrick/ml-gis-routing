"""Entry point for checking, recalculating and rerunning the article experiment."""
import argparse
import os
import runpy
from pathlib import Path

from src.paths import DATA, ROOT

REPORT_MODULES = (
    "finalize_porto_alt_results",
    "analyze_porto_heldout_alt",
    "audit_porto_alt_robustness",
    "analyze_porto_alt_workload",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "reports", "benchmark"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/recomputed")
    parser.add_argument("--benchmark", type=Path, help="Measurements to analyze (reports only)")
    args = parser.parse_args(argv)
    if args.benchmark and args.command != "reports":
        parser.error("--benchmark is supported only for reports")
    from src.validation import verify_published
    if args.command == "verify":
        verify_published()
        return
    output = args.output_dir.resolve()
    if output == ROOT / "data" or ROOT / "data" in output.parents:
        parser.error("--output-dir must be outside published data/")
    source = (args.benchmark or DATA / "porto_heldout_alt_benchmark.parquet").resolve()
    if args.command == "reports":
        import pandas as pd
        from src.validation import validate_measurements
        validate_measurements(pd.read_parquet(source), pd.read_parquet(DATA / "porto_heldout_600_od.parquet"))
    if source.parent == output:
        parser.error("Keep input measurements separate from the report output directory")
    if args.command == "benchmark" and (output / "porto_heldout_alt_benchmark.parquet").exists():
        parser.error("Output benchmark already exists; choose a new --output-dir")
    old_env = {key: os.environ.get(key) for key in ("PORTO_OUTPUT_DIR", "PORTO_BENCHMARK_PATH")}
    try:
        os.environ["PORTO_OUTPUT_DIR"] = str(output)
        os.environ["PORTO_BENCHMARK_PATH"] = str(source)
        modules = REPORT_MODULES if args.command == "reports" else ("benchmark_porto_heldout_alt",)
        for name in modules:
            runpy.run_module(f"src.routing.{name}", run_name="__main__")
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    main()
