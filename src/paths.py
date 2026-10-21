"""Locations for the article workflow; published measurements are read-only inputs."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"


def benchmark_path():
    return Path(os.environ.get("PORTO_BENCHMARK_PATH", DATA / "porto_heldout_alt_benchmark.parquet")).resolve()


def output_path(name):
    directory = Path(os.environ.get("PORTO_OUTPUT_DIR", ROOT / "reports" / "recomputed")).resolve()
    if directory == ROOT / "data" or ROOT / "data" in directory.parents:
        raise ValueError("New results must be written outside published data/.")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name
