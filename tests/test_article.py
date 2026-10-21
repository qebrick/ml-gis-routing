"""Regression checks for integrity and separation of published/new measurements."""
import os
import unittest
from unittest.mock import patch

import pandas as pd

from src.paths import DATA, ROOT, output_path
from src.validation import validate_measurements, verify_published


class ArticleValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = pd.read_parquet(DATA / "porto_heldout_alt_benchmark.parquet")
        cls.od = pd.read_parquet(DATA / "porto_heldout_600_od.parquet")

    def test_published_results(self):
        verify_published()

    def test_foreign_pair_id_rejected_even_when_row_count_matches(self):
        corrupt = self.bench.copy()
        corrupt.loc[corrupt.pair_id == corrupt.pair_id.iloc[0], "pair_id"] = -1
        with self.assertRaisesRegex(ValueError, "foreign"):
            validate_measurements(corrupt, self.od)

    def test_mislabeled_destination_rejected(self):
        corrupt = self.bench.copy()
        corrupt["destination"] = corrupt.destination.astype(str)
        corrupt.loc[0, "destination"] = "missing-node"
        with self.assertRaisesRegex(ValueError, "destination"):
            validate_measurements(corrupt, self.od)

    def test_duplicate_rejected(self):
        corrupt = self.bench.copy()
        corrupt.iloc[0] = corrupt.iloc[1]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_measurements(corrupt, self.od)

    def test_nonfinite_time_rejected(self):
        corrupt = self.bench.copy()
        corrupt.loc[0, "total_ms"] = float("inf")
        with self.assertRaisesRegex(ValueError, "timing"):
            validate_measurements(corrupt, self.od)

    def test_output_cannot_overwrite_published_data(self):
        for directory in (ROOT / "data", DATA, DATA / "nested"):
            with self.subTest(directory=directory), patch.dict(os.environ, {"PORTO_OUTPUT_DIR": str(directory)}):
                with self.assertRaisesRegex(ValueError, "outside"):
                    output_path("porto_heldout_alt_benchmark.parquet")


if __name__ == "__main__":
    unittest.main()
