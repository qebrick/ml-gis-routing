"""Inspect the UCI Porto Taxi CSV without loading more rows than requested."""
import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/train.csv")
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()
    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download Porto Taxi from UCI and place the CSV there."
        )
    df = pd.read_csv(path, nrows=args.limit)
    print(f"File: {path}")
    print(f"Rows inspected: {len(df):,}")
    print("Columns:", list(df.columns))
    print("\nDtypes:\n", df.dtypes)
    print("\nMissing values:\n", df.isna().sum())
    if "POLYLINE" in df.columns:
        print("\nExample POLYLINE:", str(df["POLYLINE"].iloc[0])[:180])
    if "TIMESTAMP" in df.columns:
        ts = pd.to_numeric(df["TIMESTAMP"], errors="coerce")
        print("\nTimestamp range (UTC):", pd.to_datetime(ts, unit="s", errors="coerce", utc=True).min(),
              "to", pd.to_datetime(ts, unit="s", errors="coerce", utc=True).max())
    if "MISSING_DATA" in df.columns:
        print("\nMISSING_DATA counts:\n", df["MISSING_DATA"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
