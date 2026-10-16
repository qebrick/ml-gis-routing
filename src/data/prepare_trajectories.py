"""Parse Porto Taxi polylines and create point-to-point interval observations.

This is a quality-control intermediate, not yet a road-edge training table.
GPS points in this dataset are sampled every ~15 seconds; do not treat each
interval as the exact travel time of one road segment.
"""
import argparse
import ast
import json
import math
from pathlib import Path
import pandas as pd

GPS_INTERVAL_SECONDS = 15.0
EARTH_RADIUS_M = 6_371_000.0

def parse_polyline(value):
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        points = json.loads(value)
    except json.JSONDecodeError:
        try:
            points = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
    clean = []
    for p in points:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            try:
                lon, lat = float(p[0]), float(p[1])
            except (TypeError, ValueError):
                continue
            if -180 <= lon <= 180 and -90 <= lat <= 90:
                clean.append((lon, lat))
    return clean

def haversine_m(p1, p2):
    lon1, lat1 = map(math.radians, p1)
    lon2, lat2 = map(math.radians, p2)
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, a)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/train.csv")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--output", default="data/processed/gps_intervals.parquet")
    parser.add_argument("--max-speed-kmh", type=float, default=160)
    parser.add_argument("--min-points", type=int, default=5)
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; download the UCI CSV first.")
    df = pd.read_csv(path, nrows=args.limit)
    required = {"TRIP_ID", "TIMESTAMP", "POLYLINE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Expected columns missing: {sorted(missing)}")

    rows = []
    trips_seen = 0
    trips_kept = 0
    for row in df.itertuples(index=False):
        d = row._asdict()
        trips_seen += 1
        points = parse_polyline(d.get("POLYLINE"))
        if len(points) < args.min_points:
            continue
        start_ts = pd.to_numeric(pd.Series([d.get("TIMESTAMP")]), errors="coerce").iloc[0]
        if pd.isna(start_ts):
            continue
        trips_kept += 1
        trip_id = str(d.get("TRIP_ID"))
        for i, (p1, p2) in enumerate(zip(points[:-1], points[1:])):
            dist = haversine_m(p1, p2)
            dt = GPS_INTERVAL_SECONDS
            speed_kmh = (dist / dt) * 3.6
            if not math.isfinite(speed_kmh) or speed_kmh > args.max_speed_kmh:
                continue
            rows.append({
                "trip_id": trip_id,
                "point_index": i,
                "timestamp_utc": pd.to_datetime(float(start_ts) + i * dt, unit="s", utc=True),
                "lon_1": p1[0], "lat_1": p1[1],
                "lon_2": p2[0], "lat_2": p2[1],
                "distance_m": dist,
                "interval_seconds": dt,
                "observed_speed_kmh": speed_kmh,
            })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No valid intervals were created; inspect the CSV format and QC thresholds.")
    result.to_parquet(out, index=False)
    print(f"Trips inspected: {trips_seen:,}; trips retained: {trips_kept:,}")
    print(f"Valid GPS intervals: {len(result):,}")
    print(f"Saved: {out}")
    print("\nSpeed summary (km/h):")
    print(result["observed_speed_kmh"].describe(percentiles=[.5, .9, .95, .99]))
    print("\nImportant: this file is for trajectory QC and map-matching input, not final edge labels.")

if __name__ == "__main__":
    main()
