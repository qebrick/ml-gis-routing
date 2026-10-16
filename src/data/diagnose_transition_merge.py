import pandas as pd

trans = pd.read_parquet(
    "data/processed/edge_transition_diagnostics.parquet"
)
gps = pd.read_parquet(
    "data/processed/gps_intervals.parquet"
)

keys = ["trip_id", "point_index"]

print("Transitions:", len(trans))
print("GPS intervals:", len(gps))

print("\nTransition key types:")
print(trans[["trip_id", "point_index_1"]].dtypes)

print("\nGPS key types:")
print(gps[keys].dtypes)

print("\nDuplicate GPS interval keys:")
print(gps.duplicated(keys).sum())

merged = trans.merge(
    gps[keys + ["distance_m", "interval_seconds"]],
    left_on=["trip_id", "point_index_1"],
    right_on=keys,
    how="left",
    indicator=True,
)

print("\nMerge results:")
print(merged["_merge"].value_counts())

missing = merged[merged["_merge"] == "left_only"]

print("\nMissing matches:", len(missing))
print("\nFirst 10 missing transitions:")
print(
    missing[
        ["trip_id", "point_index_1", "point_index_2"]
    ].head(10).to_string(index=False)
)

print("\nTransitions with non-consecutive point indices:")
print(
    (trans["point_index_2"] != trans["point_index_1"] + 1).sum()
)
