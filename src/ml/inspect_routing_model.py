from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor

MODEL_PATH = Path("models/catboost_speed_pilot.cbm")
DATA_PATH = Path("data/processed/ml_training_data.parquet")

model = CatBoostRegressor()
model.load_model(str(MODEL_PATH))

df = pd.read_parquet(DATA_PATH)

print("=== MODEL FEATURES ===")
print("Feature names:", model.feature_names_)
print("Categorical feature indices:", model.get_cat_feature_indices())

print("\n=== TRAINING DATA TYPES ===")
for feature in model.feature_names_:
    if feature not in df.columns:
        print(f"{feature}: MISSING")
        continue

    column = df[feature]
    print(
        f"{feature}: "
        f"dtype={column.dtype}, "
        f"nulls={column.isna().sum()}"
    )

print("\n=== CATEGORICAL EXAMPLES ===")
for index in model.get_cat_feature_indices():
    feature = model.feature_names_[index]
    if feature in df.columns:
        print(
            feature,
            df[feature].dropna().astype(str).unique()[:15].tolist()
        )

print("\n=== NUMERIC RANGES ===")
numeric_features = [
    feature
    for feature in model.feature_names_
    if feature in df.columns
    and pd.api.types.is_numeric_dtype(df[feature])
]

if numeric_features:
    print(
        df[numeric_features]
        .describe()
        .round(2)
        .to_string()
    )
