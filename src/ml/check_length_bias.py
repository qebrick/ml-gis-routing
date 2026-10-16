from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error

DATA_PATH = Path("data/processed/ml_training_data.parquet")

ALL_FEATURES = [
    "edge_length_m",
    "highway",
    "maxspeed_kmh",
    "lanes",
    "oneway",
    "hour",
    "day_of_week",
    "has_maxspeed",
    "has_lanes",
]

CATEGORICAL = ["highway", "oneway"]

data = pd.read_parquet(DATA_PATH)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(data, groups=data["trip_id"])
)

train = data.iloc[train_idx].copy()
test = data.iloc[test_idx].copy()


def evaluate_model(name, features):
    X_train = train[features].copy()
    X_test = test[features].copy()

    categorical = [
        c for c in CATEGORICAL if c in features
    ]

    for col in categorical:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    model = CatBoostRegressor(
        iterations=500,
        depth=4,
        learning_rate=0.03,
        loss_function="MAE",
        random_seed=42,
        verbose=False,
        thread_count=4,
        l2_leaf_reg=5,
    )

    model.fit(
        X_train,
        train["target_speed_kmh"],
        cat_features=categorical,
    )

    pred = np.maximum(model.predict(X_test), 3.0)

    speed_mae = mean_absolute_error(
        test["target_speed_kmh"],
        pred,
    )

    predicted_time = (
        3.6
        * test["traversed_length_m"].to_numpy()
        / pred
    )

    time_mae = mean_absolute_error(
        test["observed_time_seconds"],
        predicted_time,
    )

    print(
        f"{name:20s} | "
        f"Speed MAE: {speed_mae:7.3f} km/h | "
        f"Time MAE: {time_mae:7.3f} s"
    )


print("=== LENGTH BIAS CHECK ===")
print("Rows:", len(data))
print("Train trips:", train["trip_id"].nunique())
print("Test trips:", test["trip_id"].nunique())

print("\n=== MODEL COMPARISON ===")

evaluate_model(
    "With edge length",
    ALL_FEATURES,
)

evaluate_model(
    "Without edge length",
    [f for f in ALL_FEATURES if f != "edge_length_m"],
)

print("\n=== LENGTH CORRELATIONS ===")

print(
    "Corr(edge length, target speed):",
    round(
        data["edge_length_m"].corr(
            data["target_speed_kmh"]
        ),
        4,
    ),
)

print(
    "Corr(traversed length, target speed):",
    round(
        data["traversed_length_m"].corr(
            data["target_speed_kmh"]
        ),
        4,
    ),
)

print(
    "Corr(edge length, traversed length):",
    round(
        data["edge_length_m"].corr(
            data["traversed_length_m"]
        ),
        4,
    ),
)
