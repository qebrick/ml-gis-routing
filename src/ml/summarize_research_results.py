from pathlib import Path

import pandas as pd

BASE = Path("data/processed")

nested = pd.read_parquet(
    BASE / "nested_eta_validation.parquet"
)

routes = pd.read_parquet(
    BASE / "gps_od_route_comparison.parquet"
)

overlap = pd.read_parquet(
    BASE / "gps_traversed_overlap_comparison.parquet"
)

bootstrap = pd.read_csv(
    BASE / "gps_overlap_bootstrap_summary.csv"
)

models = ["osm", "class", "conservative"]

# Nested CV: each route belongs to one outer test fold.
eta_mae = {}

for model in models:
    eta_mae[model] = (
        nested[f"mae_{model}_s"]
        .mul(nested["test_routes"])
        .sum()
        / nested["test_routes"].sum()
    )

records = []

for model in models:
    row = {
        "model": model,
        "eta_mae_s": eta_mae[model],
        "eta_improvement_pct": (
            100
            * (
                eta_mae["osm"] - eta_mae[model]
            )
            / eta_mae["osm"]
        ),
        "routing_od_pairs": len(routes),
        "overlap_od_pairs": len(overlap),
        "gps_recall_mean": (
            overlap[f"recall_{model}"].mean()
        ),
    }

    if model == "osm":
        row.update({
            "changed_routes": 0,
            "changed_share_pct": 0.0,
            "residential_change_pp": 0.0,
            "gps_recall_delta_pp": 0.0,
            "gps_recall_ci_low_pp": 0.0,
            "gps_recall_ci_high_pp": 0.0,
        })

    else:
        changed = routes[f"{model}_changed"]

        ci = bootstrap.loc[
            (bootstrap["model"] == model)
            & (bootstrap["scope"] == "all")
        ]

        if len(ci) != 1:
            raise ValueError(
                f"Unexpected bootstrap rows: {model}"
            )

        ci = ci.iloc[0]

        row.update({
            "changed_routes": int(changed.sum()),
            "changed_share_pct": (
                100 * changed.mean()
            ),
            "residential_change_pp": (
                routes[
                    f"{model}_residential_change_pp"
                ].mean()
            ),
            "gps_recall_delta_pp": (
                100
                * overlap[f"delta_{model}"].mean()
            ),
            "gps_recall_ci_low_pp": (
                100 * ci["ci_low"]
            ),
            "gps_recall_ci_high_pp": (
                100 * ci["ci_high"]
            ),
        })

    records.append(row)

summary = pd.DataFrame(records)

output = BASE / "research_results_summary.csv"
summary.to_csv(output, index=False)

print("=== ETA PERFORMANCE ===")

print(
    summary[
        [
            "model",
            "eta_mae_s",
            "eta_improvement_pct",
        ]
    ].round(3).to_string(index=False)
)

print("\n=== ROUTING PERFORMANCE ===")

print(
    summary[
        [
            "model",
            "routing_od_pairs",
            "changed_routes",
            "changed_share_pct",
            "residential_change_pp",
        ]
    ].round(3).to_string(index=False)
)

print("\n=== GPS OVERLAP ===")

print(
    summary[
        [
            "model",
            "gps_recall_mean",
            "gps_recall_delta_pp",
            "gps_recall_ci_low_pp",
            "gps_recall_ci_high_pp",
        ]
    ].round(3).to_string(index=False)
)

print("\n=== SAMPLE SIZES ===")

print(
    "Nested ETA routes:",
    int(nested["test_routes"].sum())
)

print(
    "Nested ETA test trips (sum across folds):",
    int(nested["test_trips"].sum())
)

print(
    "Routing OD pairs:",
    len(routes)
)

print(
    "GPS overlap OD pairs:",
    len(overlap)
)

print(
    "GPS overlap unique trips:",
    overlap["trip_id"].nunique()
)

print("\nSaved:", output)
