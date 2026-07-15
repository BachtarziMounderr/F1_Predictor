"""Create the single end-to-end portfolio notebook for the F1 Predictor project."""
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "F1_Predictor_End_to_End_Portfolio_Notebook.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(
        """
# Formula 1 Race Outcome Prediction

## End-to-End Data Science and Machine Learning Notebook

This notebook is the main portfolio notebook for the **F1 Predictor** project.

The objective is to predict, for each driver and each Grand Prix:

- the probability of finishing in the **top 10**;
- the probability of finishing on the **podium**;
- the driver's **expected points**.

The project uses historical Kaggle Formula 1 data as the core dataset, extends the latest seasons with Jolpica-F1 snapshots, builds a driver-race table, creates leakage-safe rolling features, trains temporal machine learning models, evaluates them on a future season, and generates 2026 live/application predictions.
"""
    ),
    md(
        """
# Import Libraries

The notebook uses the same stack as the project code: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost` when available, and the reusable modules under `src/`.
"""
    ),
    code(
        """
from pathlib import Path
import sys
import json
import warnings
import subprocess

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from IPython.display import display, Image, Markdown
except ImportError:
    def display(obj):
        print(obj)

    def Markdown(text):
        return text

    class Image:
        def __init__(self, filename):
            self.filename = filename

        def __repr__(self):
            return f"Image(filename={self.filename!r})"

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.split import temporal_split
from src.modeling import select_feature_columns

sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.05)
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 11

print(f"Project root: {PROJECT_ROOT}")
"""
    ),
    md(
        """
# Project Initialization

This section loads the project configuration and checks the expected directories. The goal is to make the notebook reproducible while keeping all paths centralized in `config.yaml`.
"""
    ),
    code(
        """
cfg = load_config(PROJECT_ROOT / "config.yaml")

paths = {
    "processed": cfg["resolved_paths"]["processed_data_dir"],
    "external": cfg["resolved_paths"]["external_data_dir"],
    "predictions": cfg["resolved_paths"]["predictions_dir"],
    "models": cfg["resolved_paths"]["models_dir"],
    "reports": cfg["resolved_paths"]["reports_dir"],
}

display(pd.DataFrame({
    "path_name": list(paths.keys()),
    "path": [str(p) for p in paths.values()],
    "exists": [p.exists() for p in paths.values()],
}))
"""
    ),
    code(
        """
print(json.dumps(cfg["data"], indent=2))
print(json.dumps(cfg["models"], indent=2))
"""
    ),
    md(
        """
# Optional Pipeline Execution

The cells below are intentionally disabled by default. The notebook is designed to inspect and explain the saved artifacts, but it can also rebuild the full pipeline when needed.
"""
    ),
    code(
        """
RUN_FULL_PIPELINE = False

if RUN_FULL_PIPELINE:
    commands = [
        [sys.executable, str(PROJECT_ROOT / "scripts" / "fetch_jolpica_data.py"), "--seasons", "2025", "2026"],
        [sys.executable, str(PROJECT_ROOT / "scripts" / "build_dataset.py")],
        [sys.executable, str(PROJECT_ROOT / "scripts" / "train_models.py")],
        [sys.executable, str(PROJECT_ROOT / "scripts" / "predict_2026.py")],
    ]
    for command in commands:
        print("Running:", " ".join(command))
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
else:
    print("Pipeline execution skipped. Set RUN_FULL_PIPELINE = True to rebuild all artifacts.")
"""
    ),
    md(
        """
# Data Loading

The final modeling table should contain one row per driver-race. It combines Kaggle historical data and Jolpica snapshots for 2025 and 2026.
"""
    ),
    code(
        """
processed_dir = paths["processed"]

dataset_path = processed_dir / "f1_driver_race_dataset_with_jolpica.csv"
base_path = processed_dir / "base_driver_race_table_with_jolpica.csv"

if not dataset_path.exists():
    raise FileNotFoundError(f"Missing dataset: {dataset_path}")

df = pd.read_csv(dataset_path)
base_df = pd.read_csv(base_path) if base_path.exists() else pd.DataFrame()

print(f"Final dataset shape: {df.shape}")
print(f"Base table shape: {base_df.shape}")
display(df.head())
"""
    ),
    code(
        """
summary = pd.DataFrame({
    "column": df.columns,
    "dtype": df.dtypes.astype(str).values,
    "missing_values": df.isna().sum().values,
    "missing_rate": df.isna().mean().round(4).values,
    "unique_values": df.nunique(dropna=True).values,
})

display(summary.head(30))
"""
    ),
    md(
        """
# Data Availability Audit

Before modeling, we verify which seasons are present, how many rows each season contains, and whether the driver-race key is unique.
"""
    ),
    code(
        """
season_counts = (
    df.groupby("season")
    .size()
    .reset_index(name="driver_race_rows")
    .sort_values("season")
)

display(season_counts)

ax = sns.barplot(data=season_counts, x="season", y="driver_race_rows", color="#4C72B0")
ax.set_title("Driver-race rows by season")
ax.set_xlabel("Season")
ax.set_ylabel("Rows")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
key_columns = ["season", "round", "driverId"] if "driverId" in df.columns else ["season", "round", "driverRef"]
duplicate_count = df.duplicated(key_columns).sum()

print(f"Key columns: {key_columns}")
print(f"Duplicate driver-race rows: {duplicate_count}")

target_columns = ["target_top10", "target_podium", "target_points"]
display(df.groupby("season")[target_columns].apply(lambda x: x.isna().sum()))
"""
    ),
    md(
        """
# Jolpica Coverage Check

Jolpica is used for the official 2025 test season and the 2026 live/application layer. This section compares the available Jolpica snapshots with the final dataset.
"""
    ),
    code(
        """
jolpica_dir = paths["external"] / "jolpica"

coverage_rows = []
for season in [2025, 2026]:
    for table_name in ["schedule", "results", "qualifying", "driver_standings", "constructor_standings"]:
        path = jolpica_dir / f"jolpica_{season}_{table_name}.csv"
        if path.exists():
            tmp = pd.read_csv(path)
            rounds = pd.to_numeric(tmp.get("round"), errors="coerce").dropna().astype(int).nunique() if "round" in tmp else np.nan
            rows = len(tmp)
        else:
            rounds = np.nan
            rows = 0
        coverage_rows.append({"season": season, "snapshot": table_name, "rows": rows, "rounds": rounds})

jolpica_coverage = pd.DataFrame(coverage_rows)
display(jolpica_coverage)
"""
    ),
    code(
        """
fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=jolpica_coverage, x="snapshot", y="rows", hue="season", ax=ax)
ax.set_title("Jolpica snapshot row counts")
ax.set_xlabel("Snapshot")
ax.set_ylabel("Rows")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
schedule_2026_path = jolpica_dir / "jolpica_2026_schedule.csv"
if schedule_2026_path.exists():
    schedule_2026 = pd.read_csv(schedule_2026_path)
    dataset_2026_rounds = set(df.loc[df["season"].eq(2026), "round"].dropna().astype(int))
    schedule_2026["in_final_dataset"] = schedule_2026["round"].astype(int).isin(dataset_2026_rounds)
    display(schedule_2026[["round", "raceName", "date", "in_final_dataset"]])

    ax = sns.countplot(data=schedule_2026, x="in_final_dataset")
    ax.set_title("2026 schedule coverage in the final driver-race dataset")
    ax.set_xlabel("Round represented in final dataset")
    ax.set_ylabel("Number of races")
    plt.tight_layout()
    plt.show()
"""
    ),
    md(
        """
# Exploratory Data Analysis

The following charts explain the raw modeling problem: how results, points, grids, constructors, drivers, and circuits are distributed across the dataset.
"""
    ),
    code(
        """
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(df["positionOrder"].dropna(), bins=22, kde=False, ax=axes[0], color="#4C72B0")
axes[0].set_title("Finish position distribution")
axes[0].set_xlabel("Finish position")

sns.histplot(df["points"].dropna(), bins=25, kde=False, ax=axes[1], color="#55A868")
axes[1].set_title("Points distribution")
axes[1].set_xlabel("Points")

sns.histplot(df["grid_position"].dropna(), bins=22, kde=False, ax=axes[2], color="#C44E52")
axes[2].set_title("Grid position distribution")
axes[2].set_xlabel("Grid position")

plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
target_rates = (
    df.groupby("season")[["target_top10", "target_podium"]]
    .mean()
    .reset_index()
    .melt(id_vars="season", var_name="target", value_name="positive_rate")
)

ax = sns.lineplot(data=target_rates, x="season", y="positive_rate", hue="target", marker="o")
ax.set_title("Target positive rate by season")
ax.set_xlabel("Season")
ax.set_ylabel("Positive rate")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
constructor_points = (
    df[df["season"].between(2014, 2025)]
    .groupby("constructor_name")["points"]
    .sum()
    .sort_values(ascending=False)
    .head(15)
    .reset_index()
)

ax = sns.barplot(data=constructor_points, y="constructor_name", x="points", palette="viridis")
ax.set_title("Top constructors by total points, 2014-2025")
ax.set_xlabel("Total points")
ax.set_ylabel("")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
driver_points = (
    df[df["season"].between(2014, 2025)]
    .groupby("driver_name")["points"]
    .sum()
    .sort_values(ascending=False)
    .head(20)
    .reset_index()
)

ax = sns.barplot(data=driver_points, y="driver_name", x="points", palette="mako")
ax.set_title("Top drivers by total points, 2014-2025")
ax.set_xlabel("Total points")
ax.set_ylabel("")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
grid_finish = df.dropna(subset=["grid_position", "positionOrder"]).copy()
grid_finish = grid_finish[grid_finish["season"].between(2014, 2025)]

ax = sns.scatterplot(
    data=grid_finish.sample(min(2500, len(grid_finish)), random_state=42),
    x="grid_position",
    y="positionOrder",
    hue="target_top10",
    alpha=0.45,
)
ax.set_title("Grid position vs finish position")
ax.set_xlabel("Grid position")
ax.set_ylabel("Finish position")
ax.invert_yaxis()
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
avg_points_by_grid = (
    df[df["season"].between(2014, 2025)]
    .dropna(subset=["grid_position"])
    .groupby("grid_position")["points"]
    .mean()
    .reset_index()
)

ax = sns.lineplot(data=avg_points_by_grid, x="grid_position", y="points", marker="o")
ax.set_title("Average points by starting grid position")
ax.set_xlabel("Grid position")
ax.set_ylabel("Average points")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
circuit_profile = (
    df[df["season"].between(2014, 2025)]
    .groupby("raceName")
    .agg(
        avg_points=("points", "mean"),
        podium_rate=("target_podium", "mean"),
        top10_rate=("target_top10", "mean"),
        races=("round", "count"),
    )
    .sort_values("avg_points", ascending=False)
    .head(15)
    .reset_index()
)

display(circuit_profile)
"""
    ),
    md(
        """
# Data Quality and Missing Values

The model intentionally keeps missingness indicators for cold starts and limited history. We still inspect missing values to understand where imputations will happen.
"""
    ),
    code(
        """
missing_profile = (
    df.isna()
    .mean()
    .sort_values(ascending=False)
    .head(25)
    .reset_index()
)
missing_profile.columns = ["column", "missing_rate"]

ax = sns.barplot(data=missing_profile, y="column", x="missing_rate", color="#8172B3")
ax.set_title("Top missing-value rates")
ax.set_xlabel("Missing rate")
ax.set_ylabel("")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
# Feature Engineering

The central modeling rule is simple: **a feature may use current grid/qualifying information or past race information, but never the current race result**.

Rolling features such as `driver_avg_finish_last_5` and `constructor_points_last_5` are computed with `shift(1)` before the rolling window. This prevents the current race from leaking into its own prediction.
"""
    ),
    code(
        """
feature_groups = {
    "Grid and qualifying": ["grid", "grid_position", "quali_position", "grid_vs_quali_delta", "qualified_top10", "started_top10"],
    "Driver rolling form": [c for c in df.columns if c.startswith("driver_") and ("last_" in c or "before" in c)],
    "Constructor rolling form": [c for c in df.columns if c.startswith("constructor_") and ("last_" in c or "before" in c)],
    "Teammate features": [c for c in df.columns if c.startswith("teammate_") or c.endswith("_vs_teammate")],
    "Circuit history": [c for c in df.columns if c.startswith("circuit_")],
    "Missingness flags": [c for c in df.columns if c.startswith("missing_")],
}

for group, columns in feature_groups.items():
    print(f"\\n{group} ({len(columns)} columns)")
    print(columns[:20])
"""
    ),
    code(
        """
selected_features = select_feature_columns(df)
print(f"Selected model features: {len(selected_features)}")
display(pd.DataFrame({"feature": selected_features}).head(80))
"""
    ),
    code(
        """
feature_presence = pd.DataFrame({
    "feature": selected_features,
    "missing_rate": df[selected_features].isna().mean().values,
    "std": df[selected_features].std(numeric_only=True).values,
}).sort_values("missing_rate", ascending=False)

display(feature_presence.head(20))
"""
    ),
    code(
        """
example_driver = (
    df[df["season"].between(2023, 2026)]
    .groupby("driver_name")
    .size()
    .sort_values(ascending=False)
    .index[0]
)

driver_timeline = df[df["driver_name"].eq(example_driver)].sort_values(["season", "round"]).copy()

cols_to_show = [
    "season", "round", "raceName", "positionOrder", "points",
    "driver_avg_finish_last_5", "driver_points_last_5", "driver_top10_rate_last_10"
]

display(Markdown(f"### Rolling-feature example: {example_driver}"))
display(driver_timeline[cols_to_show].tail(18))

ax = sns.lineplot(data=driver_timeline, x=np.arange(len(driver_timeline)), y="positionOrder", label="Actual finish")
sns.lineplot(data=driver_timeline, x=np.arange(len(driver_timeline)), y="driver_avg_finish_last_5", label="Rolling avg finish before race", ax=ax)
ax.set_title(f"Leakage-safe rolling finish feature for {example_driver}")
ax.set_xlabel("Chronological driver race index")
ax.set_ylabel("Finish position")
ax.invert_yaxis()
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
important_feature_candidates = [
    "grid_position",
    "quali_position",
    "driver_points_last_5",
    "driver_avg_finish_last_5",
    "constructor_points_last_5",
    "constructor_avg_finish_last_5",
    "driver_points_share_before",
    "constructor_points_share_before",
    "driver_top10_rate_last_10",
    "constructor_top10_rate_last_10",
]
available_corr_features = [c for c in important_feature_candidates if c in df.columns]
corr_cols = available_corr_features + ["target_top10", "target_podium", "target_points"]

corr = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(12, 9))
sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, annot=True, fmt=".2f", linewidths=0.5)
plt.title("Correlation matrix for key features and targets")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

plot_specs = [
    ("driver_points_last_5", "target_top10", "Driver points last 5 vs top 10"),
    ("constructor_points_last_5", "target_top10", "Constructor points last 5 vs top 10"),
    ("driver_avg_finish_last_5", "target_podium", "Driver avg finish last 5 vs podium"),
    ("quali_position", "target_podium", "Qualifying position vs podium"),
]

for ax, (feature, target, title) in zip(axes.ravel(), plot_specs):
    if feature in df.columns:
        sns.boxplot(data=df[df["season"].between(2014, 2025)], x=target, y=feature, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(target)
        ax.set_ylabel(feature)

plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
# Temporal Split

The project uses a chronological split:

- train: 2014-2023
- validation: 2024
- official test: 2025
- live/application: 2026

This avoids random train-test leakage across time.
"""
    ),
    code(
        """
parts = temporal_split(df, cfg)

split_summary = []
for split_name, split_df in parts.items():
    split_summary.append({
        "split": split_name,
        "rows": len(split_df),
        "min_season": int(split_df["season"].min()) if len(split_df) else None,
        "max_season": int(split_df["season"].max()) if len(split_df) else None,
        "n_seasons": split_df["season"].nunique(),
    })

split_summary = pd.DataFrame(split_summary)
display(split_summary)

ax = sns.barplot(data=split_summary, x="split", y="rows", palette="Set2")
ax.set_title("Rows by temporal split")
ax.set_xlabel("Split")
ax.set_ylabel("Rows")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
split_target_rates = []
for split_name, split_df in parts.items():
    split_target_rates.append({
        "split": split_name,
        "top10_rate": split_df["target_top10"].mean(),
        "podium_rate": split_df["target_podium"].mean(),
        "avg_points": split_df["target_points"].mean(),
    })

split_target_rates = pd.DataFrame(split_target_rates)
display(split_target_rates)
"""
    ),
    md(
        """
# Model Training Artifacts

The project trains:

- an XGBoost classifier for top 10;
- an XGBoost classifier for podium, with class imbalance handling;
- an XGBoost regressor for expected points.

If XGBoost is unavailable, the project code falls back to Random Forest models.
"""
    ),
    code(
        """
model_dir = paths["models"]
model_paths = {
    "top10": model_dir / "top10_model.pkl",
    "podium": model_dir / "podium_model.pkl",
    "points": model_dir / "points_model.pkl",
    "preprocessor": model_dir / "preprocessing_pipeline.pkl",
    "feature_columns": model_dir / "feature_columns.json",
}

artifact_table = pd.DataFrame({
    "artifact": model_paths.keys(),
    "path": [str(p) for p in model_paths.values()],
    "exists": [p.exists() for p in model_paths.values()],
})
display(artifact_table)
"""
    ),
    code(
        """
models = {}
for name in ["top10", "podium", "points"]:
    if model_paths[name].exists():
        models[name] = joblib.load(model_paths[name])

saved_features = json.loads(model_paths["feature_columns"].read_text(encoding="utf-8")) if model_paths["feature_columns"].exists() else []

for name, model in models.items():
    print(f"{name}: {type(model)}")
    if hasattr(model, "get_params"):
        params = model.get_params()
        display(pd.Series({
            "n_estimators": params.get("n_estimators"),
            "learning_rate": params.get("learning_rate"),
            "max_depth": params.get("max_depth"),
            "scale_pos_weight": params.get("scale_pos_weight"),
            "objective": params.get("objective"),
        }).to_frame("value"))

print(f"Saved feature columns: {len(saved_features)}")
"""
    ),
    md(
        """
# Evaluation Metrics

This section loads the saved metrics from `reports/metrics/` and visualizes model performance on validation 2024 and official test 2025.
"""
    ),
    code(
        """
metrics_dir = paths["reports"] / "metrics"

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

top10_metrics = load_json(metrics_dir / "top10_metrics.json")
podium_metrics = load_json(metrics_dir / "podium_metrics.json")
points_metrics = load_json(metrics_dir / "points_metrics.json")
baseline_metrics = load_json(metrics_dir / "baseline_metrics.json")

display(Markdown("## Top 10 metrics"))
display(pd.DataFrame(top10_metrics).T)

display(Markdown("## Podium metrics"))
display(pd.DataFrame(podium_metrics).T)

display(Markdown("## Points metrics"))
display(pd.DataFrame(points_metrics).T)
"""
    ),
    code(
        """
classification_records = []
for task_name, metric_dict in [("top10", top10_metrics), ("podium", podium_metrics)]:
    for split_name, values in metric_dict.items():
        for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision", "brier_score"]:
            classification_records.append({
                "task": task_name,
                "split": split_name,
                "metric": metric_name,
                "value": values.get(metric_name),
            })

classification_metric_df = pd.DataFrame(classification_records)

g = sns.catplot(
    data=classification_metric_df,
    x="metric",
    y="value",
    hue="split",
    col="task",
    kind="bar",
    height=5,
    aspect=1.4,
)
g.set_xticklabels(rotation=45, ha="right")
g.set_titles("{col_name} classification metrics")
g.set_axis_labels("", "Value")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
regression_records = []
for split_name, values in points_metrics.items():
    for metric_name, value in values.items():
        regression_records.append({"split": split_name, "metric": metric_name, "value": value})

regression_metric_df = pd.DataFrame(regression_records)

ax = sns.barplot(data=regression_metric_df, x="metric", y="value", hue="split")
ax.set_title("Expected-points regression metrics")
ax.set_xlabel("")
ax.set_ylabel("Value")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
figures_dir = paths["reports"] / "figures"

figure_files = [
    "confusion_matrix_top10_test.png",
    "confusion_matrix_podium_test.png",
    "roc_top10_test.png",
    "roc_podium_test.png",
    "precision_recall_podium_test.png",
    "predicted_vs_actual_points_test.png",
]

for file_name in figure_files:
    path = figures_dir / file_name
    if path.exists():
        display(Markdown(f"### {file_name}"))
        display(Image(filename=str(path)))
"""
    ),
    md(
        """
# Feature Importance

Feature importance helps explain which variables the trained models rely on most. These charts come from the fitted model artifacts.
"""
    ),
    code(
        """
for file_name in [
    "feature_importance_top10.png",
    "feature_importance_podium.png",
    "feature_importance_points.png",
]:
    path = figures_dir / file_name
    if path.exists():
        display(Markdown(f"### {file_name}"))
        display(Image(filename=str(path)))
"""
    ),
    code(
        """
importance_frames = []
for task_name, model in models.items():
    values = getattr(model, "feature_importances_", None)
    if values is not None and saved_features:
        tmp = pd.DataFrame({
            "feature": saved_features,
            "importance": values,
            "task": task_name,
        }).sort_values("importance", ascending=False).head(20)
        importance_frames.append(tmp)

if importance_frames:
    importance_df = pd.concat(importance_frames, ignore_index=True)
    display(importance_df)

    g = sns.catplot(
        data=importance_df,
        y="feature",
        x="importance",
        col="task",
        kind="bar",
        sharey=False,
        height=7,
        aspect=0.8,
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("Importance", "")
    plt.tight_layout()
    plt.show()
"""
    ),
    md(
        """
# Baseline Comparison

Baselines are important because Formula 1 is highly grid-dependent. A model must be compared against simple grid- and qualifying-based heuristics.
"""
    ),
    code(
        """
baseline_records = []
for split_name, split_values in baseline_metrics.items():
    for task_name in ["top10", "podium"]:
        for baseline_name, metric_values in split_values.get(task_name, {}).items():
            baseline_records.append({
                "split": split_name,
                "task": task_name,
                "baseline": baseline_name,
                "f1": metric_values.get("f1"),
                "roc_auc": metric_values.get("roc_auc"),
                "accuracy": metric_values.get("accuracy"),
            })

baseline_df = pd.DataFrame(baseline_records)
display(baseline_df)

if not baseline_df.empty:
    ax = sns.barplot(data=baseline_df, x="baseline", y="f1", hue="split")
    ax.set_title("Classification baseline F1 scores")
    ax.set_xlabel("Baseline")
    ax.set_ylabel("F1")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()
"""
    ),
    md(
        """
# 2026 Live Predictions

The prediction layer exports dashboard-ready CSV files. The notebook checks whether probabilities are valid and visualizes the live/application predictions.
"""
    ),
    code(
        """
prediction_dir = paths["predictions"]
combined_prediction_path = prediction_dir / "2026_combined_predictions.csv"
missing_inputs_path = prediction_dir / "2026_missing_inputs.csv"

pred_2026 = pd.read_csv(combined_prediction_path) if combined_prediction_path.exists() else pd.DataFrame()
missing_2026 = pd.read_csv(missing_inputs_path) if missing_inputs_path.exists() else pd.DataFrame()

print(f"2026 predictions shape: {pred_2026.shape}")
print(f"2026 missing inputs shape: {missing_2026.shape}")

display(pred_2026.head())
display(missing_2026.head())
"""
    ),
    code(
        """
if not pred_2026.empty:
    checks = {
        "top10_probability_valid": pred_2026["proba_top10"].between(0, 1).all(),
        "podium_probability_valid": pred_2026["proba_podium"].between(0, 1).all(),
        "expected_points_non_negative": pred_2026["expected_points"].ge(0).all(),
    }
    display(pd.Series(checks).to_frame("passed"))
"""
    ),
    code(
        """
if not pred_2026.empty:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.histplot(pred_2026["proba_top10"], bins=20, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Predicted top 10 probability")

    sns.histplot(pred_2026["proba_podium"], bins=20, ax=axes[1], color="#55A868")
    axes[1].set_title("Predicted podium probability")

    sns.histplot(pred_2026["expected_points"], bins=20, ax=axes[2], color="#C44E52")
    axes[2].set_title("Expected points")

    plt.tight_layout()
    plt.show()
"""
    ),
    code(
        """
if not pred_2026.empty:
    top_expected_points = (
        pred_2026.sort_values("expected_points", ascending=False)
        .head(20)
        .copy()
    )

    ax = sns.barplot(data=top_expected_points, y="driver_name", x="expected_points", hue="constructor_name", dodge=False)
    ax.set_title("Top 2026 expected-points predictions")
    ax.set_xlabel("Expected points")
    ax.set_ylabel("")
    plt.legend(title="Constructor", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
"""
    ),
    code(
        """
if not pred_2026.empty:
    round_summary = (
        pred_2026.groupby(["round", "raceName"])
        .agg(
            drivers=("driver_name", "count"),
            avg_top10_probability=("proba_top10", "mean"),
            avg_podium_probability=("proba_podium", "mean"),
            total_expected_points=("expected_points", "sum"),
        )
        .reset_index()
    )

    display(round_summary)

    ax = sns.lineplot(data=round_summary, x="round", y="total_expected_points", marker="o")
    ax.set_title("Total expected points by 2026 round represented in predictions")
    ax.set_xlabel("Round")
    ax.set_ylabel("Total expected points")
    plt.tight_layout()
    plt.show()
"""
    ),
    md(
        """
# Risks, Limitations, and Next Steps

This V1 is intentionally focused and reproducible. It does not use weather, tire strategy, detailed telemetry, safety-car context, or free-practice pace.

The main current limitation is 2026 coverage: the Jolpica 2026 schedule contains future races, but the driver-race prediction table only contains rounds with available driver-level result/qualification rows. Future schedule-only races should be exported as missing application inputs instead of being silently absent.
"""
    ),
    code(
        """
audit_notes = []

train_seasons = set(parts["train"]["season"].unique())
if 2026 in train_seasons:
    audit_notes.append("2026 appears in the training split.")
else:
    audit_notes.append("PASS: 2026 is not used in the default training split.")

if 2025 in train_seasons:
    audit_notes.append("2025 appears in the training split.")
else:
    audit_notes.append("PASS: 2025 is reserved for official testing.")

if schedule_2026_path.exists():
    missing_schedule_rounds = schedule_2026.loc[~schedule_2026["in_final_dataset"], ["round", "raceName", "date"]]
    audit_notes.append(f"2026 schedule rounds absent from final dataset: {len(missing_schedule_rounds)}")
    display(missing_schedule_rounds)

for note in audit_notes:
    print(note)
"""
    ),
    md(
        """
# Final Summary

The project now has a single notebook that follows a portfolio-style data science structure:

1. project initialization;
2. data loading and audit;
3. exploratory data analysis;
4. feature engineering explanation;
5. temporal split;
6. model artifacts;
7. metrics and figures;
8. feature importance;
9. baseline comparison;
10. 2026 live predictions;
11. limitations and next steps.

The notebook is meant to be read from top to bottom and used as the main presentation artifact for the V1.
"""
    ),
]


def main():
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }
    nb["cells"] = cells
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH}")
    print(f"Cells: {len(cells)}")


if __name__ == "__main__":
    main()
