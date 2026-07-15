"""Create a GitHub-ready end-to-end notebook for the F1 Predictor project."""
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "F1_Predictor_GitHub_Portfolio.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(
        """
# F1 Predictor: End-to-End Formula 1 Race Outcome Prediction

This notebook presents a complete, reproducible machine learning project for Formula 1 race prediction.

The project predicts, for each driver and Grand Prix:

- the probability of finishing in the **top 10**;
- the probability of finishing on the **podium**;
- the driver's **expected Grand Prix points**;
- a race-level ranking score used to build a coherent predicted top 10.

The project uses:

- Kaggle historical Formula 1 data for 2014-2024;
- Jolpica-F1 snapshots for 2025 official testing;
- Jolpica-F1 2026 snapshots and schedule data for live race prediction;
- leakage-safe rolling features;
- temporal validation;
- separate pre-qualifying and post-qualifying model families;
- a Streamlit dashboard for interactive 2026 race predictions.
"""
    ),
    md(
        """
## 1. Project Design

The project separates three concepts that are often mixed in sports prediction projects:

1. **Evaluation model**  
   Frozen temporal benchmark: train on 2014-2023, validate on 2024, test on 2025.

2. **Live production model**  
   Trained for the dashboard using all completed information up to a configurable cutoff date.

3. **Future race inputs**  
   Driver-race rows for upcoming 2026 races, built without inventing future qualifying, grid positions, results, or points.

This distinction is essential because a model can only use information that would actually be known before the race.
"""
    ),
    md("## 2. Imports and Configuration"),
    code(
        """
from pathlib import Path
import sys
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from IPython.display import display, Markdown, Image
except ImportError:
    def display(x):
        print(x)
    def Markdown(x):
        return x
    class Image:
        def __init__(self, filename):
            self.filename = filename
        def __repr__(self):
            return f"Image(filename={self.filename!r})"

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.05)
plt.rcParams["figure.figsize"] = (12, 6)

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.modeling import select_feature_columns, QUALI_FEATURES
from src.split import temporal_split

cfg = load_config(PROJECT_ROOT / "config.yaml")
paths = cfg["resolved_paths"]
print(PROJECT_ROOT)
"""
    ),
    code(
        """
display(pd.DataFrame({
    "component": ["processed data", "external data", "models", "reports", "predictions"],
    "path": [
        paths["processed_data_dir"],
        paths["external_data_dir"],
        paths["models_dir"],
        paths["reports_dir"],
        paths["predictions_dir"],
    ],
    "exists": [
        paths["processed_data_dir"].exists(),
        paths["external_data_dir"].exists(),
        paths["models_dir"].exists(),
        paths["reports_dir"].exists(),
        paths["predictions_dir"].exists(),
    ],
}))
"""
    ),
    md("## 3. Reproducible Pipeline Commands"),
    code(
        """
commands = [
    "python scripts/fetch_jolpica_data.py --seasons 2025 2026",
    "python scripts/build_dataset.py",
    "python scripts/audit_project.py",
    "python scripts/build_future_2026_inputs.py --as-of-date 2026-07-12",
    "python scripts/train_live_models.py",
    "python scripts/predict_selected_race.py --round 10 --as-of-date 2026-07-12",
    "streamlit run app/streamlit_app.py",
]

for command in commands:
    print(command)
"""
    ),
    md(
        """
## 4. Load the Final Driver-Race Dataset

The dataset contains one row per driver and race. All target columns are race-level outcomes.

Important semantic decision:

- `target_points` represents **Grand Prix race points**.
- `grand_prix_points`, `sprint_points`, and `event_points` are separated to avoid mixing race points with championship standings.
- Official standings before a race are modeled as separate features.
"""
    ),
    code(
        """
dataset_path = paths["processed_data_dir"] / "f1_driver_race_dataset_with_jolpica.csv"
df = pd.read_csv(dataset_path)

print(f"Dataset shape: {df.shape}")
display(df.head())
"""
    ),
    code(
        """
season_counts = df.groupby("season").size().reset_index(name="driver_race_rows")
display(season_counts)

ax = sns.barplot(data=season_counts, x="season", y="driver_race_rows", color="#d62728")
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
key_cols = ["season", "round", "driverId"]
print("Duplicate driver-race keys:", df.duplicated(key_cols).sum())
display(df.groupby("season")[["target_top10", "target_podium", "target_points"]].apply(lambda x: x.isna().sum()))
"""
    ),
    md("## 5. Jolpica 2025/2026 Coverage"),
    code(
        """
jolpica_dir = paths["external_data_dir"] / "jolpica"
coverage = []
for season in [2025, 2026]:
    for name in ["schedule", "results", "qualifying", "driver_standings", "constructor_standings"]:
        p = jolpica_dir / f"jolpica_{season}_{name}.csv"
        if p.exists():
            tmp = pd.read_csv(p)
            rounds = tmp["round"].nunique() if "round" in tmp else np.nan
            rows = len(tmp)
        else:
            rounds = np.nan
            rows = 0
        coverage.append({"season": season, "snapshot": name, "rows": rows, "rounds": rounds})

coverage = pd.DataFrame(coverage)
display(coverage)

ax = sns.barplot(data=coverage, x="snapshot", y="rows", hue="season")
ax.set_title("Jolpica snapshot coverage")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
schedule_2026 = pd.read_csv(jolpica_dir / "jolpica_2026_schedule.csv")
completed_2026_rounds = set(df[df["season"].eq(2026)]["round"].astype(int))
schedule_2026["in_completed_dataset"] = schedule_2026["round"].astype(int).isin(completed_2026_rounds)
display(schedule_2026[["round", "raceName", "date", "in_completed_dataset"]])
"""
    ),
    md("## 6. Exploratory Data Analysis"),
    code(
        """
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df["positionOrder"].dropna(), bins=22, ax=axes[0], color="#1f77b4")
axes[0].set_title("Finish position distribution")
sns.histplot(df["target_points"].dropna(), bins=25, ax=axes[1], color="#2ca02c")
axes[1].set_title("Grand Prix points distribution")
sns.histplot(df["grid_position"].dropna(), bins=22, ax=axes[2], color="#d62728")
axes[2].set_title("Grid position distribution")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
rates = df.groupby("season")[["target_top10", "target_podium"]].mean().reset_index()
rates = rates.melt(id_vars="season", var_name="target", value_name="positive_rate")

ax = sns.lineplot(data=rates, x="season", y="positive_rate", hue="target", marker="o")
ax.set_title("Target rate over time")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
constructor_2026 = (
    df[df["season"].eq(2026)]
    .groupby("constructor_name")
    .agg(points=("target_points", "sum"), top10_rate=("target_top10", "mean"), podiums=("target_podium", "sum"))
    .sort_values("points", ascending=False)
    .reset_index()
)
display(constructor_2026)

ax = sns.barplot(data=constructor_2026, y="constructor_name", x="points", palette="rocket")
ax.set_title("2026 constructor performance up to the cutoff")
ax.set_xlabel("Grand Prix points in local dataset")
ax.set_ylabel("")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
driver_2026 = (
    df[df["season"].eq(2026)]
    .groupby("driver_name")
    .agg(points=("target_points", "sum"), top10_rate=("target_top10", "mean"), podiums=("target_podium", "sum"), avg_finish=("positionOrder", "mean"))
    .sort_values("points", ascending=False)
    .head(20)
    .reset_index()
)
display(driver_2026)

ax = sns.barplot(data=driver_2026, y="driver_name", x="points", palette="mako")
ax.set_title("Top 2026 drivers up to the cutoff")
ax.set_xlabel("Grand Prix points in local dataset")
ax.set_ylabel("")
plt.tight_layout()
plt.show()
"""
    ),
    md("## 7. Leakage-Safe Feature Engineering"),
    code(
        """
pre_quali_features = select_feature_columns(df, mode="pre_quali")
post_quali_features = select_feature_columns(df, mode="post_quali")

print(f"Pre-qualifying features: {len(pre_quali_features)}")
print(f"Post-qualifying features: {len(post_quali_features)}")
print("Qualifying features excluded from pre-quali:", sorted(set(post_quali_features) & QUALI_FEATURES))
assert not (set(pre_quali_features) & QUALI_FEATURES)
"""
    ),
    code(
        """
feature_groups = pd.DataFrame({
    "feature_group": [
        "pre_quali_features",
        "post_quali_features",
        "qualifying_only_difference",
    ],
    "count": [
        len(pre_quali_features),
        len(post_quali_features),
        len(set(post_quali_features) - set(pre_quali_features)),
    ],
})
display(feature_groups)

ax = sns.barplot(data=feature_groups, x="feature_group", y="count", color="#d62728")
ax.set_title("Feature schema by prediction mode")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
key_features = [
    "driver_points_last_5", "constructor_points_last_5",
    "driver_top10_rate_last_10", "constructor_top10_rate_last_10",
    "driver_official_standings_points_before",
    "constructor_official_standings_points_before",
    "grid_position", "quali_position",
    "target_top10", "target_podium", "target_points",
]
key_features = [c for c in key_features if c in df.columns]

plt.figure(figsize=(12, 8))
sns.heatmap(df[key_features].corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f")
plt.title("Correlation of selected engineered features and targets")
plt.tight_layout()
plt.show()
"""
    ),
    md("## 8. Temporal Split"),
    code(
        """
parts = temporal_split(df, cfg)
split_summary = pd.DataFrame([
    {
        "split": name,
        "rows": len(part),
        "min_season": part["season"].min(),
        "max_season": part["season"].max(),
    }
    for name, part in parts.items()
])
display(split_summary)

ax = sns.barplot(data=split_summary, x="split", y="rows", palette="Set2")
ax.set_title("Official temporal split")
plt.tight_layout()
plt.show()
"""
    ),
    md("## 9. Baseline Evaluation Metrics"),
    code(
        """
def load_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

metrics_dir = paths["reports_dir"] / "metrics"
top10_metrics = load_json(metrics_dir / "top10_metrics.json")
podium_metrics = load_json(metrics_dir / "podium_metrics.json")
points_metrics = load_json(metrics_dir / "points_metrics.json")

display(Markdown("### Top 10"))
display(pd.DataFrame(top10_metrics).T)
display(Markdown("### Podium"))
display(pd.DataFrame(podium_metrics).T)
display(Markdown("### Points"))
display(pd.DataFrame(points_metrics).T)
"""
    ),
    code(
        """
baseline_records = []
for task, metric_dict in [("top10", top10_metrics), ("podium", podium_metrics)]:
    for split, values in metric_dict.items():
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision", "brier_score"]:
            baseline_records.append({"task": task, "split": split, "metric": metric, "value": values.get(metric)})

baseline_plot = pd.DataFrame(baseline_records)
g = sns.catplot(data=baseline_plot, x="metric", y="value", hue="split", col="task", kind="bar", height=4.5, aspect=1.4)
g.set_xticklabels(rotation=45, ha="right")
g.set_titles("{col_name}")
plt.tight_layout()
plt.show()
"""
    ),
    md("## 10. New Pre-Qualifying and Post-Qualifying Models"),
    code(
        """
pre_quali_metrics = load_json(metrics_dir / "pre_quali_metrics.json")
post_quali_metrics = load_json(metrics_dir / "post_quali_metrics.json")
ranking_metrics = load_json(metrics_dir / "ranking_metrics.json")

display(Markdown("### Pre-qualifying evaluation"))
display(pd.DataFrame({
    "top10_test": pre_quali_metrics.get("test", {}).get("top10", {}),
    "podium_test": pre_quali_metrics.get("test", {}).get("podium", {}),
    "points_test": pre_quali_metrics.get("test", {}).get("points", {}),
}).T)

display(Markdown("### Post-qualifying evaluation"))
display(pd.DataFrame({
    "top10_test": post_quali_metrics.get("test", {}).get("top10", {}),
    "podium_test": post_quali_metrics.get("test", {}).get("podium", {}),
    "points_test": post_quali_metrics.get("test", {}).get("points", {}),
}).T)

display(Markdown("### Ranking metrics"))
display(pd.DataFrame(ranking_metrics).T)
"""
    ),
    code(
        """
summary_path = metrics_dir / "new_model_results_summary.csv"
model_summary = pd.read_csv(summary_path)
display(model_summary)

plot_metrics = model_summary[model_summary["split"].eq("test")]
plot_metrics = plot_metrics[plot_metrics["task"].isin(["top10", "podium"])]
plot_metrics = plot_metrics.melt(
    id_vars=["model_family", "split", "task"],
    value_vars=["f1", "roc_auc", "average_precision", "brier_score"],
    var_name="metric",
    value_name="value",
)

g = sns.catplot(data=plot_metrics, x="metric", y="value", hue="model_family", col="task", kind="bar", height=5, aspect=1.4)
g.set_xticklabels(rotation=45, ha="right")
g.set_titles("{col_name} test comparison")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
### Interpretation

The original 2025 benchmark remains the official frozen evaluation baseline. The new pre/post-qualifying model families are saved and compared, but they should not be described as universally better unless the relevant metrics confirm it.

The post-qualifying model can use real grid and qualifying information. The pre-qualifying model cannot, which makes it more realistic before qualifying but less informative.
"""
    ),
    md("## 11. Temporal Hyperparameter Search"),
    code(
        """
tuning_summary_path = paths["reports_dir"] / "model_tuning_summary.csv"
tuning_trials_path = metrics_dir / "temporal_tuning_trials.json"

tuning_summary = pd.read_csv(tuning_summary_path) if tuning_summary_path.exists() else pd.DataFrame()
display(tuning_summary)

if tuning_trials_path.exists():
    trials = pd.DataFrame(load_json(tuning_trials_path))
    display(trials.head())
    ax = sns.barplot(data=trials, x="task", y="mean_score", color="#d62728")
    ax.set_title("Temporal tuning mean score by task")
    plt.tight_layout()
    plt.show()
"""
    ),
    md(
        """
## 12. Standings and Points Reconciliation

Race points and championship points are not the same thing when sprint events exist. The project now keeps separate columns:

- `grand_prix_points`
- `sprint_points`
- `event_points`
- `driver_official_standings_points_before`
- `constructor_official_standings_points_before`

The reconciliation report compares reconstructed event points with official standings snapshots when available.
"""
    ),
    code(
        """
reconciliation_path = paths["reports_dir"] / "data_quality" / "standings_reconciliation.csv"
reconciliation = pd.read_csv(reconciliation_path)
display(reconciliation.head())

diff = pd.to_numeric(reconciliation["difference"], errors="coerce")
nonzero = reconciliation[diff.abs().fillna(0) > 0.01]
print(f"Rows with non-zero reconstructed vs official difference: {len(nonzero)}")
display(nonzero.head(20))
"""
    ),
    md(
        """
## 13. Future 2026 Race Inputs

The model cannot predict future races from results that do not exist yet. Instead, the project builds future driver-race rows from:

- the Jolpica 2026 schedule;
- the latest completed 2026 race before the cutoff;
- known driver and constructor entries;
- historical and 2026 rolling features computed before the selected race.

No future grid, qualifying, result or points are invented.
"""
    ),
    code(
        """
future_path = paths["processed_data_dir"] / "2026_future_race_inputs.csv"
future_inputs = pd.read_csv(future_path)
print(f"Future inputs shape: {future_inputs.shape}")
display(future_inputs[["round", "raceName", "date", "driver_name", "constructor_name", "prediction_mode", "missing_qualifying"]].head(25))
"""
    ),
    code(
        """
future_rounds = future_inputs.groupby(["round", "raceName", "date", "prediction_mode"]).size().reset_index(name="drivers")
display(future_rounds)

ax = sns.barplot(data=future_rounds, x="round", y="drivers", hue="prediction_mode")
ax.set_title("Future 2026 race inputs by round")
ax.set_xlabel("Round")
ax.set_ylabel("Driver rows")
plt.tight_layout()
plt.show()
"""
    ),
    md("## 14. Next Race Prediction: Belgian Grand Prix"),
    code(
        """
next_prediction_path = paths["predictions_dir"] / "2026_next_races_predictions.csv"
next_prediction = pd.read_csv(next_prediction_path)
display(next_prediction)

checks = {
    "exactly_10_rows": len(next_prediction) == cfg["prediction"]["top_n"],
    "unique_drivers": next_prediction["driver_name"].is_unique,
    "top10_probability_valid": next_prediction["proba_top10"].between(0, 1).all(),
    "podium_probability_valid": next_prediction["proba_podium"].between(0, 1).all(),
    "expected_points_non_negative": next_prediction["expected_points"].ge(0).all(),
    "rank_score_descending": next_prediction["race_rank_score"].is_monotonic_decreasing,
}
display(pd.Series(checks).to_frame("passed"))
"""
    ),
    code(
        """
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.barplot(data=next_prediction, y="driver_name", x="expected_points", hue="constructor_name", dodge=False, ax=axes[0])
axes[0].set_title("Expected points for predicted top 10")
axes[0].set_ylabel("")
axes[0].legend(loc="lower right")

sns.barplot(data=next_prediction, y="driver_name", x="race_rank_score", color="#d62728", ax=axes[1])
axes[1].set_title("Race rank score")
axes[1].set_ylabel("")
plt.tight_layout()
plt.show()
"""
    ),
    md("## 15. Saved Model Artifacts"),
    code(
        """
artifact_rows = []
for family in ["evaluation", "live"]:
    family_dir = paths["models_dir"] / family
    for p in sorted(family_dir.glob("*")):
        artifact_rows.append({"family": family, "artifact": p.name, "size_kb": round(p.stat().st_size / 1024, 1)})

artifacts = pd.DataFrame(artifact_rows)
display(artifacts)
"""
    ),
    md("## 16. Streamlit Dashboard"),
    code(
        """
dashboard_files = [
    "app/streamlit_app.py",
    "app/components.py",
    "app/data_service.py",
    "app/prediction_service.py",
    ".streamlit/config.toml",
]

display(pd.DataFrame({
    "file": dashboard_files,
    "exists": [(PROJECT_ROOT / f).exists() for f in dashboard_files],
}))

print("Run the dashboard with:")
print("streamlit run app/streamlit_app.py")
"""
    ),
    md(
        """
The dashboard provides:

- race selector for future 2026 races;
- PRE-QUALIFYING / POST-QUALIFYING mode detection;
- predicted top 10 table;
- probabilities, expected points, ranking score and confidence level;
- prediction context and data status pages;
- downloadable CSV output.
"""
    ),
    md("## 17. Tests and Quality Gates"),
    code(
        """
test_files = sorted((PROJECT_ROOT / "tests").glob("test_*.py"))
display(pd.DataFrame({"test_file": [str(p.relative_to(PROJECT_ROOT)) for p in test_files]}))

print("Run tests with:")
print("python -m pytest tests")
"""
    ),
    md(
        """
The implemented tests check:

- exactly ten drivers in the predicted top 10;
- no duplicate drivers;
- probabilities are between 0 and 1;
- expected points are non-negative;
- ranking score is sorted descending;
- pre-qualifying feature schema excludes qualifying and grid features.
"""
    ),
    md("## 18. Main Limitations"),
    md(
        """
- Sprint points are structurally separated, but full sprint result integration still depends on available Jolpica sprint feeds.
- Context/news adjustments are intentionally not hidden inside the model. Optional manual adjustments must be auditable in `data/external/2026_context_adjustments.csv`.
- The temporal tuning implemented here is a lightweight reproducible search, not a full Optuna study.
- SHAP explanations are not computed in the dashboard V1.
- Future predictions before qualifying are provisional by design.
- Post-qualifying predictions should only be used when real qualifying and grid data are available.
"""
    ),
    md("## 19. Final Takeaway"),
    md(
        """
This project is now more than a static historical classifier. It includes:

- a reproducible historical modeling pipeline;
- official temporal evaluation;
- future 2026 input construction;
- pre-qualifying and post-qualifying prediction modes;
- ranking for coherent race top 10 output;
- saved live production artifacts;
- a Streamlit dashboard ready to run locally;
- audit and reconciliation reports.

The most important methodological rule is preserved throughout the project:

> A prediction can only use information available before the selected race.
"""
    ),
]


def main():
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb["cells"] = cells
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH}")
    print(f"Cells: {len(cells)}")


if __name__ == "__main__":
    main()
