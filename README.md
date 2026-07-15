# F1 Predictor

End-to-end Formula 1 race outcome prediction project built as a Data Science / Machine Learning portfolio case study.

The goal is to predict, for each driver and Grand Prix:

- probability of finishing in the **top 10**;
- probability of finishing on the **podium**;
- expected **Grand Prix points**;
- a race-level ranking score used to build a coherent predicted top 10.

The project focuses on temporal validation, leakage-safe feature engineering, reproducibility, and honest model evaluation.

## Project Highlights

- Historical Formula 1 dataset pipeline using Kaggle-style CSV tables.
- Jolpica-F1 integration for recent seasons and 2026 schedule/live context.
- Driver-race modeling table with one row per driver per race.
- Rolling and expanding features computed with `shift(1)` to avoid using the current race result.
- Official temporal split:
  - train: 2014-2023
  - validation: 2024
  - test: 2025
  - live/application context: 2026
- XGBoost classification and regression models.
- Separate **pre-qualifying** and **post-qualifying** model families.
- Ranking model for a more coherent predicted race top 10.
- Data quality audit and standings reconciliation reports.
- A single executed portfolio notebook with plots, metrics, explanations, and limitations.

## Main Notebook

The main GitHub notebook is:

```text
notebooks/F1_Predictor_End_to_End_Portfolio_Notebook.ipynb
```

It is designed to be read from top to bottom and includes:

- project objective and data sources;
- exploratory data analysis;
- feature engineering explanation;
- leakage checks;
- temporal split;
- model metrics;
- baseline comparison;
- feature importance;
- 2026 live prediction logic;
- limitations and next steps.

## Data Sources

This repository does **not** include the full raw datasets or generated processed CSV files.

The project expects:

- local Formula 1 historical CSV files, for example in `data_f1/` or `data/raw/`;
- Jolpica-F1 snapshots generated with the provided scripts.

Large data files and generated prediction outputs are excluded from GitHub through `.gitignore`.

## Methodology

The modeling unit is a **driver-race row**.

The pipeline creates targets:

- `target_top10`
- `target_podium`
- `target_points`

`target_points` represents **Grand Prix race points**. Sprint/championship points are treated separately through:

- `grand_prix_points`
- `sprint_points`
- `event_points`
- `driver_official_standings_points_before`
- `constructor_official_standings_points_before`

This distinction is important because race points and championship standings are not always the same, especially when sprint events exist.

## Leakage Control

The project avoids obvious data leakage by:

- using chronological splits only;
- computing rolling features with `shift(1)`;
- excluding raw outcome columns from model features;
- separating pre-qualifying features from post-qualifying features;
- never inventing future qualifying, grid, results, or points.

Before a future race, the model can only use information available before that race.

## Model Families

### Evaluation Baseline

The official benchmark is kept frozen:

```text
train: 2014-2023
validation: 2024
test: 2025
```

This is the main benchmark used to judge model quality.

### Pre-Qualifying Models

Used before qualifying results or starting grid are available.

They do **not** use:

- qualifying position;
- grid position;
- grid/qualifying deltas;
- teammate qualifying/grid features.

These predictions are more uncertain but realistic before qualifying.

### Post-Qualifying Models

Used only when real qualifying and grid data are available.

They can use:

- qualifying position;
- grid position;
- teammate qualifying/grid comparisons;
- grid vs qualifying deltas.

## Current Official Test Results

The 2025 test set is held out from training in the official evaluation setup.

### Top 10

| Metric | Value |
|---|---:|
| Accuracy | 0.7620 |
| Precision | 0.7561 |
| Recall | 0.7750 |
| F1 | 0.7654 |
| ROC-AUC | 0.8346 |
| Average Precision | 0.8206 |
| Brier Score | 0.1659 |

### Podium

| Metric | Value |
|---|---:|
| Accuracy | 0.8914 |
| Precision | 0.5909 |
| Recall | 0.9028 |
| F1 | 0.7143 |
| ROC-AUC | 0.9426 |
| Average Precision | 0.7342 |
| Brier Score | 0.0809 |

### Points

| Metric | Value |
|---|---:|
| MAE | 2.9193 |
| RMSE | 4.5000 |
| R2 | 0.6068 |
| Spearman | 0.7040 |

These results are reasonable for a first professional version, but they should not be interpreted as perfect race prediction. Formula 1 outcomes depend on many factors not included in this V1, such as weather, tire strategy, safety cars, incidents, upgrades, and free practice pace.

## Repository Structure

```text
F1_Predictor/
├── app/                         # lightweight prediction app/service layer
├── notebooks/
│   └── F1_Predictor_End_to_End_Portfolio_Notebook.ipynb
├── scripts/
│   ├── fetch_jolpica_data.py
│   ├── build_dataset.py
│   ├── train_models.py
│   ├── train_live_models.py
│   ├── build_future_2026_inputs.py
│   ├── predict_selected_race.py
│   ├── tune_models.py
│   ├── audit_project.py
│   └── save_model_results.py
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── future_inputs.py
│   ├── modeling.py
│   ├── models.py
│   ├── evaluation.py
│   └── ...
├── tests/
├── reports/
│   ├── model_audit.md
│   ├── model_comparison.csv
│   ├── data_quality/
│   └── metrics/
├── config.yaml
├── requirements.txt
└── README.md
```

## Reproducing the Pipeline

Create an environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Fetch recent Jolpica snapshots:

```powershell
python scripts/fetch_jolpica_data.py --seasons 2025 2026
```

Build the modeling dataset:

```powershell
python scripts/build_dataset.py
```

Run the audit:

```powershell
python scripts/audit_project.py
```

Train the official baseline models:

```powershell
python scripts/train_models.py
```

Train the pre-qualifying and post-qualifying live model families:

```powershell
python scripts/train_live_models.py
```

Build future 2026 race inputs:

```powershell
python scripts/build_future_2026_inputs.py --as-of-date 2026-07-12
```

Predict a selected future race:

```powershell
python scripts/predict_selected_race.py --round 10 --as-of-date 2026-07-12
```

## Lightweight App

A small Streamlit interface exists for local exploration, but the main focus of this repository is the data science pipeline and notebook.

```powershell
streamlit run app/streamlit_app.py
```

## Tests

Run:

```powershell
python -m pytest tests
```

The tests check prediction contract basics:

- exactly ten drivers in the predicted top 10;
- no duplicate drivers;
- probabilities between 0 and 1;
- non-negative expected points;
- ranking score sorted descending;
- pre-qualifying schema excludes qualifying/grid features.

## What Is Not Included Yet

This V1 does not include:

- weather data;
- tire strategy;
- telemetry;
- free practice pace;
- safety car probability;
- detailed team upgrade modeling;
- complete sprint point reconstruction when official sprint feeds are missing.

Those are natural next steps, but they are intentionally not hidden or simulated in the current model.

## Honest Interpretation

This project is defensible as a portfolio project because it demonstrates:

- clean data engineering;
- temporal machine learning evaluation;
- feature leakage awareness;
- reproducible scripts;
- model comparison;
- future prediction input construction;
- clear documentation of limitations.

It should be presented as a robust **V1 Formula 1 race prediction pipeline**, not as a perfect predictor of Formula 1 results.
