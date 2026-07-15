from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from src.config import load_config, ensure_directories
from src.modeling import select_feature_columns


def _json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _standings_reconciliation(df, cfg):
    outdir = cfg["resolved_paths"]["reports_dir"] / "data_quality"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    if "event_points" not in df:
        df["event_points"] = df.get("points", 0)
    for season, frame in df.groupby("season"):
        recon = frame.groupby("driver_name")["event_points"].sum().reset_index(name="reconstructed_event_points")
        standings_path = cfg["resolved_paths"]["external_data_dir"] / "jolpica" / f"jolpica_{int(season)}_driver_standings.csv"
        if standings_path.exists():
            st = pd.read_csv(standings_path)
            if not st.empty and "Driver.givenName" in st and "Driver.familyName" in st:
                st["driver_name"] = (st["Driver.givenName"].fillna("") + " " + st["Driver.familyName"].fillna("")).str.strip()
                st["official_points"] = pd.to_numeric(st["points"], errors="coerce")
                merged = recon.merge(st[["driver_name", "official_points"]], on="driver_name", how="outer")
            else:
                merged = recon.copy()
                merged["official_points"] = pd.NA
        else:
            merged = recon.copy()
            merged["official_points"] = pd.NA
        merged["season"] = season
        merged["difference"] = merged["reconstructed_event_points"] - pd.to_numeric(merged["official_points"], errors="coerce")
        rows.append(merged)
    result = pd.concat(rows, ignore_index=True, sort=False)
    path = outdir / "standings_reconciliation.csv"
    result.to_csv(path, index=False)
    return path, result


def main():
    cfg = load_config()
    ensure_directories(cfg)
    reports = cfg["resolved_paths"]["reports_dir"]
    processed = cfg["resolved_paths"]["processed_data_dir"]
    models_dir = cfg["resolved_paths"]["models_dir"]
    dataset = processed / "f1_driver_race_dataset_with_jolpica.csv"
    df = pd.read_csv(dataset)
    if "grand_prix_points" not in df:
        df["grand_prix_points"] = df["points"]
        df["sprint_points"] = 0.0
        df["event_points"] = df["grand_prix_points"].fillna(0) + df["sprint_points"].fillna(0)
    recon_path, recon = _standings_reconciliation(df.copy(), cfg)

    metrics = {
        "top10": _json(reports / "metrics" / "top10_metrics.json"),
        "podium": _json(reports / "metrics" / "podium_metrics.json"),
        "points": _json(reports / "metrics" / "points_metrics.json"),
        "baseline": _json(reports / "metrics" / "baseline_metrics.json"),
        "pre_quali": _json(reports / "metrics" / "pre_quali_metrics.json"),
        "post_quali": _json(reports / "metrics" / "post_quali_metrics.json"),
        "ranking": _json(reports / "metrics" / "ranking_metrics.json"),
    }
    feature_cols = _json(models_dir / "feature_columns.json")
    model_types = {}
    for name in ["top10", "podium", "points"]:
        path = models_dir / f"{name}_model.pkl"
        if path.exists():
            model = joblib.load(path)
            model_types[name] = str(type(model))
    comparison_rows = []
    for task in ["top10", "podium", "points"]:
        test = metrics.get(task, {}).get("test", {})
        row = {"task": task, "model_family": "current_evaluation_baseline", **{k: v for k, v in test.items() if k != "confusion_matrix"}}
        comparison_rows.append(row)
    for family in ["pre_quali", "post_quali"]:
        test = metrics.get(family, {}).get("test", {})
        for task in ["top10", "podium", "points", "ranking"]:
            values = test.get(task, {})
            if values:
                comparison_rows.append({"task": task, "model_family": family, **{k: v for k, v in values.items() if k != "confusion_matrix"}})
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(reports / "model_comparison.csv", index=False)

    seasons = df.groupby("season").size().to_dict()
    duplicates = int(df.duplicated(["season", "round", "driverId"]).sum())
    pre_features = select_feature_columns(df, "pre_quali")
    post_features = select_feature_columns(df, "post_quali")
    recon_issues = recon[pd.to_numeric(recon["difference"], errors="coerce").abs().fillna(0) > 0.01]

    audit = f"""# F1 Predictor model audit

Generated from local artifacts.

## Data status

- Dataset: `{dataset}`
- Shape: {df.shape[0]} rows x {df.shape[1]} columns
- Seasons: {sorted(df['season'].dropna().astype(int).unique().tolist())}
- Rows by season: `{seasons}`
- Duplicate season/round/driverId rows: {duplicates}
- 2026 completed rounds in dataset: {sorted(df.loc[df.season.eq(2026), 'round'].dropna().astype(int).unique().tolist())}
- `target_points` currently represents Grand Prix race points. Sprint/championship points are separated into `sprint_points` and `event_points` when rebuilt.

## Model status

- Current model types: `{model_types}`
- Current saved feature columns: {len(feature_cols)}
- Pre-qualifying feature count after filtering: {len(pre_features)}
- Post-qualifying feature count after filtering: {len(post_features)}

## Hyperparameters

```json
{json.dumps(cfg['models'], indent=2)}
```

## Current official test metrics

```json
{json.dumps({k: v.get('test', {}) for k, v in metrics.items() if k != 'baseline'}, indent=2)}
```

## Temporal tuning

- Tuning trials: `{reports / "metrics" / "temporal_tuning_trials.json"}`
- Tuning summary: `{reports / "model_tuning_summary.csv"}`
- Tuning is comparative only; it does not replace the selected production artifacts automatically.

## Baselines

`reports/metrics/baseline_metrics.json` is present: {(reports / 'metrics' / 'baseline_metrics.json').exists()}.

## Standings reconciliation

- File: `{recon_path}`
- Rows with non-zero difference: {len(recon_issues)}
- Known limitation: historical and current snapshots may not include full sprint/event points unless sprint result feeds are integrated.

## Leakage and schema risks

- Rolling features use `shift(1)` in `src/features.py`.
- Raw outcomes are excluded in `src/modeling.py`.
- Risk: current model features include season/round, which can encode time/calendar effects.
- Risk: original `points` semantics mixed race points with championship interpretation; this audit separates `grand_prix_points`, `sprint_points`, and `event_points`.
- Risk: 2026 future races need schedule-only driver-race rows; no future grid, qualifying, result, or points should be invented.

## Recommendations

1. Keep the current 2025 test metrics frozen as the evaluation baseline.
2. Use pre-quali models before qualifying and post-quali models only when real qualifying/grid data exists.
3. Integrate sprint result snapshots if available from Jolpica to improve official standings reconciliation.
4. Treat context/news adjustments as optional auditable overlays, not hidden model inputs.
5. Run temporal tuning before replacing any baseline model.
"""
    (reports / "model_audit.md").write_text(audit, encoding="utf-8")
    print(reports / "model_audit.md")
    print(reports / "model_comparison.csv")
    print(recon_path)


if __name__ == "__main__":
    main()
