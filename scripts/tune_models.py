from pathlib import Path
import itertools
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.config import load_config, ensure_directories
from src.evaluation import classification_metrics, regression_metrics
from src.modeling import select_feature_columns
from src.models import HAS_XGB
from src.utils import save_json

if HAS_XGB:
    from xgboost import XGBClassifier, XGBRegressor
else:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def _folds():
    return [(2019, 2020), (2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024)]


def _candidate_params():
    grid = {
        "n_estimators": [300, 500],
        "learning_rate": [0.03, 0.06],
        "max_depth": [3, 4],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*[grid[k] for k in keys])]


def _make_model(task, params, cfg):
    params = {**params, "random_state": cfg["project"]["random_state"], "n_jobs": cfg.get("tuning", {}).get("n_jobs", -1)}
    if HAS_XGB:
        if task in ("top10", "podium"):
            params["eval_metric"] = "logloss"
            return XGBClassifier(**params)
        params["objective"] = "reg:squarederror"
        return XGBRegressor(**params)
    if task in ("top10", "podium"):
        return RandomForestClassifier(n_estimators=300, random_state=cfg["project"]["random_state"], n_jobs=-1, class_weight="balanced")
    return RandomForestRegressor(n_estimators=300, random_state=cfg["project"]["random_state"], n_jobs=-1)


def main():
    cfg = load_config()
    ensure_directories(cfg)
    df = pd.read_csv(cfg["resolved_paths"]["processed_data_dir"] / "f1_driver_race_dataset_with_jolpica.csv")
    features = select_feature_columns(df, "post_quali")
    trials = []
    for task in ["top10", "podium", "points"]:
        target = {"top10": "target_top10", "podium": "target_podium", "points": "target_points"}[task]
        for params in _candidate_params():
            fold_scores = []
            for train_until, val_year in _folds():
                train = df[df["season"].between(2014, train_until)].copy()
                val = df[df["season"].eq(val_year)].copy()
                imputer = SimpleImputer(strategy="median").fit(train[features])
                model = _make_model(task, params, cfg)
                model.fit(imputer.transform(train[features]), train[target])
                Xv = imputer.transform(val[features])
                if task in ("top10", "podium"):
                    proba = model.predict_proba(Xv)[:, 1]
                    pred = proba >= 0.5
                    m = classification_metrics(val[target], pred, proba)
                    score = m["average_precision"] if task == "podium" else m["roc_auc"]
                else:
                    pred = np.clip(model.predict(Xv), 0, None)
                    m = regression_metrics(val[target], pred)
                    score = -m["mae"]
                fold_scores.append(float(score))
            trials.append({"task": task, "params": params, "mean_score": float(np.mean(fold_scores)), "fold_scores": fold_scores})
    out_json = cfg["resolved_paths"]["reports_dir"] / "metrics" / "temporal_tuning_trials.json"
    save_json(trials, out_json)
    best = pd.DataFrame(trials).sort_values(["task", "mean_score"], ascending=[True, False]).groupby("task").head(1)
    best.to_csv(cfg["resolved_paths"]["reports_dir"] / "model_tuning_summary.csv", index=False)
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
