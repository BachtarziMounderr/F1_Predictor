from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import ndcg_score

from src.config import load_config, ensure_directories
from src.evaluation import classification_metrics, regression_metrics
from src.modeling import select_feature_columns
from src.models import train_top10_model, train_podium_model, train_points_model
from src.utils import save_json

try:
    from xgboost import XGBRanker
except ImportError:
    XGBRanker = None


def _copy_existing_baseline(cfg):
    src = cfg["resolved_paths"]["models_dir"]
    dst = src / "evaluation"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["top10_model.pkl", "podium_model.pkl", "points_model.pkl", "feature_columns.json", "preprocessing_pipeline.pkl"]:
        p = src / name
        if p.exists():
            (dst / name).write_bytes(p.read_bytes())


def _train_parts(df, cfg):
    as_of = pd.to_datetime(cfg["live_training"]["as_of_date"])
    dates = pd.to_datetime(df["date"], errors="coerce")
    live_mask = df["season"].eq(2026) & dates.le(as_of)
    eval_train = df[df["season"].between(2014, 2023)].copy()
    validation = df[df["season"].eq(2024)].copy()
    test = df[df["season"].eq(2025)].copy()
    live_train = df[df["season"].between(2014, 2025) | live_mask].copy()
    return {"eval_train": eval_train, "validation": validation, "test": test, "live_train": live_train}


def _weights(frame, cfg):
    w = np.ones(len(frame))
    if not cfg["live_training"].get("recency_weighting", True):
        return w
    w = np.where(frame["season"].le(2021), 0.5, w)
    w = np.where(frame["season"].between(2022, 2023), 0.8, w)
    w = np.where(frame["season"].between(2024, 2025), 1.0, w)
    w = np.where(frame["season"].eq(2026), float(cfg["live_training"].get("regulation_boost_2026", 2.0)), w)
    return w


def _threshold(y, proba, metric="f1"):
    from sklearn.metrics import f1_score
    candidates = np.linspace(0.05, 0.95, 91)
    scores = [(t, f1_score(y, proba >= t, zero_division=0)) for t in candidates]
    return float(max(scores, key=lambda x: x[1])[0])


def _ranking_metrics(frame, scores):
    rows = []
    tmp = frame[["season", "round", "target_top10", "positionOrder"]].copy()
    tmp["score"] = scores
    for _, race in tmp.groupby(["season", "round"], sort=False):
        if len(race) < 10:
            continue
        ranked = race.sort_values("score", ascending=False)
        true_top10 = set(race.sort_values("positionOrder").head(10).index)
        pred_top10 = set(ranked.head(10).index)
        rows.append({
            "precision_at_10": len(true_top10 & pred_top10) / 10,
            "recall_at_10": len(true_top10 & pred_top10) / max(1, len(true_top10)),
            "top10_overlap": len(true_top10 & pred_top10),
            "ndcg_at_10": ndcg_score([race["target_top10"].to_numpy()], [race["score"].to_numpy()], k=10),
        })
    return pd.DataFrame(rows).mean(numeric_only=True).to_dict() if rows else {}


def _fit_ranker(X, y, groups, cfg):
    if XGBRanker is None:
        return None
    params = dict(
        objective=cfg.get("ranking", {}).get("objective", "rank:pairwise"),
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=cfg["project"]["random_state"],
        n_jobs=cfg.get("tuning", {}).get("n_jobs", -1),
    )
    ranker = XGBRanker(**params)
    ranker.fit(X, y, group=groups)
    return ranker


def _train_family(df, parts, cfg, mode, model_dir, train_key="live_train", write_artifacts=True):
    features = select_feature_columns(df, mode=mode)
    train = parts[train_key].dropna(subset=["target_top10", "target_podium", "target_points"]).copy()
    validation = parts["validation"].copy()
    test = parts["test"].copy()
    imputer = SimpleImputer(strategy="median").fit(train[features])
    X_train = imputer.transform(train[features])
    weights = _weights(train, cfg)
    models = {
        "top10": train_top10_model(X_train, train["target_top10"], None, None, cfg, weights),
        "podium": train_podium_model(X_train, train["target_podium"], None, None, cfg, weights),
        "points": train_points_model(X_train, train["target_points"], None, None, cfg, weights),
    }
    train_sorted = train.sort_values(["season", "round"]).copy()
    group_sizes = train_sorted.groupby(["season", "round"]).size().to_list()
    X_rank = imputer.transform(train_sorted[features])
    relevance = (30 - train_sorted["positionOrder"].fillna(30)).clip(lower=0)
    ranker = _fit_ranker(X_rank, relevance, group_sizes, cfg)
    if ranker is not None:
        models["ranker"] = ranker

    metrics = {}
    thresholds = {}
    for split_name, frame in [("validation", validation), ("test", test)]:
        X = imputer.transform(frame[features])
        metrics[split_name] = {}
        for name, target in [("top10", "target_top10"), ("podium", "target_podium")]:
            proba = models[name].predict_proba(X)[:, 1]
            if split_name == "validation":
                thresholds[name] = _threshold(frame[target], proba)
            pred = proba >= thresholds.get(name, 0.5)
            metrics[split_name][name] = classification_metrics(frame[target], pred, proba)
        points = models["points"].predict(X).clip(min=0)
        metrics[split_name]["points"] = regression_metrics(frame["target_points"], points)
        rank_scores = models["ranker"].predict(X) if "ranker" in models else points
        metrics[split_name]["ranking"] = _ranking_metrics(frame, rank_scores)

    if write_artifacts:
        model_dir.mkdir(parents=True, exist_ok=True)
        for name, model in models.items():
            joblib.dump(model, model_dir / f"{mode}_{name}_model.pkl" if name != "ranker" else model_dir / f"{mode}_ranker.pkl")
        joblib.dump(imputer, model_dir / f"{mode}_preprocessing_pipeline.pkl")
        (model_dir / f"{mode}_feature_columns.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
        (model_dir / f"{mode}_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    return metrics


def main():
    cfg = load_config()
    ensure_directories(cfg)
    _copy_existing_baseline(cfg)
    df = pd.read_csv(cfg["resolved_paths"]["processed_data_dir"] / "f1_driver_race_dataset_with_jolpica.csv")
    parts = _train_parts(df, cfg)
    live_dir = cfg["resolved_paths"]["models_dir"] / "live"
    eval_dir = cfg["resolved_paths"]["models_dir"] / "evaluation"
    report_dir = cfg["resolved_paths"]["reports_dir"] / "metrics"
    all_metrics = {
        "pre_quali": _train_family(df, parts, cfg, "pre_quali", eval_dir, train_key="eval_train", write_artifacts=True),
        "post_quali": _train_family(df, parts, cfg, "post_quali", eval_dir, train_key="eval_train", write_artifacts=True),
    }
    _train_family(df, parts, cfg, "pre_quali", live_dir, train_key="live_train", write_artifacts=True)
    _train_family(df, parts, cfg, "post_quali", live_dir, train_key="live_train", write_artifacts=True)
    save_json(all_metrics["pre_quali"], report_dir / "pre_quali_metrics.json")
    save_json(all_metrics["post_quali"], report_dir / "post_quali_metrics.json")
    ranking = {"pre_quali": all_metrics["pre_quali"].get("test", {}).get("ranking", {}),
               "post_quali": all_metrics["post_quali"].get("test", {}).get("ranking", {})}
    save_json(ranking, report_dir / "ranking_metrics.json")
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
