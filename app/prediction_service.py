from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.future_inputs import build_future_2026_inputs


IDENTITY = ["season", "round", "raceName", "date", "driver_name", "constructor_name"]


def _load_json(path, default):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _model_paths(cfg, mode):
    live = cfg["resolved_paths"]["models_dir"] / "live"
    if (live / f"{mode}_top10_model.pkl").exists():
        return {
            "dir": live,
            "top10": live / f"{mode}_top10_model.pkl",
            "podium": live / f"{mode}_podium_model.pkl",
            "points": live / f"{mode}_points_model.pkl",
            "ranker": live / f"{mode}_ranker.pkl",
            "features": live / f"{mode}_feature_columns.json",
            "preprocessor": live / f"{mode}_preprocessing_pipeline.pkl",
            "thresholds": live / f"{mode}_thresholds.json",
            "version": cfg.get("prediction", {}).get("model_version", "live"),
        }
    root = cfg["resolved_paths"]["models_dir"]
    return {
        "dir": root,
        "top10": root / "top10_model.pkl",
        "podium": root / "podium_model.pkl",
        "points": root / "points_model.pkl",
        "ranker": None,
        "features": root / "feature_columns.json",
        "preprocessor": root / "preprocessing_pipeline.pkl",
        "thresholds": None,
        "version": "evaluation_baseline_fallback",
    }


def _confidence(row, mode):
    base = 0.55 if mode == "pre_quali" else 0.75
    if row.get("proba_top10", 0) > 0.75 or row.get("proba_top10", 0) < 0.25:
        base += 0.1
    if pd.isna(row.get("race_rank_score")):
        base -= 0.1
    return "High" if base >= 0.8 else "Medium" if base >= 0.6 else "Low"


def predict_race(round_number, as_of_date=None, cfg=None):
    cfg = cfg or load_config()
    top_n = int(cfg.get("prediction", {}).get("top_n", 10))
    future = build_future_2026_inputs(cfg, as_of_date=as_of_date, round_number=round_number)
    if future.empty:
        raise ValueError(f"No future inputs available for round {round_number}.")
    race = future[future["round"].eq(int(round_number))].copy()
    if race.empty:
        raise ValueError(f"Round {round_number} is not available in future inputs.")

    has_quali = race[["grid_position", "quali_position"]].notna().all(axis=None)
    mode = "post_quali" if has_quali and cfg.get("prediction", {}).get("auto_detect_prediction_mode", True) else "pre_quali"
    paths = _model_paths(cfg, mode)
    features = _load_json(paths["features"], [])
    missing_features = [c for c in features if c not in race.columns]
    if missing_features:
        for col in missing_features:
            race[col] = np.nan

    preprocessor = joblib.load(paths["preprocessor"])
    models = {name: joblib.load(paths[name]) for name in ["top10", "podium", "points"]}
    ranker = joblib.load(paths["ranker"]) if paths.get("ranker") and Path(paths["ranker"]).exists() else None
    thresholds = _load_json(paths["thresholds"], {"top10": 0.5, "podium": 0.5}) if paths.get("thresholds") else {"top10": 0.5, "podium": 0.5}

    X = preprocessor.transform(race[features])
    out = race[[c for c in IDENTITY if c in race.columns]].copy()
    out["proba_top10"] = models["top10"].predict_proba(X)[:, 1]
    out["proba_podium"] = models["podium"].predict_proba(X)[:, 1]
    out["expected_points"] = models["points"].predict(X).clip(min=0)
    out["race_rank_score"] = ranker.predict(X) if ranker is not None else (
        out["expected_points"] * 1.0 + out["proba_podium"] * 3.0 + out["proba_top10"]
    )
    out["predicted_top10"] = out["proba_top10"].ge(float(thresholds.get("top10", 0.5))).astype(int)
    out["predicted_podium"] = out["proba_podium"].ge(float(thresholds.get("podium", 0.5))).astype(int)
    out = out.sort_values(["race_rank_score", "expected_points", "proba_podium", "proba_top10"], ascending=False)
    out = out.drop_duplicates("driver_name").head(top_n).copy()
    out.insert(0, "predicted_rank", np.arange(1, len(out) + 1))
    out["confidence"] = out.apply(lambda r: _confidence(r, mode), axis=1)

    metadata = {
        "data_cutoff": as_of_date or cfg.get("prediction", {}).get("as_of_date"),
        "model_version": paths["version"],
        "prediction_mode": mode.upper().replace("_", "-"),
        "missing_features": missing_features,
        "confidence_level": out["confidence"].mode().iloc[0] if not out.empty else "Low",
        "round": int(round_number),
        "raceName": race["raceName"].iloc[0],
        "date": race["date"].iloc[0],
        "sprint_weekend": bool(race.get("sprint_weekend_flag", pd.Series([0])).iloc[0]),
        "drivers_available": int(len(race)),
    }
    return out, metadata
