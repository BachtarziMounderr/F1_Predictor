from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.prediction_service import predict_race
from src.config import load_config
from src.modeling import select_feature_columns, QUALI_FEATURES
import pandas as pd


def test_pre_quali_schema_excludes_qualifying_features():
    cfg = load_config()
    df = pd.read_csv(cfg["resolved_paths"]["processed_data_dir"] / "f1_driver_race_dataset_with_jolpica.csv")
    features = set(select_feature_columns(df, "pre_quali"))
    assert not features & QUALI_FEATURES


def test_predict_round_10_top10_contract():
    cfg = load_config()
    predictions, meta = predict_race(10, cfg["prediction"]["as_of_date"], cfg)
    assert len(predictions) == cfg["prediction"]["top_n"]
    assert predictions["driver_name"].is_unique
    assert predictions["proba_top10"].between(0, 1).all()
    assert predictions["proba_podium"].between(0, 1).all()
    assert predictions["expected_points"].ge(0).all()
    assert predictions["race_rank_score"].is_monotonic_decreasing
    assert meta["prediction_mode"] in {"PRE-QUALI", "POST-QUALI"}
