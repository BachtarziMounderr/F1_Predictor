from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.future_inputs import build_future_2026_inputs


@st.cache_data(show_spinner=False)
def get_config():
    return load_config()


@st.cache_data(show_spinner=False)
def load_csv(path):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def future_races(as_of_date):
    cfg = load_config()
    future = build_future_2026_inputs(cfg, as_of_date=as_of_date)
    if future.empty:
        return pd.DataFrame()
    cols = ["round", "raceName", "date", "sprint_weekend_flag", "prediction_mode"]
    return future[cols].drop_duplicates().sort_values("round")


def metrics_tables():
    cfg = load_config()
    metrics = cfg["resolved_paths"]["reports_dir"] / "metrics"
    return {
        "top10": load_csv(metrics / "top10_metrics.csv"),
        "comparison": load_csv(cfg["resolved_paths"]["reports_dir"] / "model_comparison.csv"),
        "reconciliation": load_csv(cfg["resolved_paths"]["reports_dir"] / "data_quality" / "standings_reconciliation.csv"),
    }
