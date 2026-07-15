from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.components import inject_css, hero, race_summary, prediction_table, context_block
from app.data_service import get_config, future_races, load_csv
from app.prediction_service import predict_race
from src.future_inputs import build_future_2026_inputs


st.set_page_config(page_title="F1 2026 Race Predictor", page_icon="🏁", layout="wide")
inject_css()

cfg = get_config()
page = st.sidebar.radio("Navigation", ["Race Predictor", "Model Performance", "Data Status"])
as_of_date = st.sidebar.text_input("Data cutoff", cfg.get("prediction", {}).get("as_of_date", "2026-07-12"))

if page == "Race Predictor":
    hero("F1 2026 Race Predictor", "Machine-learning predictions based on data available before the selected race.")
    races = future_races(as_of_date)
    if races.empty:
        st.error("No future races available for the selected cutoff. Build or refresh Jolpica snapshots first.")
        st.stop()
    race_labels = {
        f"Round {int(r.round)} · {r.raceName} · {r.date}": int(r.round)
        for _, r in races.iterrows()
    }
    selected = st.selectbox("Select next race", list(race_labels.keys()))
    round_number = race_labels[selected]
    race_row = races[races["round"].eq(round_number)].iloc[0]
    st.markdown(
        f"<span class='soft-badge'>Circuit date: {race_row.date}</span>"
        f"<span class='soft-badge'>Round {round_number}</span>"
        f"<span class='soft-badge'>{'Sprint' if race_row.sprint_weekend_flag else 'Standard'} weekend</span>",
        unsafe_allow_html=True,
    )

    if st.button("Lancer la prédiction", type="primary"):
        with st.spinner("Generating race prediction..."):
            predictions, meta = predict_race(round_number, as_of_date, cfg)
            out_path = cfg["resolved_paths"]["predictions_dir"] / "2026_next_races_predictions.csv"
            predictions.to_csv(out_path, index=False)
        race_summary(meta)
        st.subheader("TOP 10 PRÉDIT")
        prediction_table(predictions)
        st.download_button(
            "Download CSV",
            predictions.to_csv(index=False).encode("utf-8"),
            file_name=f"round_{round_number}_prediction.csv",
            mime="text/csv",
        )
        st.subheader("Prediction context")
        future_inputs = build_future_2026_inputs(cfg, as_of_date=as_of_date, round_number=round_number)
        context_cols = [
            "driver_name", "constructor_name", "driver_recent_points_last_3_2026",
            "constructor_recent_points_last_3_2026", "driver_top10_rate_2026_before",
            "constructor_top10_rate_2026_before", "driver_official_standings_points_before",
            "constructor_official_standings_points_before", "missing_qualifying",
        ]
        context_block(meta, future_inputs[[c for c in context_cols if c in future_inputs.columns]])
        st.subheader("Why this prediction?")
        st.info("SHAP is not computed in this V1 dashboard. Use feature importance and model metrics on the Model Performance page; no hidden manual context bonus is applied.")

elif page == "Model Performance":
    hero("Model Performance", "Evaluation artifacts, baselines and live model diagnostics.")
    metrics_dir = cfg["resolved_paths"]["reports_dir"] / "metrics"
    for name in ["top10_metrics.json", "podium_metrics.json", "points_metrics.json", "pre_quali_metrics.json", "post_quali_metrics.json", "ranking_metrics.json"]:
        path = metrics_dir / name
        if path.exists():
            st.subheader(name)
            st.json(path.read_text(encoding="utf-8"))
    figures_dir = cfg["resolved_paths"]["reports_dir"] / "figures"
    fig_cols = st.columns(2)
    for i, fig in enumerate(sorted(figures_dir.glob("*.png"))):
        fig_cols[i % 2].image(str(fig), caption=fig.name, use_container_width=True)

else:
    hero("Data Status", "Local data freshness, future inputs and reconciliation checks.")
    processed = cfg["resolved_paths"]["processed_data_dir"]
    df = load_csv(processed / "f1_driver_race_dataset_with_jolpica.csv")
    if not df.empty:
        st.subheader("Seasons available")
        st.dataframe(df.groupby("season").size().reset_index(name="rows"), use_container_width=True, hide_index=True)
        st.subheader("2026 completed races in local dataset")
        st.dataframe(df[df["season"].eq(2026)][["round", "raceName", "date"]].drop_duplicates().sort_values("round"), use_container_width=True, hide_index=True)
    st.subheader("Future races")
    st.dataframe(future_races(as_of_date), use_container_width=True, hide_index=True)
    rec = load_csv(cfg["resolved_paths"]["reports_dir"] / "data_quality" / "standings_reconciliation.csv")
    if not rec.empty:
        st.subheader("Standings reconciliation")
        st.dataframe(rec, use_container_width=True, hide_index=True)
