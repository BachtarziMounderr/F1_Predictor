from pathlib import Path

import numpy as np
import pandas as pd

from .features import add_pre_race_features


OUTCOME_COLUMNS = [
    "position", "positionOrder", "points", "grand_prix_points", "sprint_points",
    "event_points", "laps", "milliseconds", "fastestLap", "rank",
    "fastestLapSpeed", "statusId", "status", "positionText", "time",
    "fastestLapTime", "target_top10", "target_podium",
    "target_points", "is_dnf",
]


def _read(path):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _load_mapping(external_dir, entity):
    path = Path(external_dir) / f"id_mapping_{entity}.csv"
    return _read(path)


def _internal_lookup(mapping, jolpica_col="jolpica_id"):
    if mapping.empty:
        return {}, {}
    return (
        mapping.set_index(jolpica_col)["internal_id"].to_dict(),
        mapping.set_index(jolpica_col)["internal_ref"].to_dict(),
    )


def _completed_history(df, as_of_date, selected_round=None, season=2026):
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    as_of = pd.to_datetime(as_of_date)
    if selected_round is not None:
        return _strip_engineered_columns(data[(data["season"].lt(season)) | ((data["season"].eq(season)) & (data["round"].lt(selected_round)))].copy())
    return _strip_engineered_columns(data[data["date"].le(as_of)].copy())


def _strip_engineered_columns(data):
    keep_prefixed = {
        "driverId", "driverRef", "driver_name",
        "constructorId", "constructorRef", "constructor_name",
        "circuitId", "circuitRef",
    }
    drop = []
    engineered_prefixes = (
        "driver_", "constructor_", "teammate_", "circuit_", "races_completed_",
        "quali_position_vs_", "grid_position_vs_",
    )
    engineered_exact = {
        "grid_vs_quali_delta", "qualified_top3", "qualified_top10", "started_top3",
        "started_top10", "race_round", "is_2026_regulation", "is_dnf",
        "missing_driver_history", "missing_constructor_history", "missing_circuit_history",
        "driver_official_standings_points_before", "constructor_official_standings_points_before",
    }
    for col in data.columns:
        if col in keep_prefixed:
            continue
        if col in engineered_exact or col.startswith(engineered_prefixes):
            drop.append(col)
    return data.drop(columns=drop, errors="ignore")


def _future_schedule(schedule, as_of_date, selected_round=None):
    if schedule.empty:
        raise FileNotFoundError("Missing Jolpica 2026 schedule snapshot.")
    sched = schedule.copy()
    sched["date"] = pd.to_datetime(sched["date"], errors="coerce")
    sched["season"] = pd.to_numeric(sched.get("season", 2026), errors="coerce").fillna(2026).astype(int)
    sched["round"] = pd.to_numeric(sched["round"], errors="coerce").astype(int)
    if selected_round is not None:
        sched = sched[sched["round"].eq(int(selected_round))]
    else:
        sched = sched[sched["date"].gt(pd.to_datetime(as_of_date))]
    return sched.sort_values("round").copy()


def _future_quali(jolpica_dir, round_number):
    q = _read(Path(jolpica_dir) / "jolpica_2026_qualifying.csv")
    if q.empty:
        return pd.DataFrame()
    q["round"] = pd.to_numeric(q["round"], errors="coerce")
    return q[q["round"].eq(int(round_number))].copy()


def _make_round_rows(history, race, cfg, external_dir, jolpica_dir):
    season = int(race.get("season", 2026))
    round_number = int(race["round"])
    last_round = history[history["season"].eq(2026)]["round"].max()
    if pd.isna(last_round):
        entrants = history.sort_values(["season", "round"]).groupby("driverId", as_index=False).tail(1)
    else:
        entrants = history[(history["season"].eq(2026)) & (history["round"].eq(last_round))].copy()
    if entrants.empty:
        raise ValueError("Cannot infer future entrants: no completed 2026 driver rows are available.")

    circuit_map, circuit_ref = _internal_lookup(_load_mapping(external_dir, "circuits"))
    circuit_key = race.get("Circuit.circuitId")
    circuit_id = circuit_map.get(circuit_key, entrants["circuitId"].iloc[0])
    circuit_name = circuit_ref.get(circuit_key, circuit_key)

    quali = _future_quali(jolpica_dir, round_number)
    quali_map = {}
    if not quali.empty and "Driver.driverId" in quali and "position" in quali:
        quali_map = dict(zip(quali["Driver.driverId"], pd.to_numeric(quali["position"], errors="coerce")))

    rows = entrants.copy()
    rows["season"] = season
    rows["round"] = round_number
    rows["raceName"] = race.get("raceName")
    rows["date"] = race.get("date")
    rows["circuitId"] = circuit_id
    rows["circuitRef"] = circuit_name
    rows["raceId"] = 2026000 + round_number
    rows["resultId"] = np.nan
    rows["grid"] = np.nan
    rows["grid_position"] = np.nan
    rows["quali_position"] = rows["driverRef"].map(quali_map) if quali_map else np.nan
    if quali_map:
        rows["grid"] = rows["quali_position"]
        rows["grid_position"] = rows["quali_position"]
    for col in OUTCOME_COLUMNS:
        if col in rows:
            rows[col] = np.nan
    rows["grand_prix_points"] = np.nan
    rows["sprint_points"] = np.nan
    rows["event_points"] = np.nan
    rows["target_top10"] = np.nan
    rows["target_podium"] = np.nan
    rows["target_points"] = np.nan
    rows["missing_qualifying"] = rows["quali_position"].isna().astype("int8")
    rows["sprint_weekend_flag"] = int(round_number in set(cfg.get("calendar", {}).get("sprint_rounds_2026", [])))
    return rows


def build_future_2026_inputs(cfg, as_of_date=None, round_number=None):
    as_of_date = as_of_date or cfg.get("prediction", {}).get("as_of_date") or cfg.get("live_training", {}).get("as_of_date")
    processed = cfg["resolved_paths"]["processed_data_dir"]
    external = cfg["resolved_paths"]["external_data_dir"]
    jolpica_dir = external / "jolpica"
    dataset = processed / "f1_driver_race_dataset_with_jolpica.csv"
    if not dataset.exists():
        raise FileNotFoundError("Run scripts/build_dataset.py before building future inputs.")
    df = pd.read_csv(dataset)
    schedule = _read(jolpica_dir / "jolpica_2026_schedule.csv")
    future = _future_schedule(schedule, as_of_date, round_number)
    if future.empty:
        return pd.DataFrame()

    output = []
    for _, race in future.iterrows():
        history = _completed_history(df, as_of_date, int(race["round"]), season=2026)
        round_rows = _make_round_rows(history, race, cfg, external, jolpica_dir)
        base_like = pd.concat([history, round_rows], ignore_index=True, sort=False)
        featured = add_pre_race_features(base_like, cfg["features"]["rolling_windows"])
        future_rows = featured[(featured["season"].eq(2026)) & (featured["round"].eq(int(race["round"])))].copy()
        future_rows["data_cutoff_date"] = as_of_date
        has_quali = future_rows[["grid_position", "quali_position"]].notna().all(axis=None)
        future_rows["prediction_mode"] = "post_quali" if has_quali else "pre_quali"
        output.append(future_rows)
    return pd.concat(output, ignore_index=True, sort=False)


def write_future_2026_inputs(cfg, as_of_date=None, round_number=None):
    future = build_future_2026_inputs(cfg, as_of_date, round_number)
    out = cfg["resolved_paths"]["processed_data_dir"] / "2026_future_race_inputs.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    future.to_csv(out, index=False)
    return future, out
