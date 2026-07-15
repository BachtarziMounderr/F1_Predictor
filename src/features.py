import numpy as np
import pandas as pd

def _shifted_rolling(s, group, window, func="mean"):
    shifted = s.groupby(group, sort=False).shift(1)
    return shifted.groupby(group, sort=False).transform(lambda x: x.rolling(window, min_periods=1).agg(func))

def _expanding_before(s, group, func):
    shifted = s.groupby(group, sort=False).shift(1)
    return shifted.groupby(group, sort=False).transform(lambda x: x.expanding().agg(func))

def add_pre_race_features(df, windows=(3,5,10)):
    """Create features strictly from the current grid/quali or prior race outcomes."""
    x = df.sort_values(["date","round","driverId"]).copy()
    if "grand_prix_points" not in x:
        x["grand_prix_points"] = x["points"]
    if "sprint_points" not in x:
        x["sprint_points"] = 0.0
    if "event_points" not in x:
        x["event_points"] = x["grand_prix_points"].fillna(0) + x["sprint_points"].fillna(0)
    if "sprint_weekend_flag" not in x:
        x["sprint_weekend_flag"] = 0
    x["grid_vs_quali_delta"] = x.grid_position - x.quali_position
    for n in (3,10):
        x[f"qualified_top{n}"] = x.quali_position.le(n).astype("int8")
        x[f"started_top{n}"] = x.grid_position.le(n).astype("int8")
    x["race_round"] = x["round"]
    x["is_2026_regulation"] = x.season.ge(2026).astype("int8")
    known_result=x.positionOrder.notna()
    x["is_dnf"] = (~x.get("status", "").fillna("").str.contains("Finished|Lap", case=False, regex=True)).where(known_result).astype("Int8")

    dg = x.driverId
    x["driver_races_before"] = x.groupby("driverId").cumcount()
    for w in windows:
        x[f"driver_avg_finish_last_{w}"] = _shifted_rolling(x.positionOrder, dg, w)
        x[f"driver_points_last_{w}"] = _shifted_rolling(x.grand_prix_points, dg, w, "sum")
        x[f"driver_event_points_last_{w}"] = _shifted_rolling(x.event_points, dg, w, "sum")
        if w in (5,10):
            x[f"driver_top10_rate_last_{w}"] = _shifted_rolling(x.target_top10, dg, w)
            x[f"driver_podium_rate_last_{w}"] = _shifted_rolling(x.target_podium, dg, w)
            x[f"driver_dnf_rate_last_{w}"] = _shifted_rolling(x.is_dnf, dg, w)
    x["driver_avg_grid_last_5"] = _shifted_rolling(x.grid_position, dg, 5)
    x["driver_avg_quali_last_5"] = _shifted_rolling(x.quali_position, dg, 5)

    # Constructor outcomes are aggregated to one row per race before rolling.
    cr = x.groupby(["season","round","date","constructorId"], as_index=False).agg(
        constructor_points=("grand_prix_points","sum"), constructor_event_points=("event_points","sum"), constructor_finish=("positionOrder","mean"),
        constructor_top10=("target_top10","mean"), constructor_podium=("target_podium","mean"),
        constructor_dnf=("is_dnf","mean"), constructor_grid=("grid_position","mean"),
        constructor_quali=("quali_position","mean"))
    cr = cr.sort_values(["date","round","constructorId"]); cg = cr.constructorId
    cr["constructor_races_before"] = cr.groupby("constructorId").cumcount()
    for w in windows:
        cr[f"constructor_avg_finish_last_{w}"] = _shifted_rolling(cr.constructor_finish, cg, w)
        cr[f"constructor_points_last_{w}"] = _shifted_rolling(cr.constructor_points, cg, w, "sum")
        cr[f"constructor_event_points_last_{w}"] = _shifted_rolling(cr.constructor_event_points, cg, w, "sum")
        if w in (5,10):
            for src, name in [("constructor_top10","top10_rate"),("constructor_podium","podium_rate"),("constructor_dnf","dnf_rate")]:
                cr[f"constructor_{name}_last_{w}"] = _shifted_rolling(cr[src], cg, w)
    cr["constructor_avg_grid_last_5"] = _shifted_rolling(cr.constructor_grid, cg, 5)
    cr["constructor_avg_quali_last_5"] = _shifted_rolling(cr.constructor_quali, cg, 5)
    # Never merge current-race constructor aggregates back: only shifted features.
    current_aggregates={"constructor_points","constructor_event_points","constructor_finish","constructor_top10","constructor_podium","constructor_dnf","constructor_grid","constructor_quali"}
    keep = [c for c in cr if c.startswith("constructor_") and c not in current_aggregates] + ["season","round","constructorId"]
    x = x.merge(cr[keep], on=["season","round","constructorId"], how="left", validate="many_to_one")

    # Season-to-date totals are shifted, so race N never includes race N.
    x["driver_points_before_season"] = x.groupby(["season","driverId"])["grand_prix_points"].transform(lambda s: s.shift().cumsum()).fillna(0)
    x["driver_event_points_before_season"] = x.groupby(["season","driverId"])["event_points"].transform(lambda s: s.shift().cumsum()).fillna(0)
    x["races_completed_current_season_before_driver"] = x.groupby(["season","driverId"]).cumcount()
    cr["constructor_points_before_season"] = cr.groupby(["season","constructorId"])["constructor_points"].transform(lambda s: s.shift().cumsum()).fillna(0)
    cr["constructor_event_points_before_season"] = cr.groupby(["season","constructorId"])["constructor_event_points"].transform(lambda s: s.shift().cumsum()).fillna(0)
    cr["races_completed_current_season_before_constructor"] = cr.groupby(["season","constructorId"]).cumcount()
    x = x.merge(cr[["season","round","constructorId","constructor_points_before_season","constructor_event_points_before_season","races_completed_current_season_before_constructor"]], on=["season","round","constructorId"], how="left", validate="many_to_one")

    # Teammate values known pre-race/current quali.
    for source, target in [("quali_position","teammate_quali_position"),("grid_position","teammate_grid_position"),("driver_points_last_5","teammate_recent_points_last_5")]:
        totals = x.groupby(["raceId","constructorId"])[source].transform("sum")
        counts = x.groupby(["raceId","constructorId"])[source].transform("count")
        x[target] = (totals - x[source]).where(counts > 1)
    x["quali_position_vs_teammate"] = x.quali_position - x.teammate_quali_position
    x["grid_position_vs_teammate"] = x.grid_position - x.teammate_grid_position

    # Historical circuit features.
    x["driver_avg_finish_on_circuit_before"] = _expanding_before(x.positionOrder, [x.driverId,x.circuitId], "mean")
    x["driver_points_on_circuit_before"] = _expanding_before(x.grand_prix_points, [x.driverId,x.circuitId], "sum")
    x["constructor_avg_finish_on_circuit_before"] = _expanding_before(x.positionOrder, [x.constructorId,x.circuitId], "mean")
    x["constructor_points_on_circuit_before"] = _expanding_before(x.grand_prix_points, [x.constructorId,x.circuitId], "sum")
    x["circuit_dnf_rate_before"] = _expanding_before(x.is_dnf, x.circuitId, "mean")
    x["grid_to_finish_delta"] = x.positionOrder - x.grid_position
    x["circuit_avg_grid_to_finish_delta_before"] = _expanding_before(x.grid_to_finish_delta, x.circuitId, "mean")

    x["driver_points_share_before"] = x.driver_points_before_season / x.groupby(["season","round"]).driver_points_before_season.transform("max").replace(0,np.nan)
    x["constructor_points_share_before"] = x.constructor_points_before_season / x.groupby(["season","round"]).constructor_points_before_season.transform("max").replace(0,np.nan)
    x["driver_official_standings_points_before"] = x["driver_event_points_before_season"]
    x["constructor_official_standings_points_before"] = x["constructor_event_points_before_season"]
    x["driver_recent_points_last_3_2026"] = np.where(x.season.eq(2026), _shifted_rolling(x.event_points, dg, 3, "sum"), np.nan)
    x["driver_recent_points_last_5_2026"] = np.where(x.season.eq(2026), _shifted_rolling(x.event_points, dg, 5, "sum"), np.nan)
    x["constructor_recent_points_last_3_2026"] = np.where(x.season.eq(2026), _shifted_rolling(x.event_points, x.constructorId, 3, "sum"), np.nan)
    x["constructor_recent_points_last_5_2026"] = np.where(x.season.eq(2026), _shifted_rolling(x.event_points, x.constructorId, 5, "sum"), np.nan)
    x["driver_top10_rate_2026_before"] = np.where(x.season.eq(2026), _expanding_before(x.target_top10, x.driverId, "mean"), np.nan)
    x["constructor_top10_rate_2026_before"] = np.where(x.season.eq(2026), _expanding_before(x.target_top10, x.constructorId, "mean"), np.nan)
    x["driver_avg_finish_2026_before"] = np.where(x.season.eq(2026), _expanding_before(x.positionOrder, x.driverId, "mean"), np.nan)
    x["constructor_avg_finish_2026_before"] = np.where(x.season.eq(2026), _expanding_before(x.positionOrder, x.constructorId, "mean"), np.nan)
    x["driver_avg_quali_2026_before"] = np.where(x.season.eq(2026), _expanding_before(x.quali_position, x.driverId, "mean"), np.nan)
    x["constructor_avg_quali_2026_before"] = np.where(x.season.eq(2026), _expanding_before(x.quali_position, x.constructorId, "mean"), np.nan)
    x["driver_reliability_rate_2026_before"] = np.where(x.season.eq(2026), 1 - _expanding_before(x.is_dnf, x.driverId, "mean"), np.nan)
    x["constructor_reliability_rate_2026_before"] = np.where(x.season.eq(2026), 1 - _expanding_before(x.is_dnf, x.constructorId, "mean"), np.nan)
    x["driver_strength_rank_before"] = x.groupby(["season","round"]).driver_points_before_season.rank(method="dense", ascending=False)
    x["constructor_strength_rank_before"] = x.groupby(["season","round"]).constructor_points_before_season.rank(method="dense", ascending=False)
    x["missing_driver_history"] = x.driver_avg_finish_last_5.isna().astype("int8")
    x["missing_constructor_history"] = x.constructor_avg_finish_last_5.isna().astype("int8")
    x["missing_circuit_history"] = x.driver_avg_finish_on_circuit_before.isna().astype("int8")
    return x

def impute_features(df, feature_columns=None):
    """Median-impute numeric model features; missingness flags preserve cold starts."""
    out = df.copy(); cols = feature_columns or out.select_dtypes(include="number").columns
    for col in cols:
        if out[col].isna().any(): out[col] = out[col].fillna(out[col].median())
    return out

def assert_no_leakage(base, featured):
    """Spot-check first driver row and uniqueness invariants."""
    assert not featured.duplicated(["season","round","driverId"]).any()
    first = featured.groupby("driverId", sort=False).head(1)
    assert first.driver_races_before.eq(0).all()
    assert first.driver_avg_finish_last_5.isna().all(), "Current race leaked into rolling history"
