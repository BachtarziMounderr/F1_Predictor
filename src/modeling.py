"""Shared modeling schema and preprocessing helpers."""
import numpy as np

CURRENT_OUTCOMES={"positionOrder","points","statusId","target_top10","target_podium","target_points","is_dnf","grid_to_finish_delta"}
IDENTIFIERS={"raceId","driverId","constructorId","circuitId"}
QUALI_FEATURES={
    "quali_position","grid_position","grid","grid_vs_quali_delta",
    "qualified_top3","qualified_top10","started_top3","started_top10",
    "teammate_quali_position","teammate_grid_position",
    "quali_position_vs_teammate","grid_position_vs_teammate",
}

def select_feature_columns(df, mode="post_quali"):
    """Whitelist numeric features known before lights-out; reject raw result fields."""
    numeric=set(df.select_dtypes(include=np.number).columns)
    exact={"season","race_round","round","grid","grid_position","quali_position","grid_vs_quali_delta",
           "qualified_top3","qualified_top10","started_top3","started_top10","is_2026_regulation",
           "missing_qualifying","missing_driver_history","missing_constructor_history","missing_circuit_history",
           "sprint_weekend_flag"}
    prefixes=("driver_","constructor_","teammate_","quali_position_vs_","grid_position_vs_","circuit_","races_completed_")
    cols=sorted(c for c in numeric if c in exact or c.startswith(prefixes))
    raw_outcomes={"position","positionOrder","points","grand_prix_points","sprint_points","event_points",
                  "laps","milliseconds","fastestLap","rank","fastestLapSpeed","statusId","resultId","number"}
    cols=[c for c in cols if c not in CURRENT_OUTCOMES|IDENTIFIERS|raw_outcomes]
    if mode == "pre_quali":
        cols=[c for c in cols if c not in QUALI_FEATURES]
    elif mode != "post_quali":
        raise ValueError("mode must be 'pre_quali' or 'post_quali'")
    forbidden=[c for c in cols if c in CURRENT_OUTCOMES]
    if forbidden: raise AssertionError(f"Outcome leakage columns selected: {forbidden}")
    return cols
