import pandas as pd

NUMERIC = ["season","round","raceId","driverId","constructorId","circuitId","grid","positionOrder","points","quali_position","statusId"]

def clean_driver_race_table(df):
    """Normalize types while retaining missing qualifying positions explicitly."""
    out = df.replace(r"\N", pd.NA).copy()
    for col in NUMERIC:
        if col in out: out[col] = pd.to_numeric(out[col], errors="coerce")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["missing_qualifying"] = out["quali_position"].isna().astype("int8")
    out["grid_position"] = out["grid"].astype("float64").mask(out["grid"].eq(0))
    return out
