def create_targets(df):
    out = df.copy()
    known = out["positionOrder"].notna()
    out["target_top10"] = out["positionOrder"].le(10).where(known).astype("Int8")
    out["target_podium"] = out["positionOrder"].le(3).where(known).astype("Int8")
    out["target_points"] = out["points"]
    return out
