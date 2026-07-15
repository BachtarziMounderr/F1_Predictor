def temporal_split(df, config):
    d = config["data"]; lo, hi = d["train_seasons"]
    parts = {"train":df[df.season.between(lo,hi)], "validation":df[df.season.eq(d["validation_season"])],
             "test":df[df.season.eq(d["test_season"])], "live":df[df.season.eq(d["live_season"])]}
    seasons = [set(p.season.unique()) for p in parts.values()]
    assert all(not seasons[i] & seasons[j] for i in range(4) for j in range(i+1,4))
    assert d["live_season"] not in seasons[0] or d.get("use_2026_for_finetuning",False)
    if parts["train"].empty: raise ValueError("Training split is empty")
    if parts["validation"].empty: raise ValueError("Validation split is empty")
    if parts["test"].empty: raise ValueError("Official test season is empty; fetch and integrate Jolpica 2025")
    return parts

def temporal_sample_weights(seasons, config):
    w=config["sample_weights"]["weights"]
    return seasons.map(lambda y: w["historical_2014_2021"] if y<=2021 else w["recent_2022_2023"] if y<=2023 else w["validation_test_2024_2025"] if y<=2025 else w["live_2026"])
