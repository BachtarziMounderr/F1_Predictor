from pathlib import Path
import pandas as pd

IDENTITY=["season","round","raceName","date","driver_name","constructor_name","grid","quali_position"]

def generate_2026_predictions(df_live, models, config, feature_columns, preprocessor=None):
    outdir=Path(config["resolved_paths"]["predictions_dir"]); outdir.mkdir(parents=True,exist_ok=True)
    missing=df_live[df_live[["grid_position","quali_position"]].isna().any(axis=1)].copy()
    ready=df_live.drop(index=missing.index).copy(); missing[IDENTITY].to_csv(outdir/"2026_missing_inputs.csv",index=False)
    if ready.empty:
        for name in ("2026_combined_predictions.csv","2026_top10_predictions.csv","2026_podium_predictions.csv","2026_expected_points.csv"):
            pd.DataFrame().to_csv(outdir/name,index=False)
        return pd.DataFrame(),missing
    combined=ready[IDENTITY].copy(); X=preprocessor.transform(ready[feature_columns]) if preprocessor is not None else ready[feature_columns]
    combined["proba_top10"]=models["top10"].predict_proba(X)[:,1]
    combined["proba_podium"]=models["podium"].predict_proba(X)[:,1]
    combined["expected_points"]=models["points"].predict(X).clip(min=0)
    combined["predicted_top10"]=(combined.proba_top10>=.5).astype(int); combined["predicted_podium"]=(combined.proba_podium>=.5).astype(int)
    combined.to_csv(outdir/"2026_combined_predictions.csv",index=False)
    combined[IDENTITY+["proba_top10","predicted_top10"]].to_csv(outdir/"2026_top10_predictions.csv",index=False)
    combined[IDENTITY+["proba_podium","predicted_podium"]].to_csv(outdir/"2026_podium_predictions.csv",index=False)
    combined[IDENTITY+["expected_points"]].to_csv(outdir/"2026_expected_points.csv",index=False)
    return combined,missing
