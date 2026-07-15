from pathlib import Path
import sys,json,joblib,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import load_config,ensure_directories
from src.prediction import generate_2026_predictions

def main():
    cfg=load_config(); ensure_directories(cfg); modeldir=cfg["resolved_paths"]["models_dir"]
    dataset=cfg["resolved_paths"]["processed_data_dir"]/"f1_driver_race_dataset_with_jolpica.csv"
    df=pd.read_csv(dataset); live=df[df.season.eq(cfg["data"]["live_season"])].copy()
    if live.empty: raise ValueError("No 2026 rows available; fetch Jolpica and rebuild first")
    models={name:joblib.load(modeldir/f"{name}_model.pkl") for name in ("top10","podium","points")}
    preprocessor=joblib.load(modeldir/"preprocessing_pipeline.pkl"); features=json.loads((modeldir/"feature_columns.json").read_text(encoding="utf-8"))
    combined,missing=generate_2026_predictions(live,models,cfg,features,preprocessor)
    print(f"2026 predictions: {len(combined)}; missing critical inputs: {len(missing)}")
if __name__=="__main__":main()
