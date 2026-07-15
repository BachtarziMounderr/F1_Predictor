"""Build the leakage-safe modeling dataset from local Kaggle CSVs."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config,ensure_directories
from src.data_loader import load_csv_data,build_driver_race_table
from src.cleaning import clean_driver_race_table
from src.targets import create_targets
from src.features import add_pre_race_features,assert_no_leakage
from src.integrate_jolpica import load_snapshots,integrate_jolpica

def main():
    cfg=load_config(); ensure_directories(cfg)
    fallbacks=[cfg["root"]/p for p in cfg["paths"].get("raw_data_fallbacks",[])]
    tables=load_csv_data(cfg["resolved_paths"]["raw_data_dir"],fallbacks)
    base=clean_driver_race_table(build_driver_race_table(tables,cfg["data"]["min_season"]))
    base.to_csv(cfg["resolved_paths"]["processed_data_dir"]/"base_driver_race_table.csv",index=False)
    snapshots=load_snapshots(cfg["resolved_paths"]["external_data_dir"]/"jolpica",cfg["jolpica"]["seasons_to_fetch"])
    if any(not s["results"].empty or not s["qualifying"].empty for s in snapshots.values()):
        base,unmapped=integrate_jolpica(base,tables,snapshots,cfg["resolved_paths"]["external_data_dir"])
        base=clean_driver_race_table(base)
        base.to_csv(cfg["resolved_paths"]["processed_data_dir"]/"base_driver_race_table_with_jolpica.csv",index=False)
    base["grand_prix_points"]=base["points"]
    base["sprint_points"]=0.0
    base["event_points"]=base["grand_prix_points"].fillna(0)+base["sprint_points"].fillna(0)
    sprint_rounds=set(cfg.get("calendar",{}).get("sprint_rounds_2026",[]))
    base["sprint_weekend_flag"]=base.apply(lambda r: int(r["season"]==2026 and r["round"] in sprint_rounds), axis=1)
    final=add_pre_race_features(create_targets(base),cfg["features"]["rolling_windows"])
    assert_no_leakage(base,final)
    historical=final.season.le(cfg["data"]["test_season"])
    if final.loc[historical,["target_top10","target_podium","target_points"]].isna().any().any(): raise AssertionError("Missing targets in 2014-2025")
    name="f1_driver_race_dataset_with_jolpica.csv" if final.season.max()>cfg["data"]["kaggle_until_season"] else "f1_driver_race_dataset.csv"
    final.to_csv(cfg["resolved_paths"]["processed_data_dir"]/name,index=False)
    print(f"Saved {len(final):,} rows; seasons {final.season.min()}-{final.season.max()}")
if __name__=="__main__": main()
