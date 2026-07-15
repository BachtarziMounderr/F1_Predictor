from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import load_config,ensure_directories
from src.jolpica_api import fetch_season_snapshot

def main():
    parser=argparse.ArgumentParser(); group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--season",type=int); group.add_argument("--seasons",type=int,nargs="+")
    args=parser.parse_args(); cfg=load_config(); ensure_directories(cfg)
    seasons=args.seasons or [args.season]; output=cfg["resolved_paths"]["external_data_dir"]/"jolpica"
    opts={"base_url":cfg["jolpica"]["base_url"],"timeout":cfg["jolpica"]["timeout"],"limit":cfg["jolpica"]["limit"]}
    for season in seasons:
        frames,paths=fetch_season_snapshot(season,output,**opts)
        print(f"{season}: races={len(frames['schedule'])}, results={len(frames['results'])}, qualifying={len(frames['qualifying'])}, driver standings={len(frames['driver_standings'])}, constructor standings={len(frames['constructor_standings'])}")
        for path in paths.values(): print(f"  saved: {path}")
if __name__=="__main__": main()
