from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.prediction_service import predict_race
from src.config import load_config, ensure_directories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--as-of-date", default=None)
    args = parser.parse_args()
    cfg = load_config()
    ensure_directories(cfg)
    predictions, metadata = predict_race(args.round, args.as_of_date, cfg)
    out = cfg["resolved_paths"]["predictions_dir"] / "2026_next_races_predictions.csv"
    predictions.to_csv(out, index=False)
    print(metadata)
    print(predictions.to_string(index=False))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
