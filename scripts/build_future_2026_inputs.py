from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, ensure_directories
from src.future_inputs import write_future_2026_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of-date", default=None)
    parser.add_argument("--round", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    ensure_directories(cfg)
    future, path = write_future_2026_inputs(cfg, args.as_of_date, args.round)
    print(f"Saved {len(future)} future driver-race rows to {path}")
    if not future.empty:
        print(future[["season", "round", "raceName", "date", "driver_name", "constructor_name", "prediction_mode"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
