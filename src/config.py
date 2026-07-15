from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config(path=ROOT / "config.yaml"):
    """Load YAML config and resolve all configured project paths."""
    with Path(path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["root"] = ROOT
    cfg["resolved_paths"] = {k: ROOT / v for k, v in cfg["paths"].items() if k != "raw_data_fallbacks"}
    return cfg

def ensure_directories(cfg):
    for path in cfg["resolved_paths"].values():
        path.mkdir(parents=True, exist_ok=True)
    (cfg["resolved_paths"]["reports_dir"] / "metrics").mkdir(parents=True, exist_ok=True)
    (cfg["resolved_paths"]["reports_dir"] / "figures").mkdir(parents=True, exist_ok=True)

