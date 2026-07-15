import json, logging
from pathlib import Path
import numpy as np

def get_logger(name):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    return logging.getLogger(name)

def save_json(data, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=lambda v: float(v) if isinstance(v,(np.floating,np.integer)) else str(v)), encoding="utf-8")
