from pathlib import Path
import pandas as pd
from .utils import get_logger

LOG = get_logger(__name__)
REQUIRED = {"races", "results", "drivers", "constructors", "qualifying", "circuits"}

def discover_data_dir(data_dir, fallbacks=("data_f1",)):
    candidates = [Path(data_dir), *(Path(x) for x in fallbacks)]
    for directory in candidates:
        if directory.is_dir() and REQUIRED <= {p.stem for p in directory.glob("*.csv")}:
            return directory
    raise FileNotFoundError(f"Required CSVs not found in: {candidates}")

def load_csv_data(data_dir, fallbacks=("data_f1",)):
    """Load all available CSVs after validating the V1 core tables."""
    directory = discover_data_dir(data_dir, fallbacks)
    tables = {p.stem: pd.read_csv(p, na_values=[r"\N"], low_memory=False) for p in directory.glob("*.csv")}
    missing = REQUIRED - tables.keys()
    if missing: raise FileNotFoundError(f"Missing required tables: {sorted(missing)}")
    for name, df in tables.items():
        miss = df.isna().mean().nlargest(3); LOG.info("%s: %s rows x %s cols; missing: %s", name, *df.shape, miss.to_dict())
    return tables

def build_driver_race_table(tables, min_season=2014):
    """Build one pre-feature row per driver/race from Kaggle tables."""
    races = tables["races"].rename(columns={"year":"season", "name":"raceName"})
    results = tables["results"].copy()
    base = results.merge(races[["raceId","season","round","raceName","date","circuitId"]], on="raceId", validate="many_to_one")
    base = base.merge(tables["drivers"][["driverId","driverRef","forename","surname"]], on="driverId", validate="many_to_one")
    base["driver_name"] = (base.pop("forename").fillna("") + " " + base.pop("surname").fillna("")).str.strip()
    constructors = tables["constructors"][["constructorId","constructorRef","name"]].rename(columns={"name":"constructor_name"})
    base = base.merge(constructors, on="constructorId", validate="many_to_one")
    base = base.merge(tables["circuits"][["circuitId","circuitRef"]], on="circuitId", validate="many_to_one")
    q = tables["qualifying"][["raceId","driverId","position"]].rename(columns={"position":"quali_position"})
    # Early F1 records can contain shared drives (multiple result rows per driver/race).
    # Qualifying remains unique on the right; post-2014 uniqueness is asserted below.
    base = base.merge(q, on=["raceId","driverId"], how="left", validate="many_to_one")
    if "status" in tables:
        base = base.merge(tables["status"][["statusId","status"]], on="statusId", how="left", validate="many_to_one")
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    before = len(base); base = base[base.season >= min_season].copy()
    LOG.info("Minimum season filter removed %d rows", before-len(base))
    key = ["season","round","driverId"]
    if base.duplicated(key).any(): raise ValueError("Duplicate season/round/driver rows")
    return base.sort_values(["date","round","driverId"]).reset_index(drop=True)
