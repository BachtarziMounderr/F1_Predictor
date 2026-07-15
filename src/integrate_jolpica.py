from pathlib import Path
import pandas as pd
from .id_mapping import build_id_mappings,apply_or_allocate
from .utils import get_logger

LOG=get_logger(__name__)
def _read(path): return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()

def load_snapshots(directory,seasons=(2025,2026)):
    directory=Path(directory); return {s:{k:_read(directory/f"jolpica_{s}_{k}.csv") for k in ("schedule","results","qualifying","driver_standings","constructor_standings")} for s in seasons}

def integrate_jolpica(base,tables,snapshots,external_dir):
    """Convert result/qualifying snapshots to the canonical driver-race schema."""
    entity_frames=[]
    for snap in snapshots.values(): entity_frames += [snap["results"],snap["qualifying"],snap["schedule"]]
    mappings,unmapped=build_id_mappings(tables,[x for x in entity_frames if not x.empty],external_dir)
    dm,_=apply_or_allocate(mappings["drivers"],int(tables["drivers"].driverId.max()))
    cm,_=apply_or_allocate(mappings["constructors"],int(tables["constructors"].constructorId.max()))
    xim,_=apply_or_allocate(mappings["circuits"],int(tables["circuits"].circuitId.max()))
    d_id=dm.set_index("jolpica_id").internal_id.to_dict(); d_ref=dm.set_index("jolpica_id").internal_ref.to_dict()
    c_id=cm.set_index("jolpica_id").internal_id.to_dict(); c_ref=cm.set_index("jolpica_id").internal_ref.to_dict()
    x_id=xim.set_index("jolpica_id").internal_id.to_dict(); x_ref=xim.set_index("jolpica_id").internal_ref.to_dict()
    status_lookup=tables.get("status",pd.DataFrame(columns=["statusId","status"])).set_index("status").statusId.to_dict()
    additions=[]; race_start=int(tables["races"].raceId.max())+1
    for season,snap in snapshots.items():
        results=snap["results"].copy(); quali=snap["qualifying"].copy(); schedule=snap["schedule"].copy()
        if results.empty and quali.empty: LOG.warning("No driver-level Jolpica rows for %s",season); continue
        keys=["season","round","Driver.driverId"]
        merged=pd.merge(results if not results.empty else pd.DataFrame(columns=keys),
                        quali[[c for c in quali if c in keys+["position"]]].rename(columns={"position":"quali_position"}) if not quali.empty else pd.DataFrame(columns=keys+["quali_position"]),
                        on=keys,how="outer",suffixes=("","_q"),indicator=True)
        if season<=2025:
            orphaned=merged._merge.eq("right_only").sum()
            if orphaned: LOG.warning("Dropping %d qualifying-only 2025 entries without an official race result",orphaned)
            merged=merged[~merged._merge.eq("right_only")].copy()
        schedule_idx=schedule.copy(); schedule_idx["round"]=pd.to_numeric(schedule_idx.get("round"),errors="coerce")
        sched=schedule_idx.set_index("round").to_dict("index") if not schedule_idx.empty else {}
        merged["season"]=pd.to_numeric(merged.season,errors="coerce").fillna(season).astype(int); merged["round"]=pd.to_numeric(merged["round"],errors="coerce").astype(int)
        merged["raceId"]=merged["round"].map(lambda r:race_start+(season-2025)*100+r)
        merged["raceName"]=merged.apply(lambda r:r.get("raceName") or sched.get(r["round"],{}).get("raceName"),axis=1)
        merged["date"]=merged.apply(lambda r:r.get("date") or sched.get(r["round"],{}).get("date"),axis=1)
        circuit=merged.apply(lambda r:r.get("Circuit.circuitId") or sched.get(r["round"],{}).get("Circuit.circuitId"),axis=1)
        merged["circuitId"]=circuit.map(x_id); merged["circuitRef"]=circuit.map(x_ref)
        merged["driverId"]=merged["Driver.driverId"].map(d_id); merged["driverRef"]=merged["Driver.driverId"].map(d_ref)
        merged["driver_name"]=(merged.get("Driver.givenName",pd.Series("",index=merged.index)).fillna("")+" "+merged.get("Driver.familyName",pd.Series("",index=merged.index)).fillna("")).str.strip()
        constructor=merged.get("Constructor.constructorId",pd.Series(index=merged.index,dtype=object))
        merged["constructorId"]=constructor.map(c_id); merged["constructorRef"]=constructor.map(c_ref); merged["constructor_name"]=merged.get("Constructor.name")
        merged["grid"]=pd.to_numeric(merged.get("grid"),errors="coerce"); merged["positionOrder"]=pd.to_numeric(merged.get("position"),errors="coerce")
        merged["points"]=pd.to_numeric(merged.get("points"),errors="coerce"); merged["statusId"]=merged.get("status",pd.Series(index=merged.index)).map(status_lookup)
        additions.append(merged)
    if not additions: return base.copy(),unmapped
    add=pd.concat(additions,ignore_index=True)
    canonical=list(base.columns); add=add.reindex(columns=canonical)
    combined=pd.concat([base,add],ignore_index=True).sort_values(["date","round","driverId"])
    if combined.duplicated(["season","round","driverRef"]).any(): raise ValueError("Duplicate season/round/driverRef after Jolpica integration")
    return combined.reset_index(drop=True),unmapped
