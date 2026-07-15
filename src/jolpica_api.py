"""Resilient, season-agnostic Jolpica-F1 client."""
from pathlib import Path
import logging
import requests
import pandas as pd

LOG=logging.getLogger(__name__)
DEFAULT_BASE="https://api.jolpi.ca/ergast/f1"

def _get(path, base_url=DEFAULT_BASE, timeout=30, limit=1000):
    """Fetch every paginated MRData page; return an empty dict on network failure."""
    offset=0; merged=None
    while True:
        try:
            response=requests.get(f"{base_url.rstrip('/')}/{path.lstrip('/')}.json",params={"limit":limit,"offset":offset},timeout=timeout)
            response.raise_for_status(); mr=response.json().get("MRData",{})
        except (requests.RequestException,ValueError) as exc:
            LOG.warning("Jolpica unavailable for %s: %s",path,exc); return merged or {}
        if merged is None: merged=mr
        else: _merge_mrdata(merged,mr)
        total=int(mr.get("total",0)); page_limit=int(mr.get("limit",limit)); offset += page_limit
        if offset>=total or page_limit<=0: break
    return merged or {}

def _merge_mrdata(target,page):
    """Append race/standing lists from a subsequent API page."""
    for table_name,list_name in (("RaceTable","Races"),("StandingsTable","StandingsLists")):
        src=page.get(table_name,{}).get(list_name,[])
        if src: target.setdefault(table_name,{}).setdefault(list_name,[]).extend(src)

def _race_rows(season,endpoint=None,round_number=None,base_url=DEFAULT_BASE,timeout=30,limit=1000):
    parts=[str(season)];
    if round_number is not None: parts.append(str(round_number))
    if endpoint: parts.append(endpoint)
    return _get("/".join(parts),base_url,timeout,limit).get("RaceTable",{}).get("Races",[])

def fetch_schedule(season,**kwargs): return pd.json_normalize(_race_rows(season,**kwargs))

def _nested_races(season,key,endpoint,round_number=None,**kwargs):
    rows=[]
    for race in _race_rows(season,endpoint,round_number,**kwargs):
        common={"season":race.get("season"),"round":race.get("round"),"raceName":race.get("raceName"),"date":race.get("date"),
                "Circuit.circuitId":race.get("Circuit",{}).get("circuitId")}
        rows.extend({**common,**item} for item in race.get(key,[]))
    return pd.json_normalize(rows)

def fetch_results(season,round_number=None,**kwargs): return _nested_races(season,"Results","results",round_number,**kwargs)
def fetch_qualifying(season,round_number=None,**kwargs): return _nested_races(season,"QualifyingResults","qualifying",round_number,**kwargs)

def _standings(season,kind,round_number=None,**kwargs):
    suffix=f"/{round_number}" if round_number is not None else ""
    lists=_get(f"{season}{suffix}/{kind}standings",**kwargs).get("StandingsTable",{}).get("StandingsLists",[])
    rows=[]
    key="DriverStandings" if kind=="driver" else "ConstructorStandings"
    for item in lists:
        common={"season":item.get("season"),"round":item.get("round")}
        rows.extend({**common,**row} for row in item.get(key,[]))
    return pd.json_normalize(rows)

def fetch_driver_standings(season,round_number=None,**kwargs): return _standings(season,"driver",round_number,**kwargs)
def fetch_constructor_standings(season,round_number=None,**kwargs): return _standings(season,"constructor",round_number,**kwargs)

def fetch_season_snapshot(season,output_dir="data/external/jolpica",**kwargs):
    """Fetch and persist all V1 endpoints for one season."""
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    frames={"schedule":fetch_schedule(season,**kwargs),"results":fetch_results(season,**kwargs),
            "qualifying":fetch_qualifying(season,**kwargs),"driver_standings":fetch_driver_standings(season,**kwargs),
            "constructor_standings":fetch_constructor_standings(season,**kwargs)}
    paths={}
    for name,df in frames.items():
        paths[name]=output/f"jolpica_{season}_{name}.csv"; df.to_csv(paths[name],index=False)
        if df.empty: LOG.warning("No %s returned for %s",name,season)
    return frames,paths

# Backwards-compatible alias.
def fetch_2026_schedule(**kwargs): return fetch_schedule(2026,**kwargs)
