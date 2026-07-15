"""Entity reconciliation between Jolpica text IDs and Kaggle identifiers."""
from pathlib import Path
import re,unicodedata
import pandas as pd

def normalize_name(value):
    text=unicodedata.normalize("NFKD",str(value or "")).encode("ascii","ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+","",text)

def _mapping(source,external_ids,entity,id_col,ref_col,name_col=None):
    records=[]; ref_lookup={normalize_name(v):(row[id_col],v) for _,row in source.iterrows() if pd.notna(v:=row.get(ref_col))}
    name_lookup={normalize_name(row.get(name_col)):(row[id_col],row[ref_col]) for _,row in source.iterrows()} if name_col else {}
    for ext_id,ext_name in external_ids:
        match=ref_lookup.get(normalize_name(ext_id)) or name_lookup.get(normalize_name(ext_name))
        records.append({"entity_type":entity,"jolpica_id":ext_id,"jolpica_name":ext_name,"internal_id":match[0] if match else pd.NA,
                        "internal_ref":match[1] if match else pd.NA,"match_method":"ref_or_normalized_name" if match else "unmapped"})
    return pd.DataFrame(records).drop_duplicates("jolpica_id")

def build_id_mappings(tables,jolpica_frames,output_dir):
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    combined=pd.concat(jolpica_frames,ignore_index=True) if jolpica_frames else pd.DataFrame()
    def pairs(id_col,name_cols):
        if id_col not in combined: return []
        names=combined[name_cols].fillna("").astype(str).agg(" ".join,axis=1) if all(c in combined for c in name_cols) else pd.Series("",index=combined.index)
        return list(zip(combined[id_col].dropna(),names.loc[combined[id_col].dropna().index]))
    drivers=_mapping(tables["drivers"],pairs("Driver.driverId",["Driver.givenName","Driver.familyName"]),"driver","driverId","driverRef",None)
    constructors=_mapping(tables["constructors"],pairs("Constructor.constructorId",["Constructor.name"]),"constructor","constructorId","constructorRef","name")
    circuits=_mapping(tables["circuits"],pairs("Circuit.circuitId",["Circuit.circuitName"]),"circuit","circuitId","circuitRef","name")
    mappings={"drivers":drivers,"constructors":constructors,"circuits":circuits}
    for name,df in mappings.items(): df.to_csv(output/f"id_mapping_{name}.csv",index=False)
    unmapped=pd.concat([d[d.internal_id.isna()] for d in mappings.values()],ignore_index=True)
    unmapped.to_csv(output/"unmapped_jolpica_entities.csv",index=False)
    return mappings,unmapped

def apply_or_allocate(mapping,next_id):
    """Allocate stable internal IDs for genuinely new entities while retaining audit status."""
    out=mapping.copy()
    for idx in out.index[out.internal_id.isna()]:
        next_id+=1; out.loc[idx,"internal_id"]=next_id; out.loc[idx,"internal_ref"]=out.loc[idx,"jolpica_id"]
    out["internal_id"]=out.internal_id.astype(int)
    return out,next_id
