from pathlib import Path
import sys,json,joblib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd,numpy as np,matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay,PrecisionRecallDisplay
from src.config import load_config,ensure_directories
from src.split import temporal_split,temporal_sample_weights
from src.modeling import select_feature_columns
from src.models import train_top10_model,train_podium_model,train_points_model
from src.baselines import classification_baselines,fit_points_baselines,predict_points_baseline
from src.evaluation import classification_metrics,regression_metrics
from src.utils import save_json

def _feature_importance(model,features,path,title):
    values=getattr(model,"feature_importances_",None)
    if values is None:return
    order=np.argsort(values)[-20:]; plt.figure(figsize=(8,7)); plt.barh(np.array(features)[order],values[order]); plt.title(title); plt.tight_layout(); plt.savefig(path,dpi=140); plt.close()
def main():
    cfg=load_config(); ensure_directories(cfg); processed=cfg["resolved_paths"]["processed_data_dir"]
    path=processed/"f1_driver_race_dataset_with_jolpica.csv"
    if not path.exists(): raise FileNotFoundError("Run scripts/build_dataset.py after fetching Jolpica")
    df=pd.read_csv(path); parts=temporal_split(df,cfg); features=select_feature_columns(df)
    imputer=SimpleImputer(strategy="median").fit(parts["train"][features]); X={k:imputer.transform(v[features]) for k,v in parts.items()}
    weights=temporal_sample_weights(parts["train"].season,cfg) if cfg["sample_weights"]["enabled"] else None
    models={"top10":train_top10_model(X["train"],parts["train"].target_top10,X["validation"],parts["validation"].target_top10,cfg,weights),
            "podium":train_podium_model(X["train"],parts["train"].target_podium,X["validation"],parts["validation"].target_podium,cfg,weights),
            "points":train_points_model(X["train"],parts["train"].target_points,X["validation"],parts["validation"].target_points,cfg,weights)}
    modeldir=cfg["resolved_paths"]["models_dir"]; report=cfg["resolved_paths"]["reports_dir"]; figures=report/"figures"; metricsdir=report/"metrics"
    for name,model in models.items(): joblib.dump(model,modeldir/f"{name}_model.pkl")
    joblib.dump(imputer,modeldir/"preprocessing_pipeline.pkl"); (modeldir/"feature_columns.json").write_text(json.dumps(features,indent=2),encoding="utf-8")
    all_metrics={"top10":{},"podium":{},"points":{}}; baseline_metrics={}; fitted=fit_points_baselines(parts["train"])
    for split in ("validation","test"):
        frame=parts[split]; b=classification_baselines(frame); baseline_metrics[split]={}
        for name,target in (("top10","target_top10"),("podium","target_podium")):
            proba=models[name].predict_proba(X[split])[:,1]; pred=(proba>=.5).astype(int); all_metrics[name][split]=classification_metrics(frame[target],pred,proba)
            baseline_metrics[split][name]={k:classification_metrics(frame[target],v,v.astype(float)) for k,v in b.items() if name in k}
            ConfusionMatrixDisplay.from_predictions(frame[target],pred); plt.tight_layout(); plt.savefig(figures/f"confusion_matrix_{name}_{split}.png",dpi=140); plt.close()
            RocCurveDisplay.from_predictions(frame[target],proba); plt.tight_layout(); plt.savefig(figures/f"roc_{name}_{split}.png",dpi=140); plt.close()
            if name=="podium": PrecisionRecallDisplay.from_predictions(frame[target],proba); plt.tight_layout(); plt.savefig(figures/f"precision_recall_podium_{split}.png",dpi=140); plt.close()
        pp=models["points"].predict(X[split]).clip(min=0); all_metrics["points"][split]=regression_metrics(frame.target_points,pp)
        baseline_metrics[split]["points_grid"]=regression_metrics(frame.target_points,predict_points_baseline(frame,fitted,"grid"))
        plt.figure(); plt.scatter(frame.target_points,pp,alpha=.3); plt.xlabel("Actual points"); plt.ylabel("Predicted points"); plt.tight_layout(); plt.savefig(figures/f"predicted_vs_actual_points_{split}.png",dpi=140); plt.close()
    for name,data in all_metrics.items(): save_json(data,metricsdir/f"{name}_metrics.json")
    save_json(baseline_metrics,metricsdir/"baseline_metrics.json")
    for name,model in models.items(): _feature_importance(model,features,figures/f"feature_importance_{name}.png",f"{name} feature importance")
    print({k:len(v) for k,v in parts.items()}); print(json.dumps(all_metrics,indent=2))
if __name__=="__main__":main()
