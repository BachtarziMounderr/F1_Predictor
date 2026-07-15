import numpy as np
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    HAS_XGB = False

def _params(config, task):
    p = dict(config["models"][task]); p.pop("type",None); p.pop("auto_scale_pos_weight",None)
    p["random_state"] = config["project"]["random_state"]; p["n_jobs"] = -1
    return p

def _classifier(X, y, config, task, sample_weight=None):
    p=_params(config,task)
    if task=="podium" and config["models"][task].get("auto_scale_pos_weight"):
        p["scale_pos_weight"] = max(1, (y==0).sum()/max(1,(y==1).sum()))
    model = XGBClassifier(**p) if HAS_XGB else RandomForestClassifier(n_estimators=300, random_state=p["random_state"], n_jobs=-1, class_weight="balanced")
    return model.fit(X,y,sample_weight=sample_weight)

def train_top10_model(X_train,y_train,X_val,y_val,config,sample_weight=None): return _classifier(X_train,y_train,config,"top10",sample_weight)
def train_podium_model(X_train,y_train,X_val,y_val,config,sample_weight=None): return _classifier(X_train,y_train,config,"podium",sample_weight)
def train_points_model(X_train,y_train,X_val,y_val,config,sample_weight=None):
    p=_params(config,"points")
    model=XGBRegressor(**p) if HAS_XGB else RandomForestRegressor(n_estimators=300,random_state=p["random_state"],n_jobs=-1)
    return model.fit(X_train,y_train,sample_weight=sample_weight)

