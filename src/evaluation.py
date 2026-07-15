import numpy as np
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,
 average_precision_score,brier_score_loss,confusion_matrix,mean_absolute_error,mean_squared_error,r2_score)
from scipy.stats import spearmanr

def classification_metrics(y, pred, proba):
    out={"accuracy":accuracy_score(y,pred),"precision":precision_score(y,pred,zero_division=0),"recall":recall_score(y,pred,zero_division=0),
         "f1":f1_score(y,pred,zero_division=0),"average_precision":average_precision_score(y,proba),"brier_score":brier_score_loss(y,proba),
         "confusion_matrix":confusion_matrix(y,pred).tolist()}
    out["roc_auc"] = roc_auc_score(y,proba) if len(np.unique(y))>1 else None
    return out

def regression_metrics(y,pred):
    return {"mae":mean_absolute_error(y,pred),"rmse":mean_squared_error(y,pred)**.5,"r2":r2_score(y,pred),"spearman":spearmanr(y,pred).statistic}

