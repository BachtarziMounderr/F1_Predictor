import numpy as np

def classification_baselines(df):
    return {"grid_top10":df.grid_position.le(10).astype(int), "quali_top10":df.quali_position.le(10).astype(int),
            "grid_podium":df.grid_position.le(3).astype(int), "quali_podium":df.quali_position.le(3).astype(int)}

def fit_points_baselines(train):
    return {"grid":train.groupby("grid_position").target_points.mean(), "quali":train.groupby("quali_position").target_points.mean(), "global":train.target_points.mean()}

def predict_points_baseline(df, fitted, kind="grid"):
    column = "grid_position" if kind == "grid" else "quali_position"
    return df[column].map(fitted[kind]).fillna(fitted["global"]).clip(lower=0).to_numpy()

