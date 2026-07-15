from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import load_config, ensure_directories


def load_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def flatten_metrics(label, metrics):
    rows = []
    for split, split_metrics in metrics.items():
        for task, values in split_metrics.items():
            if isinstance(values, dict):
                row = {"model_family": label, "split": split, "task": task}
                for key, value in values.items():
                    if key != "confusion_matrix":
                        row[key] = value
                rows.append(row)
    return rows


def main():
    cfg = load_config()
    ensure_directories(cfg)
    metrics_dir = cfg["resolved_paths"]["reports_dir"] / "metrics"
    predictions_dir = cfg["resolved_paths"]["predictions_dir"]

    current = {
        "top10": load_json(metrics_dir / "top10_metrics.json"),
        "podium": load_json(metrics_dir / "podium_metrics.json"),
        "points": load_json(metrics_dir / "points_metrics.json"),
    }
    pre_quali = load_json(metrics_dir / "pre_quali_metrics.json")
    post_quali = load_json(metrics_dir / "post_quali_metrics.json")
    ranking = load_json(metrics_dir / "ranking_metrics.json")

    rows = []
    for task, task_metrics in current.items():
        for split, values in task_metrics.items():
            row = {"model_family": "current_evaluation_baseline", "split": split, "task": task}
            for key, value in values.items():
                if key != "confusion_matrix":
                    row[key] = value
            rows.append(row)
    rows.extend(flatten_metrics("pre_quali_evaluation", pre_quali))
    rows.extend(flatten_metrics("post_quali_evaluation", post_quali))

    summary_df = pd.DataFrame(rows)
    csv_path = metrics_dir / "new_model_results_summary.csv"
    json_path = metrics_dir / "new_model_results_summary.json"
    summary_df.to_csv(csv_path, index=False)

    prediction_path = predictions_dir / "2026_next_races_predictions.csv"
    prediction_preview = []
    if prediction_path.exists():
        prediction_preview = pd.read_csv(prediction_path).head(10).to_dict(orient="records")

    payload = {
        "description": "Saved summary of current baseline, pre-qualifying, post-qualifying and ranking model results.",
        "important_note": (
            "The original official 2025 baseline remains frozen. Pre/post qualifying evaluation metrics are trained "
            "with the official temporal split for comparison; live production artifacts are saved separately under models/live."
        ),
        "metrics": rows,
        "ranking_metrics": ranking,
        "latest_next_race_prediction_preview": prediction_preview,
        "artifacts": {
            "csv": str(csv_path),
            "json": str(json_path),
            "next_race_predictions": str(prediction_path),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
