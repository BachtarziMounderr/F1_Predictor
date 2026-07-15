# F1 Predictor model audit

Generated from local artifacts.

## Data status

- Dataset: `C:\Users\bacht\Desktop\F1_Predictor\data\processed\f1_driver_race_dataset_with_jolpica.csv`
- Shape: 5303 rows x 122 columns
- Seasons: [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
- Rows by season: `{2014: 407, 2015: 378, 2016: 462, 2017: 400, 2018: 420, 2019: 420, 2020: 340, 2021: 440, 2022: 440, 2023: 440, 2024: 479, 2025: 479, 2026: 198}`
- Duplicate season/round/driverId rows: 0
- 2026 completed rounds in dataset: [1, 2, 3, 4, 5, 6, 7, 8, 9]
- `target_points` currently represents Grand Prix race points. Sprint/championship points are separated into `sprint_points` and `event_points` when rebuilt.

## Model status

- Current model types: `{'top10': "<class 'xgboost.sklearn.XGBClassifier'>", 'podium': "<class 'xgboost.sklearn.XGBClassifier'>", 'points': "<class 'xgboost.sklearn.XGBRegressor'>"}`
- Current saved feature columns: 65
- Pre-qualifying feature count after filtering: 76
- Post-qualifying feature count after filtering: 88

## Hyperparameters

```json
{
  "retrain_after_jolpica_integration": true,
  "top10": {
    "type": "xgboost_classifier",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss"
  },
  "podium": {
    "type": "xgboost_classifier",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "auto_scale_pos_weight": true
  },
  "points": {
    "type": "xgboost_regressor",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror"
  }
}
```

## Current official test metrics

```json
{
  "top10": {
    "accuracy": 0.7620041753653445,
    "precision": 0.7560975609756098,
    "recall": 0.775,
    "f1": 0.7654320987654321,
    "average_precision": 0.8206414842789829,
    "brier_score": 0.16587547035206668,
    "confusion_matrix": [
      [
        179,
        60
      ],
      [
        54,
        186
      ]
    ],
    "roc_auc": 0.8346408647140866
  },
  "podium": {
    "accuracy": 0.8914405010438413,
    "precision": 0.5909090909090909,
    "recall": 0.9027777777777778,
    "f1": 0.7142857142857143,
    "average_precision": 0.7341955984604142,
    "brier_score": 0.08093075460055929,
    "confusion_matrix": [
      [
        362,
        45
      ],
      [
        7,
        65
      ]
    ],
    "roc_auc": 0.9426358176358177
  },
  "points": {
    "mae": 2.919343679131102,
    "rmse": 4.499972409547522,
    "r2": 0.6068128007433635,
    "spearman": 0.7040223550870846
  },
  "pre_quali": {
    "top10": {
      "accuracy": 0.7098121085594989,
      "precision": 0.7327188940092166,
      "recall": 0.6625,
      "f1": 0.6958424507658644,
      "average_precision": 0.7622552786342145,
      "brier_score": 0.19623106869716606,
      "confusion_matrix": [
        [
          181,
          58
        ],
        [
          81,
          159
        ]
      ],
      "roc_auc": 0.771565550906555
    },
    "podium": {
      "accuracy": 0.8329853862212944,
      "precision": 0.46825396825396826,
      "recall": 0.8194444444444444,
      "f1": 0.5959595959595959,
      "average_precision": 0.5978733778990418,
      "brier_score": 0.10715701495042648,
      "confusion_matrix": [
        [
          340,
          67
        ],
        [
          13,
          59
        ]
      ],
      "roc_auc": 0.8967717717717718
    },
    "points": {
      "mae": 3.6659978155672768,
      "rmse": 5.305830947390743,
      "r2": 0.4533787822297548,
      "spearman": 0.5959589467935817
    },
    "ranking": {
      "precision_at_10": 0.7041666666666666,
      "recall_at_10": 0.7041666666666666,
      "top10_overlap": 7.041666666666667,
      "ndcg_at_10": 0.7667886031311645
    }
  },
  "post_quali": {
    "top10": {
      "accuracy": 0.7599164926931107,
      "precision": 0.7450980392156863,
      "recall": 0.7916666666666666,
      "f1": 0.7676767676767676,
      "average_precision": 0.8216975234395676,
      "brier_score": 0.1649759944367739,
      "confusion_matrix": [
        [
          174,
          65
        ],
        [
          50,
          190
        ]
      ],
      "roc_auc": 0.8358960948396096
    },
    "podium": {
      "accuracy": 0.8997912317327766,
      "precision": 0.6224489795918368,
      "recall": 0.8472222222222222,
      "f1": 0.7176470588235294,
      "average_precision": 0.7264493152417262,
      "brier_score": 0.08003689333934849,
      "confusion_matrix": [
        [
          370,
          37
        ],
        [
          11,
          61
        ]
      ],
      "roc_auc": 0.9406906906906908
    },
    "points": {
      "mae": 2.9515166509418216,
      "rmse": 4.509501180518263,
      "r2": 0.6051458760758188,
      "spearman": 0.7008114450149965
    },
    "ranking": {
      "precision_at_10": 0.7666666666666666,
      "recall_at_10": 0.7666666666666666,
      "top10_overlap": 7.666666666666667,
      "ndcg_at_10": 0.8186446101027371
    }
  },
  "ranking": {}
}
```

## Temporal tuning

- Tuning trials: `C:\Users\bacht\Desktop\F1_Predictor\reports\metrics\temporal_tuning_trials.json`
- Tuning summary: `C:\Users\bacht\Desktop\F1_Predictor\reports\model_tuning_summary.csv`
- Tuning is comparative only; it does not replace the selected production artifacts automatically.

## Baselines

`reports/metrics/baseline_metrics.json` is present: True.

## Standings reconciliation

- File: `C:\Users\bacht\Desktop\F1_Predictor\reports\data_quality\standings_reconciliation.csv`
- Rows with non-zero difference: 27
- Known limitation: historical and current snapshots may not include full sprint/event points unless sprint result feeds are integrated.

## Leakage and schema risks

- Rolling features use `shift(1)` in `src/features.py`.
- Raw outcomes are excluded in `src/modeling.py`.
- Risk: current model features include season/round, which can encode time/calendar effects.
- Risk: original `points` semantics mixed race points with championship interpretation; this audit separates `grand_prix_points`, `sprint_points`, and `event_points`.
- Risk: 2026 future races need schedule-only driver-race rows; no future grid, qualifying, result, or points should be invented.

## Recommendations

1. Keep the current 2025 test metrics frozen as the evaluation baseline.
2. Use pre-quali models before qualifying and post-quali models only when real qualifying/grid data exists.
3. Integrate sprint result snapshots if available from Jolpica to improve official standings reconciliation.
4. Treat context/news adjustments as optional auditable overlays, not hidden model inputs.
5. Run temporal tuning before replacing any baseline model.
