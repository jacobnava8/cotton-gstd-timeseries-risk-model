import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from src.config import Config
from src.data import load_data, basic_preprocess
from src.labels import add_transition_label
from src.features import add_ts_features
from src.modeling import build_xy, train_lgbm_groupkfold
from src.thresholding import pr_table, best_threshold_for_recall

def main(data_path: str):
    cfg = Config()

    df = load_data(data_path)
    df = basic_preprocess(df, cfg.run_col, cfg.time_col)

    # label
    df = add_transition_label(df, cfg.run_col, cfg.time_col, cfg.target_stage_col, cfg.horizon_days)
    y_col = f"y_trans_{cfg.horizon_days}"

    # features
    df = add_ts_features(df, cfg.run_col, cfg.time_col, cfg.base_vars, cfg.lags, cfg.roll_window)

    # choose model columns (keep it simple: drop ids + raw stage)
    drop_cols = [cfg.run_col, cfg.time_col, cfg.target_stage_col]
    X, y = build_xy(df.dropna(), y_col=y_col, drop_cols=drop_cols)
    groups = df.dropna()[cfg.run_col].to_numpy()

    model, cv_scores = train_lgbm_groupkfold(X, y, groups, n_splits=cfg.n_splits)
    print("\nCV scores:\n", cv_scores)

    # threshold selection on same set (simple)
    y_proba = model.predict_proba(X)[:, 1]
    pr_df = pr_table(y, y_proba)
    best = best_threshold_for_recall(pr_df, cfg.target_recall)
    thr = best["threshold"]
    print("\nBest threshold:", best)

    y_pred = (y_proba >= thr).astype(int)
    print("\nConfusion matrix:\n", confusion_matrix(y, y_pred))
    print("\nReport:\n", classification_report(y, y_pred))

if __name__ == "__main__":
    # example: python -m src.run_pipeline data/raw/your_dataset.csv
    import sys
    main(sys.argv[1])
