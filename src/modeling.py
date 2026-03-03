import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score

def build_xy(df: pd.DataFrame, y_col: str, drop_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    X = df.drop(columns=[y_col] + drop_cols, errors="ignore")
    y = df[y_col].astype(int).to_numpy()
    return X, y

def train_lgbm_groupkfold(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    params: dict | None = None
):
    if params is None:
        params = dict(
            objective="binary",
            learning_rate=0.05,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            n_estimators=2000,
            random_state=42
        )

    gkf = GroupKFold(n_splits=n_splits)
    fold_scores = []

    # train final model on full data (your notebook trains + evaluates; here we keep it simple)
    model = lgb.LGBMClassifier(**params)

    # quick CV scoring (Average Precision + ROC-AUC) like your notebook style
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups), 1):
        model_fold = lgb.LGBMClassifier(**params)
        model_fold.fit(
            X.iloc[tr], y[tr],
            eval_set=[(X.iloc[va], y[va])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        proba = model_fold.predict_proba(X.iloc[va])[:, 1]
        fold_scores.append({
            "fold": fold,
            "ap": float(average_precision_score(y[va], proba)),
            "roc_auc": float(roc_auc_score(y[va], proba)),
        })

    # fit final
    model.fit(X, y)
    return model, pd.DataFrame(fold_scores)
