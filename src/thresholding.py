import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

def pr_table(y_true, y_proba) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    return pd.DataFrame({
        "threshold": np.append(thresholds, 1.0),
        "precision": precision,
        "recall": recall
    }).sort_values("threshold")

def best_threshold_for_recall(pr_df: pd.DataFrame, target_recall: float) -> dict:
    """
    Same logic as your notebook:
    among thresholds meeting recall target, choose highest precision.
    """
    candidates = pr_df[pr_df["recall"] >= target_recall]
    if len(candidates) == 0:
        return {"threshold": float("nan"), "precision": float("nan"), "recall": float("nan")}
    best = candidates.sort_values(["precision", "threshold"], ascending=[False, True]).iloc[0]
    return {"threshold": float(best["threshold"]), "precision": float(best["precision"]), "recall": float(best["recall"])}
