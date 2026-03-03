import numpy as np
import pandas as pd

def label_transition_within_horizon(group: pd.DataFrame, stage_col: str, horizon: int) -> pd.Series:
    """
    y_trans_h(t)=1 if ANY stage change occurs in (t, t+h]
    Exactly matches your notebook definition.
    """
    s = group[stage_col].to_numpy()
    n = len(s)
    y = np.zeros(n, dtype=int)
    for i in range(n):
        j_end = min(n - 1, i + horizon)
        if i + 1 <= j_end and np.any(s[i+1:j_end+1] != s[i]):
            y[i] = 1
    return pd.Series(y, index=group.index)

def add_transition_label(df: pd.DataFrame, run_col: str, time_col: str, stage_col: str, horizon: int) -> pd.DataFrame:
    df = df.copy().sort_values([run_col, time_col]).reset_index(drop=True)
    y_name = f"y_trans_{horizon}"
    df[y_name] = (
        df.groupby(run_col, group_keys=False)
          .apply(lambda g: label_transition_within_horizon(g, stage_col, horizon))
    )
    return df
