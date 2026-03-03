import pandas as pd

def add_ts_features(
    df: pd.DataFrame,
    run_col: str,
    time_col: str,
    base_vars: list[str],
    lags: list[int],
    roll_window: int
) -> pd.DataFrame:
    """
    Leakage-safe TS features computed within RUNNO using ONLY past values:
    - lag features
    - deltas
    - rolling mean/std with shift(1)
    """
    df = df.copy().sort_values([run_col, time_col]).reset_index(drop=True)
    base_vars = [v for v in base_vars if v in df.columns]

    def _per_run(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        for v in base_vars:
            for L in lags:
                g[f"{v}_lag{L}"] = g[v].shift(L)
            g[f"{v}_d1"] = g[v] - g[v].shift(1)
            g[f"{v}_d7"] = g[v] - g[v].shift(7)
            g[f"{v}_roll{roll_window}_mean"] = g[v].shift(1).rolling(roll_window, min_periods=1).mean()
            g[f"{v}_roll{roll_window}_std"]  = g[v].shift(1).rolling(roll_window, min_periods=1).std()
        return g

    return df.groupby(run_col, group_keys=False).apply(_per_run)
