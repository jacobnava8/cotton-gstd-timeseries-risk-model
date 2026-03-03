import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    # adjust if you use parquet
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError("Unsupported file type. Use .csv or .parquet")

def basic_preprocess(df: pd.DataFrame, run_col: str, time_col: str) -> pd.DataFrame:
    # match notebook behavior: sort within run by time
    df = df.copy()
    df = df.sort_values([run_col, time_col]).reset_index(drop=True)

    # remove duplicate columns if present (common after merges)
    df.columns = [c.strip() for c in df.columns]

    return df

def find_exact_duplicate_columns(df: pd.DataFrame):
    """
    Compact version of your notebook duplicate-column check.
    Returns list of (col, duplicate_of).
    """
    dup_pairs = []
    cols = list(df.columns)
    seen = {}
    for c in cols:
        key = pd.util.hash_pandas_object(df[c], index=False).sum()
        if key in seen:
            # confirm exact match
            c0 = seen[key]
            if df[c].equals(df[c0]):
                dup_pairs.append((c, c0))
        else:
            seen[key] = c
    return dup_pairs
