from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    # Columns
    run_col: str = "RUNNO"
    time_col: str = "DAP"      # your notebook uses DAP for ordering
    target_stage_col: str = "GSTD"

    # Labeling
    horizon_days: int = 7      # y_trans_7

    # Feature engineering
    base_vars: List[str] = None
    lags: List[int] = None
    roll_window: int = 7

    # CV
    n_splits: int = 5

    # Thresholding
    target_recall: float = 0.80

    def __post_init__(self):
        if self.base_vars is None:
            self.base_vars = [
                "LAID", "LAI",
                "NI8D", "NI9D",
                "SWTD", "SWXD", "SW1D", "SW2D", "SW3D",
                "TMAX", "TMIN", "SRAD", "PARD", "VPD",
                "GWAD", "CWAD",
            ]
        if self.lags is None:
            self.lags = [1, 3, 7]
