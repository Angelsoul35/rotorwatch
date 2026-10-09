"""Turn raw run-to-failure sensor logs into a model-ready feature matrix.

Each row of the C-MAPSS data is one operating cycle of one engine: the operating
settings and 21 sensor channels measured that cycle, plus the Remaining Useful Life
(RUL) — how many cycles remain before that engine fails. Estimating RUL is therefore a
tabular regression, once three things are handled correctly:

1. **Drop dead channels.** Several sensors never move across the whole fleet (they
   report a constant), so they carry no information about wear. A zero-variance column
   is dropped before training. Selection is fit on the *training* engines only, so the
   held-out engines never influence which features exist.
2. **Clip the RUL target.** Early in an engine's life the sensors look healthy and RUL
   is simply not predictable from them; only once degradation begins do the sensors
   carry a usable signal. The standard C-MAPSS target is therefore a *piecewise-linear*
   RUL, capped at a constant (see :func:`clip_rul`). This is the documented method, not
   a simplification.
3. **Smooth and scale.** Sensor readings are noisy cycle to cycle. A short trailing
   rolling mean *per engine* denoises each channel using only the current and earlier
   cycles of the same engine (no leakage, no bleed between engines). Features are then
   standardised so channels on very different scales are comparable.

The oil & gas reading: an engine here is any monitored rotating machine on a platform —
a gas-turbine driver, a centrifugal compressor, a crude-export pump. The same pipeline
estimates cycles (or running hours) to failure from its vibration, temperature, pressure
and flow channels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

UNIT_COL = "unit_number"
TIME_COL = "time_in_cycles"
RUL_COL = "RUL"
OP_COLS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
CANDIDATE_COLS = OP_COLS + SENSOR_COLS

# Standard C-MAPSS piecewise-linear RUL cap. Early-life RUL is not observable from the
# sensors, so the target is held flat at this value until degradation brings true RUL
# below it; from there the target decreases linearly to 0 at failure.
RUL_CAP = 125

# A column whose standard deviation falls below this is treated as constant and dropped.
VARIANCE_THRESHOLD = 1e-6

# Window (in cycles) of the per-engine trailing rolling mean used to denoise sensors.
ROLLING_WINDOW = 5


def clip_rul(rul: np.ndarray | pd.Series, cap: int = RUL_CAP) -> np.ndarray:
    """Return the piecewise-linear RUL target: ``min(rul, cap)``.

    Clipping encodes the fact that a healthy engine 300 cycles from failure and one
    200 cycles from failure look identical to the sensors; predicting either exact
    number is impossible and training on it only adds label noise. The capped target
    asks the model the answerable question — "is failure near, and how near?" — and is
    the target used throughout the C-MAPSS literature.
    """
    return np.minimum(np.asarray(rul, dtype=float), float(cap))


def select_informative(
    df: pd.DataFrame,
    candidate_cols: list[str] | None = None,
    threshold: float = VARIANCE_THRESHOLD,
) -> list[str]:
    """Return the candidate columns that actually vary (std above ``threshold``).

    Constant sensors carry no information about degradation, so they are removed. This
    must be fit on the training engines alone to avoid leaking the held-out engines into
    the feature set.
    """
    cols = candidate_cols if candidate_cols is not None else CANDIDATE_COLS
    return [c for c in cols if float(df[c].std(ddof=0)) > threshold]


def add_rolling_means(
    df: pd.DataFrame, cols: list[str], window: int = ROLLING_WINDOW
) -> pd.DataFrame:
    """Add a per-engine trailing rolling mean for each column in ``cols``.

    The rolling mean is grouped by :data:`UNIT_COL`, so it never averages across two
    different engines, and it is trailing (``min_periods=1``), so the value at cycle *t*
    uses only cycles ``<= t`` of the same engine. Both properties keep it leakage-free.
    New columns are named ``{col}_roll{window}``.
    """
    out = df.copy()
    grouped = out.groupby(UNIT_COL, sort=False)
    for col in cols:
        out[f"{col}_roll{window}"] = (
            grouped[col].transform(lambda s: s.rolling(window, min_periods=1).mean())
        )
    return out


class FeaturePipeline:
    """Select informative channels, add per-engine smoothing, and standardise.

    Fit on the training engines only (``fit``/``fit_transform``); the fitted column
    choice and scaler are then reused unchanged on the held-out engines (``transform``),
    which is what keeps the evaluation faithful.
    """

    def __init__(
        self,
        rolling_window: int | None = ROLLING_WINDOW,
        variance_threshold: float = VARIANCE_THRESHOLD,
    ) -> None:
        self.rolling_window = rolling_window
        self.variance_threshold = variance_threshold
        self.base_cols_: list[str] = []
        self.feature_names_: list[str] = []
        self.scaler_: StandardScaler | None = None

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.rolling_window:
            return add_rolling_means(df, self.base_cols_, self.rolling_window)
        return df

    def fit(self, df: pd.DataFrame) -> FeaturePipeline:
        self.base_cols_ = select_informative(df, CANDIDATE_COLS, self.variance_threshold)
        if self.rolling_window:
            self.feature_names_ = self.base_cols_ + [
                f"{c}_roll{self.rolling_window}" for c in self.base_cols_
            ]
        else:
            self.feature_names_ = list(self.base_cols_)
        engineered = self._engineer(df)
        self.scaler_ = StandardScaler().fit(engineered[self.feature_names_].to_numpy())
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.scaler_ is None:
            raise RuntimeError("FeaturePipeline is not fitted; call fit() first")
        engineered = self._engineer(df)
        return self.scaler_.transform(engineered[self.feature_names_].to_numpy())

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)
