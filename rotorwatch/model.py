"""The RUL regressor and its metrics (RMSE, MAE, and the NASA asymmetric score).

The model is a histogram gradient-boosting regressor (``HistGradientBoostingRegressor``)
on the tabular sensor features. Gradient boosting is the right tool here: the RUL signal
is a set of non-linear, interacting thresholds on noisy sensor channels (a temperature
creeping up *and* a pressure drifting down together mean wear), which tree ensembles
capture directly, with no assumption about the shape of each channel's effect. The
histogram variant is fast on the tens of thousands of cycles in the fleet, and its
built-in regularisation (a learning rate, a leaf-count cap, an L2 penalty, and a floor on
samples per leaf) keeps the fit from memorising individual engines — which matters
because the evaluation holds out whole engines.

Three metrics are reported:

* **RMSE** — root-mean-square error in cycles; penalises large misses.
* **MAE** — mean absolute error in cycles; the plain "how many cycles off, typically".
* **NASA score** — the asymmetric scoring function from the PHM08 C-MAPSS challenge. It
  punishes *late* predictions (the model thinks more life remains than there is — you
  run a failing machine) far more than *early* ones (you service too soon). That
  asymmetry is exactly the maintenance trade-off, so it is the metric the field reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import ROLLING_WINDOW, RUL_CAP, RUL_COL, FeaturePipeline, clip_rul

# Regularised defaults, chosen a priori (not tuned against the held-out engines). The
# learning rate, leaf cap, L2 penalty, and min-samples-per-leaf trade a little training
# accuracy for generalisation to engines the model has never seen.
_DEFAULT_PARAMS: dict[str, Any] = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "max_depth": None,
    "early_stopping": False,
    "random_state": 42,
}


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def rmse(y_true, y_pred) -> float:
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))


def nasa_score(y_true, y_pred, a_early: float = 13.0, a_late: float = 10.0) -> float:
    """The C-MAPSS asymmetric score (lower is better; 0 is perfect).

    For each instance let ``d = predicted_RUL - true_RUL``. An early prediction
    (``d < 0``, the model is conservative) scores ``exp(-d / a_early) - 1``; a late
    prediction (``d >= 0``, the model overestimates remaining life) scores
    ``exp(d / a_late) - 1``. Because ``a_late < a_early``, a late call of the same size
    costs more — the maintenance asymmetry: servicing early wastes money, running a
    failing machine risks the asset. The total is summed over all scored instances.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    penalty = np.where(d < 0, np.expm1(-d / a_early), np.expm1(d / a_late))
    return float(np.sum(penalty))


def metrics(y_true, y_pred) -> dict[str, float]:
    """RMSE, MAE, and the NASA score as a dict, rounded for reporting."""
    return {
        "rmse": round(rmse(y_true, y_pred), 4),
        "mae": round(mae(y_true, y_pred), 4),
        "nasa_score": round(nasa_score(y_true, y_pred), 2),
    }


@dataclass
class TrainingReport:
    n_rows: int
    n_features: int
    feature_names: list[str]
    params: dict[str, Any]


class RULRegressor:
    """Estimate Remaining Useful Life from a run-to-failure sensor frame.

    ``fit`` learns the feature selection + scaler (on the given engines only) and the
    gradient-boosting model against the clipped RUL target. ``predict`` returns RUL
    estimates clamped to ``[0, cap]`` — a machine cannot have negative remaining life,
    and the capped target makes anything above the cap meaningless.
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        rolling_window: int | None = ROLLING_WINDOW,
        rul_cap: int = RUL_CAP,
    ) -> None:
        self.params = {**_DEFAULT_PARAMS, **(params or {})}
        self.rul_cap = rul_cap
        self.pipeline = FeaturePipeline(rolling_window=rolling_window)
        self.model: HistGradientBoostingRegressor | None = None

    def fit(self, df: pd.DataFrame) -> TrainingReport:
        x = self.pipeline.fit_transform(df)
        y = clip_rul(df[RUL_COL], self.rul_cap)
        self.model = HistGradientBoostingRegressor(**self.params)
        self.model.fit(x, y)
        return TrainingReport(
            n_rows=len(df),
            n_features=len(self.pipeline.feature_names_),
            feature_names=list(self.pipeline.feature_names_),
            params=self.params,
        )

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model is not trained; call fit() first")
        x = self.pipeline.transform(df)
        pred = self.model.predict(x)
        return np.clip(pred, 0.0, float(self.rul_cap))

    def evaluate(self, df: pd.DataFrame) -> dict[str, float]:
        """Predict on ``df`` and score against its clipped true RUL."""
        y_true = clip_rul(df[RUL_COL], self.rul_cap)
        return metrics(y_true, self.predict(df))
