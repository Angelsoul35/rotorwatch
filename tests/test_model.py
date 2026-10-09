"""Tests for the metrics (RMSE, MAE, NASA score) and the RUL regressor."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rotorwatch.features import RUL_CAP, RUL_COL, UNIT_COL, clip_rul
from rotorwatch.model import RULRegressor, mae, metrics, nasa_score, rmse


def test_rmse_and_mae_match_hand_computation():
    y_true = [10.0, 20.0, 30.0]
    y_pred = [12.0, 18.0, 33.0]
    # errors: +2, -2, +3  ->  MAE = 7/3, RMSE = sqrt((4+4+9)/3)
    assert mae(y_true, y_pred) == pytest.approx(7.0 / 3.0)
    assert rmse(y_true, y_pred) == pytest.approx(math.sqrt(17.0 / 3.0))


def test_nasa_score_is_zero_only_for_perfect_predictions():
    assert nasa_score([50, 100], [50, 100]) == pytest.approx(0.0)


def test_nasa_score_penalises_late_predictions_more_than_early():
    # A late prediction (predict more life than there is) of size 10 must cost more
    # than an early prediction (predict less) of the same size, because a_late < a_early.
    late = nasa_score([100.0], [110.0])  # d = +10 -> exp(10/10) - 1
    early = nasa_score([100.0], [90.0])  # d = -10 -> exp(10/13) - 1
    assert late > early
    assert late == pytest.approx(math.expm1(1.0))
    assert early == pytest.approx(math.expm1(10.0 / 13.0))


def test_metrics_reports_the_three_numbers():
    m = metrics([1.0, 2.0], [1.0, 2.0])
    assert set(m) == {"rmse", "mae", "nasa_score"}
    assert m["rmse"] == 0.0 and m["mae"] == 0.0 and m["nasa_score"] == 0.0


def test_predict_before_fit_raises(synth_df):
    with pytest.raises(RuntimeError):
        RULRegressor().predict(synth_df)


def test_predictions_stay_within_the_valid_rul_band(synth_df):
    model = RULRegressor()
    model.fit(synth_df)
    pred = model.predict(synth_df)
    assert pred.min() >= 0.0
    assert pred.max() <= RUL_CAP


def test_model_beats_the_naive_mean_on_unseen_engines(synth_df):
    """On held-out engines the fitted model must beat predicting the mean RUL."""
    units = np.sort(synth_df[UNIT_COL].unique())
    train_units, test_units = units[:20], units[20:]
    train = synth_df[synth_df[UNIT_COL].isin(train_units)].reset_index(drop=True)
    test = synth_df[synth_df[UNIT_COL].isin(test_units)].reset_index(drop=True)

    model = RULRegressor()
    model.fit(train)
    model_rmse = model.evaluate(test)["rmse"]

    mean_rul = float(np.mean(clip_rul(train[RUL_COL])))
    naive_rmse = rmse(clip_rul(test[RUL_COL]), np.full(len(test), mean_rul))

    assert model_rmse < 0.6 * naive_rmse, (model_rmse, naive_rmse)
