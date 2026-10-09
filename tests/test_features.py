"""Tests for feature selection, RUL clipping, per-engine smoothing, and scaling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rotorwatch.features import (
    RUL_CAP,
    FeaturePipeline,
    add_rolling_means,
    clip_rul,
    select_informative,
)


def test_clip_rul_applies_the_piecewise_linear_cap():
    out = clip_rul([10, 124, 125, 200, 361])
    assert out.tolist() == [10.0, 124.0, 125.0, 125.0, 125.0]
    assert clip_rul([200], cap=50).tolist() == [50.0]
    assert RUL_CAP == 125


def test_select_informative_drops_the_dead_channels_on_real_data(real_df):
    selected = select_informative(real_df)
    # These FD001 channels are flat and carry no wear signal -> must be dropped.
    for dead in ["op_setting_3", "sensor_1", "sensor_5", "sensor_16", "sensor_18", "sensor_19"]:
        assert dead not in selected
    # These genuinely vary -> must be kept.
    for live in ["sensor_2", "sensor_3", "sensor_4", "sensor_11"]:
        assert live in selected


def test_select_informative_keys_off_variance_not_names():
    df = pd.DataFrame(
        {
            "op_setting_1": [1.0, 1.0, 1.0],  # constant -> dropped
            "sensor_1": [0.0, 1.0, 2.0],  # varies -> kept
            "sensor_2": [5.0, 5.0, 5.0],  # constant -> dropped
        }
    )
    assert select_informative(df, candidate_cols=["op_setting_1", "sensor_1", "sensor_2"]) == [
        "sensor_1"
    ]


def test_rolling_mean_is_trailing_and_never_bleeds_across_engines():
    df = pd.DataFrame(
        {
            "unit_number": [1, 1, 1, 2, 2],
            "sensor_2": [10.0, 20.0, 30.0, 100.0, 200.0],
        }
    )
    out = add_rolling_means(df, ["sensor_2"], window=2)
    rolled = out["sensor_2_roll2"].tolist()
    # Trailing, min_periods=1: first value of each engine equals its raw reading.
    assert rolled[0] == 10.0
    assert rolled[1] == 15.0  # mean(10, 20)
    assert rolled[2] == 25.0  # mean(20, 30)
    # Engine 2 starts fresh: its first smoothed value is NOT averaged with engine 1.
    assert rolled[3] == 100.0
    assert rolled[4] == 150.0  # mean(100, 200), not touching engine 1


def test_pipeline_selects_smooths_and_standardises(real_df):
    pipe = FeaturePipeline(rolling_window=5)
    x = pipe.fit_transform(real_df)
    # 17 informative base channels, each with a rolling-mean partner.
    assert len(pipe.feature_names_) == 34
    assert x.shape == (len(real_df), 34)
    # Standardised: per-column mean ~0 and std ~1 on the fitted data.
    assert np.allclose(x.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(x.std(axis=0), 1.0, atol=1e-6)


def test_pipeline_without_rolling_uses_only_base_channels(real_df):
    pipe = FeaturePipeline(rolling_window=None)
    pipe.fit(real_df)
    assert pipe.feature_names_ == pipe.base_cols_
    assert all(not name.endswith("_roll5") for name in pipe.feature_names_)


def test_pipeline_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        FeaturePipeline().transform(pd.DataFrame({"unit_number": [1]}))
