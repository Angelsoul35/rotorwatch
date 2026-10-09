"""Functional tests for the real C-MAPSS benchmark.

They prove the evaluation is engine-disjoint (the leakage trap), that the model beats
the naive-mean baseline on engines it never trained on, that the held-out error tracks
the grouped-CV estimate (stable generalisation, not a lucky split), that the shuffle
control collapses the error back to the baseline (the edge is real learned signal), and
that the verdict and report render.
"""

from __future__ import annotations

import pytest
from sklearn.model_selection import GroupKFold

import rotorwatch.real_benchmark as rb
import rotorwatch.real_data as rd
from rotorwatch.data import make_run_to_failure
from rotorwatch.features import UNIT_COL
from rotorwatch.real_benchmark import (
    BenchmarkReport,
    CvResult,
    format_report,
    grouped_cv,
    run_benchmark,
    split_units,
)


def _report(heldout_rmse, cv_rmse, naive_rmse, shuffle_rmse):
    """Build a report with only the fields the verdict reads, for classifier tests."""
    return BenchmarkReport(
        n_rows=0, n_units=0, n_train_units=0, n_test_units=0, n_features=0, rul_cap=125,
        train_rmse=5.0, train_mae=4.0,
        heldout={"rmse": heldout_rmse, "mae": 0.0, "nasa_score": 0.0},
        cv=CvResult(5, cv_rmse, 0.0, 0.0, 0.0),
        naive={"rmse": naive_rmse, "mae": 0.0, "nasa_score": 0.0},
        shuffle_heldout={"rmse": shuffle_rmse, "mae": 0.0, "nasa_score": 0.0},
    )


@pytest.fixture(scope="module")
def report(real_df):
    return run_benchmark(real_df)


def test_split_holds_out_whole_engines(real_df):
    train, test = split_units(real_df, test_fraction=0.2, seed=0)
    train_units = set(train[UNIT_COL].unique().tolist())
    test_units = set(test[UNIT_COL].unique().tolist())
    # The split is by engine, never by row: disjoint sets that together cover the fleet.
    assert train_units.isdisjoint(test_units)
    assert train_units | test_units == set(real_df[UNIT_COL].unique().tolist())
    assert len(test_units) == 20  # 20% of 100 engines
    # No engine's cycles are split across the two sides.
    assert len(train) + len(test) == len(real_df)


def test_grouped_cv_folds_are_engine_disjoint(real_df):
    train, _ = split_units(real_df, seed=0)
    groups = train[UNIT_COL].to_numpy()
    for train_idx, val_idx in GroupKFold(n_splits=5).split(train, groups=groups):
        train_engines = set(groups[train_idx].tolist())
        val_engines = set(groups[val_idx].tolist())
        assert train_engines.isdisjoint(val_engines)


def test_run_benchmark_asserts_disjoint_engines(real_df):
    # The benchmark refuses to run on an overlapping split; the report's counts agree.
    r = run_benchmark(real_df)
    assert r.n_train_units + r.n_test_units == r.n_units == 100


def test_model_beats_naive_mean_on_held_out_engines(report):
    assert report.heldout["rmse"] < 0.6 * report.naive["rmse"]
    assert report.heldout["mae"] < report.naive["mae"]
    assert report.heldout["nasa_score"] < report.naive["nasa_score"]


def test_held_out_error_tracks_grouped_cv(report):
    # Stable generalisation: the single held-out split agrees with the CV estimate.
    assert report.cv_agreement < 0.25, report


def test_verdict_is_well_fit(report):
    assert report.verdict.startswith("well-fit"), report.verdict


def test_shuffle_control_collapses_to_the_baseline(report):
    # Permuting the labels destroys the signal: error jumps back up to the naive floor
    # and is far worse than the real model's held-out error.
    assert report.shuffle_heldout["rmse"] > 0.9 * report.naive["rmse"]
    assert report.shuffle_heldout["rmse"] > 1.5 * report.heldout["rmse"]


def test_grouped_cv_reports_all_metrics(real_df):
    train, _ = split_units(real_df, seed=0)
    cv = grouped_cv(train, n_splits=3)
    assert cv.n_splits == 3
    assert cv.rmse > 0 and cv.mae > 0 and cv.nasa_score > 0
    assert cv.rmse_std >= 0


def test_format_report_renders_the_evidence(report):
    text = format_report(report)
    assert "C-MAPSS" in text
    assert "held-out" in text and "naive" in text and "shuffle" in text
    assert "verdict:" in text
    assert f"{report.heldout['rmse']:.3f}" in text


def test_main_prints_the_report(report, monkeypatch, capsys):
    # Reuse the already-computed report so main() does not retrain the whole fleet.
    monkeypatch.setattr(rb, "run_benchmark", lambda: report)
    rb.main()
    assert "real-data benchmark" in capsys.readouterr().out


def test_run_benchmark_loads_the_vendored_data_by_default(monkeypatch):
    # With no frame passed, the benchmark must pull the vendored fleet via load_cmapss.
    small = make_run_to_failure(n_units=15, seed=9)
    monkeypatch.setattr(rd, "load_cmapss", lambda: small)
    r = run_benchmark(n_splits=3)
    assert r.n_units == 15


def test_verdict_underfit_when_it_misses_the_baseline():
    assert _report(40.0, 39.0, 42.0, 41.0).verdict.startswith("underfit")


def test_verdict_overfit_when_held_out_strays_from_cv():
    # Beats the baseline, but the held-out error is far from the CV estimate.
    assert _report(20.0, 10.0, 42.0, 41.0).verdict.startswith("overfit")


def test_verdict_well_fit_and_flags_a_non_collapsing_shuffle():
    collapsed = _report(20.0, 19.0, 42.0, 41.0)
    assert collapsed.verdict.startswith("well-fit")
    assert "shuffle control did not" not in collapsed.verdict

    not_collapsed = _report(20.0, 19.0, 42.0, 25.0)
    assert not_collapsed.verdict.startswith("well-fit")
    assert "shuffle control did not" in not_collapsed.verdict
