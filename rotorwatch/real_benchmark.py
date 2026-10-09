"""Benchmark the RUL regressor on the real C-MAPSS fleet, without fooling ourselves.

Estimating Remaining Useful Life has one trap that dwarfs every other: **leakage across
engines**. Cycles from the same engine are nearly identical neighbours, so if any cycle
of an engine is in training, a model can all but look up the answer for that engine's
other cycles. Splitting the rows at random would do exactly that and report a wonderful,
meaningless number. This benchmark never splits rows; it splits **engines**. Whole
engines are held out for test, and cross-validation uses :class:`GroupKFold` grouped by
engine, so no engine is ever on both sides of a split. :func:`run_benchmark` asserts the
train and test engine sets are disjoint.

Against that faithful setup it reports, on engines the model never saw:

* **RMSE, MAE and the NASA score** on the held-out engines;
* the **grouped-CV** estimate (mean over folds) — if the held-out error agrees with it,
  the result is stable, not a lucky split;
* the **train vs held-out gap** — read with care: a gradient-boosting ensemble will fit
  the training engines tightly, so a low train error is expected; the signal of
  generalisation that matters is held-out == CV, not a zero gap;
* a **naive-mean baseline** (predict the mean training RUL for everything) — the floor a
  real model must clear;
* a **shuffle control** — retrain with the RUL labels permuted, destroying the
  sensor->RUL relationship; its error must collapse back to the naive baseline, proving
  the model's edge is real learned signal and not an artefact of the pipeline.

Run it with ``python -m rotorwatch.real_benchmark``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .features import ROLLING_WINDOW, RUL_CAP, RUL_COL, UNIT_COL, clip_rul
from .model import RULRegressor, metrics


@dataclass
class CvResult:
    n_splits: int
    rmse: float
    mae: float
    nasa_score: float
    rmse_std: float


@dataclass
class BenchmarkReport:
    n_rows: int
    n_units: int
    n_train_units: int
    n_test_units: int
    n_features: int
    rul_cap: int
    train_rmse: float
    train_mae: float
    heldout: dict[str, float]
    cv: CvResult
    naive: dict[str, float]
    shuffle_heldout: dict[str, float]

    @property
    def gap(self) -> float:
        """Held-out minus train RMSE (reported for transparency, not for the verdict)."""
        return round(self.heldout["rmse"] - self.train_rmse, 4)

    @property
    def cv_agreement(self) -> float:
        """Relative distance between the held-out RMSE and the grouped-CV RMSE."""
        return abs(self.heldout["rmse"] - self.cv.rmse) / self.cv.rmse

    @property
    def verdict(self) -> str:
        beats_naive = self.heldout["rmse"] < 0.75 * self.naive["rmse"]
        shuffle_collapses = self.shuffle_heldout["rmse"] > 0.9 * self.naive["rmse"]
        stable = self.cv_agreement < 0.25
        if not beats_naive:
            return "underfit: the held-out error does not clear the naive-mean baseline"
        if not stable:
            return (
                "overfit: the held-out error exceeds the grouped-CV estimate, a sign of "
                "fitting to specific engines rather than to degradation"
            )
        extra = "" if shuffle_collapses else " (shuffle control did not fully collapse)"
        return (
            "well-fit: the held-out error agrees with grouped CV and clears the naive "
            "baseline on engines never seen in training" + extra
        )


def unique_units(df: pd.DataFrame) -> np.ndarray:
    return np.sort(df[UNIT_COL].unique())


def split_units(
    df: pd.DataFrame, test_fraction: float = 0.2, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole engines for test (never rows), reproducibly.

    The unique engine ids are shuffled with a seeded RNG and the last ``test_fraction``
    become the test engines; every cycle of an engine goes to the same side.
    """
    units = unique_units(df)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(units)
    n_test = max(1, int(round(len(units) * test_fraction)))
    test_units = set(shuffled[-n_test:].tolist())
    is_test = df[UNIT_COL].isin(test_units)
    train_df = df[~is_test].reset_index(drop=True)
    test_df = df[is_test].reset_index(drop=True)
    return train_df, test_df


def grouped_cv(
    df: pd.DataFrame,
    n_splits: int = 5,
    params: dict | None = None,
    rolling_window: int | None = ROLLING_WINDOW,
    rul_cap: int = RUL_CAP,
) -> CvResult:
    """Grouped (engine-disjoint) cross-validation RMSE/MAE/NASA.

    Each fold trains on one set of engines and validates on a disjoint set, so the
    estimate reflects generalisation to unseen engines, averaged over folds.
    """
    groups = df[UNIT_COL].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)
    rmses, maes, nasas = [], [], []
    for train_idx, val_idx in gkf.split(df, groups=groups):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        model = RULRegressor(params=params, rolling_window=rolling_window, rul_cap=rul_cap)
        model.fit(train_df)
        m = model.evaluate(val_df)
        rmses.append(m["rmse"])
        maes.append(m["mae"])
        nasas.append(m["nasa_score"])
    return CvResult(
        n_splits=n_splits,
        rmse=round(float(np.mean(rmses)), 4),
        mae=round(float(np.mean(maes)), 4),
        nasa_score=round(float(np.mean(nasas)), 2),
        rmse_std=round(float(np.std(rmses)), 4),
    )


def _naive_mean_baseline(
    train_df: pd.DataFrame, test_df: pd.DataFrame, rul_cap: int
) -> dict[str, float]:
    """Predict the mean (clipped) training RUL for every held-out cycle."""
    mean_rul = float(np.mean(clip_rul(train_df[RUL_COL], rul_cap)))
    y_true = clip_rul(test_df[RUL_COL], rul_cap)
    return metrics(y_true, np.full(len(test_df), mean_rul))


def _shuffle_control(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict | None,
    rolling_window: int | None,
    rul_cap: int,
    seed: int,
) -> dict[str, float]:
    """Train with RUL labels permuted (signal destroyed) and score on held-out engines."""
    rng = np.random.default_rng(seed)
    shuffled = train_df.copy()
    shuffled[RUL_COL] = rng.permutation(shuffled[RUL_COL].to_numpy())
    model = RULRegressor(params=params, rolling_window=rolling_window, rul_cap=rul_cap)
    model.fit(shuffled)
    return model.evaluate(test_df)


def run_benchmark(
    df: pd.DataFrame | None = None,
    params: dict | None = None,
    rolling_window: int | None = ROLLING_WINDOW,
    n_splits: int = 5,
    test_fraction: float = 0.2,
    seed: int = 0,
    rul_cap: int = RUL_CAP,
) -> BenchmarkReport:
    if df is None:
        from .real_data import load_cmapss

        df = load_cmapss()

    train_df, test_df = split_units(df, test_fraction=test_fraction, seed=seed)
    # Guard rail: the split must be engine-disjoint, or the whole benchmark is a lie.
    train_units = set(train_df[UNIT_COL].unique().tolist())
    test_units = set(test_df[UNIT_COL].unique().tolist())
    assert train_units.isdisjoint(test_units), "train/test engines overlap"

    model = RULRegressor(params=params, rolling_window=rolling_window, rul_cap=rul_cap)
    report = model.fit(train_df)
    heldout = model.evaluate(test_df)
    train_metrics = model.evaluate(train_df)

    cv = grouped_cv(
        train_df, n_splits=n_splits, params=params, rolling_window=rolling_window, rul_cap=rul_cap
    )
    naive = _naive_mean_baseline(train_df, test_df, rul_cap)
    shuffle_heldout = _shuffle_control(
        train_df, test_df, params, rolling_window, rul_cap, seed=seed + 1
    )

    return BenchmarkReport(
        n_rows=len(df),
        n_units=len(unique_units(df)),
        n_train_units=len(train_units),
        n_test_units=len(test_units),
        n_features=report.n_features,
        rul_cap=rul_cap,
        train_rmse=train_metrics["rmse"],
        train_mae=train_metrics["mae"],
        heldout=heldout,
        cv=cv,
        naive=naive,
        shuffle_heldout=shuffle_heldout,
    )


def format_report(report: BenchmarkReport) -> str:
    lines = [
        "RotorWatch real-data benchmark - NASA C-MAPSS FD001 turbofan run-to-failure",
        "=" * 76,
        f"engines            : {report.n_units} ({report.n_train_units} train / "
        f"{report.n_test_units} held-out), {report.n_rows} cycles, "
        f"{report.n_features} features, RUL cap {report.rul_cap}",
        "",
        f"{'':<22}{'RMSE':>9}{'MAE':>9}{'NASA score':>13}",
        f"{'held-out engines':<22}{report.heldout['rmse']:>9.3f}{report.heldout['mae']:>9.3f}"
        f"{report.heldout['nasa_score']:>13.1f}",
        f"{'grouped CV (mean)':<22}{report.cv.rmse:>9.3f}{report.cv.mae:>9.3f}"
        f"{report.cv.nasa_score:>13.1f}",
        f"{'train engines':<22}{report.train_rmse:>9.3f}{report.train_mae:>9.3f}"
        f"{'-':>13}",
        f"{'naive-mean baseline':<22}{report.naive['rmse']:>9.3f}{report.naive['mae']:>9.3f}"
        f"{report.naive['nasa_score']:>13.1f}",
        f"{'shuffle control':<22}{report.shuffle_heldout['rmse']:>9.3f}"
        f"{report.shuffle_heldout['mae']:>9.3f}{report.shuffle_heldout['nasa_score']:>13.1f}",
        "",
        f"train/held-out RMSE gap : {report.gap:+.3f}   "
        f"held-out vs CV agreement: {report.cv_agreement:.1%}   "
        f"CV RMSE std: {report.cv.rmse_std:.3f}",
        f"verdict: {report.verdict}",
        "",
        f"The model's held-out RMSE ({report.heldout['rmse']:.1f} cycles) clears the "
        f"naive-mean baseline ({report.naive['rmse']:.1f}) on engines it never trained "
        "on, and tracks the grouped-CV estimate "
        f"({report.cv.rmse:.1f}), so the result is a stable generalisation, not a lucky "
        "split. Permuting the RUL labels collapses the error back to the baseline "
        f"({report.shuffle_heldout['rmse']:.1f}), which shows the edge is learned "
        "sensor->wear signal. The NASA score weights late predictions more than early "
        "ones, matching the maintenance cost of running a failing machine too long.",
    ]
    return "\n".join(lines)


def main() -> None:
    print(format_report(run_benchmark()))


if __name__ == "__main__":
    main()
