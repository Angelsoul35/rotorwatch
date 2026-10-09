"""Tests for loading and fetching the real C-MAPSS data.

They prove the vendored file is the real FD001 fleet, that RUL behaves like a
run-to-failure target (counts down to 0, never rises within an engine), and that the
fetch/paging path works end to end against a monkeypatched network.
"""

from __future__ import annotations

import io
import json
import re

import numpy as np
import pytest

import rotorwatch.real_data as rd
from rotorwatch.features import UNIT_COL
from rotorwatch.real_data import COLUMNS, fetch_cmapss, load_cmapss


def test_loads_the_real_fd001_fleet(real_df):
    assert list(real_df.columns) == COLUMNS
    assert real_df[UNIT_COL].nunique() == 100  # FD001 has exactly 100 engines
    assert len(real_df) == 20631  # the known FD001 train row count
    assert real_df[UNIT_COL].min() == 1 and real_df[UNIT_COL].max() == 100


def test_integer_columns_are_integers(real_df):
    assert real_df[UNIT_COL].dtype.kind == "i"
    assert real_df["time_in_cycles"].dtype.kind == "i"
    assert real_df["RUL"].dtype.kind == "i"


def test_rul_is_non_increasing_within_each_unit_and_reaches_zero(real_df):
    for unit, group in real_df.groupby(UNIT_COL):
        rul = group.sort_values("time_in_cycles")["RUL"].to_numpy()
        # Run-to-failure: RUL only ever counts down, and hits 0 at the failure cycle.
        assert np.all(np.diff(rul) <= 0), f"unit {unit} RUL increases"
        assert rul[-1] == 0, f"unit {unit} does not run to failure"
        assert rul[0] > 0, f"unit {unit} starts already failed"


def test_sensor_values_are_physical(real_df):
    # Sensor 1 is a flat channel at the FD001 sea-level inlet temperature.
    assert real_df["sensor_1"].std(ddof=0) == pytest.approx(0.0, abs=1e-9)
    # Sensor 2 (a core temperature) genuinely varies and sits in a plausible band.
    assert real_df["sensor_2"].std(ddof=0) > 0.1
    assert 600 < real_df["sensor_2"].min() and real_df["sensor_2"].max() < 700


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cmapss(tmp_path / "not_here.npz")


# --- the fetch/paging path, against a fake datasets-server -----------------------------

def _row(unit: int, cycle: int, rul: int) -> dict:
    """A C-MAPSS row with every column present (sensors filled with a marker value)."""
    r = {c: float(unit * 1000 + cycle) for c in COLUMNS}
    r[UNIT_COL] = unit
    r["time_in_cycles"] = cycle
    r["RUL"] = rul
    return {"row": r}


def _fake_urlopen(pages: dict[int, list[dict]]):
    """Return a urlopen stand-in that serves ``pages`` keyed by the URL's offset."""

    def opener(url, timeout=None):
        offset = int(re.search(r"offset=(\d+)", url).group(1))
        body = json.dumps({"rows": pages.get(offset, [])}).encode()
        return io.BytesIO(body)  # BytesIO is a context manager json.load can read

    return opener


def test_fetch_stops_at_unit_reset_and_vendors_rows(tmp_path, monkeypatch):
    pages = {
        0: [_row(1, 1, 5), _row(1, 2, 4), _row(2, 1, 3)],
        # Unit 3 is real FD001; then unit_number resets to 1 -> FD002 starts, so the
        # fetch must stop and must not vendor the reset row.
        100: [_row(3, 1, 2), _row(1, 1, 99)],
    }
    monkeypatch.setattr(rd, "urlopen", _fake_urlopen(pages))
    dest = tmp_path / "fd001.npz"

    n = fetch_cmapss(dest=dest)
    assert n == 4  # unit1 x2 + unit2 x1 + unit3 x1; the reset row is excluded

    out = load_cmapss(dest)
    assert out[UNIT_COL].tolist() == [1, 1, 2, 3]
    assert out["RUL"].tolist() == [5, 4, 3, 2]
    # Sensors round-tripped through the npz unchanged.
    assert out["sensor_1"].iloc[0] == pytest.approx(1001.0)


def test_fetch_respects_max_units(tmp_path, monkeypatch):
    pages = {0: [_row(1, 1, 9), _row(2, 1, 8), _row(3, 1, 7)]}
    monkeypatch.setattr(rd, "urlopen", _fake_urlopen(pages))
    dest = tmp_path / "capped.npz"

    n = fetch_cmapss(dest=dest, max_units=2)
    assert n == 2  # only engines 1 and 2; engine 3 is past the cap

    out = load_cmapss(dest)
    assert set(out[UNIT_COL].tolist()) == {1, 2}


def test_fetch_retries_then_succeeds(tmp_path, monkeypatch):
    calls = {"n": 0}
    good = _fake_urlopen({0: [_row(1, 1, 1)], 100: []})

    def flaky(url, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient network error")
        return good(url, timeout=timeout)

    monkeypatch.setattr(rd, "urlopen", flaky)
    monkeypatch.setattr(rd.time, "sleep", lambda *_: None)  # no real backoff wait
    n = fetch_cmapss(dest=tmp_path / "r.npz")
    assert n == 1 and calls["n"] >= 2  # it retried after the first failure
