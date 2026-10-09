from __future__ import annotations

import pytest

from rotorwatch.data import make_run_to_failure
from rotorwatch.real_data import load_cmapss


@pytest.fixture(scope="session")
def real_df():
    """The vendored real C-MAPSS FD001 fleet."""
    return load_cmapss()


@pytest.fixture(scope="session")
def synth_df():
    """A small synthetic run-to-failure fleet (fast, no network)."""
    return make_run_to_failure(n_units=30, seed=3)
