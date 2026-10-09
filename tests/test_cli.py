"""Tests for the command-line interface."""

from __future__ import annotations

import pytest

import rotorwatch.cli as cli


def test_demo_runs_end_to_end(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["rotorwatch", "demo", "--units", "20", "--folds", "3"])
    cli.main()
    out = capsys.readouterr().out
    assert "synthetic run-to-failure fleet" in out
    assert "held-out" in out
    assert "verdict" in out


def test_benchmark_command_prints_report(monkeypatch, capsys, real_df):
    # Point the real benchmark at a tiny synthetic run so the command is fast but the
    # whole cmd_benchmark -> run_benchmark -> format_report path is exercised.
    from rotorwatch.data import make_run_to_failure
    from rotorwatch.real_benchmark import run_benchmark

    small = make_run_to_failure(n_units=15, seed=5)
    monkeypatch.setattr(cli, "run_benchmark", lambda seed=0: run_benchmark(small, n_splits=3))
    monkeypatch.setattr("sys.argv", ["rotorwatch", "benchmark"])
    cli.main()
    assert "real-data benchmark" in capsys.readouterr().out


def test_cli_requires_a_subcommand(monkeypatch):
    monkeypatch.setattr("sys.argv", ["rotorwatch"])
    with pytest.raises(SystemExit):
        cli.main()
