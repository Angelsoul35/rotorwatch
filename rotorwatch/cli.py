"""RotorWatch command-line interface."""

from __future__ import annotations

import argparse

from .data import make_run_to_failure
from .real_benchmark import format_report, run_benchmark


def cmd_demo(args) -> None:
    """Train and evaluate on a synthetic run-to-failure fleet (fast, no network)."""
    print("1) Generating a synthetic run-to-failure fleet ...")
    df = make_run_to_failure(n_units=args.units, seed=args.seed)
    print(f"   {df['unit_number'].nunique()} engines, {len(df)} cycles")

    print("2) Benchmarking (engine-disjoint split + grouped CV + controls) ...")
    report = run_benchmark(df, n_splits=args.folds, seed=args.seed)

    print(f"   held-out  : RMSE {report.heldout['rmse']:.3f}  MAE {report.heldout['mae']:.3f}"
          f"  NASA {report.heldout['nasa_score']:.1f}")
    print(f"   grouped CV: RMSE {report.cv.rmse:.3f}  (std {report.cv.rmse_std:.3f})")
    print(f"   baseline  : RMSE {report.naive['rmse']:.3f}  (naive mean)")
    print(f"   shuffle   : RMSE {report.shuffle_heldout['rmse']:.3f}  (labels permuted)")
    print(f"   verdict   : {report.verdict}")


def cmd_benchmark(args) -> None:
    """Run the full benchmark on the vendored real C-MAPSS FD001 data."""
    print(format_report(run_benchmark(seed=args.seed)))


def main() -> None:
    parser = argparse.ArgumentParser(description="RotorWatch CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo", help="train + evaluate on a synthetic fleet (fast)")
    p.add_argument("--units", type=int, default=40)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("benchmark", help="full benchmark on the real C-MAPSS FD001 data")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
