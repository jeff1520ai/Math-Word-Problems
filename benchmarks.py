"""
benchmarks.py
----------------

This module provides convenience functions for running the benchmark
suite and the meta‑benchmark from the command line.  It simply
delegates to the corresponding functions defined in ``solver.py``.

Users can invoke this script directly to run the full benchmark,
benchmark a specific tier, or compare the agent against the Python
baseline.  See the project README for detailed usage examples.
"""
from __future__ import annotations

import argparse
from typing import List

from solver import run_benchmark, run_meta_benchmark


def main(argv: List[str] | None = None) -> int:
    """Entry point for the benchmark CLI.

    This function parses command‑line arguments and dispatches to the
    appropriate benchmark routine.  The available options mirror those
    of ``solver.py``:

    ``--tier N``
        Only run problems in the specified tier (1–5).  If omitted, all
        tiers are included.

    ``--mock``
        Use mock mode, which exercises the pipeline without relying on
        any external language model.  In mock mode the agent employs a
        naive strategy of adding the first two numbers it encounters.

    ``--meta``
        Run the meta‑benchmark, which compares the agent against a plain
        Python baseline.  If omitted, the standard benchmark is run.
    """
    parser = argparse.ArgumentParser(description="Run benchmark suites for the Math Word Problems project.")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4, 5], help="Run a specific tier only")
    parser.add_argument("--mock", action="store_true", help="Use mock mode (naive solver)")
    parser.add_argument("--meta", action="store_true", help="Run the meta-benchmark instead of the normal benchmark")
    args = parser.parse_args(argv)
    if args.meta:
        run_meta_benchmark(mock=args.mock)
    else:
        run_benchmark(tier=args.tier, mock=args.mock)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())