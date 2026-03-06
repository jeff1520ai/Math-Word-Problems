"""
benchmarks.py
----------------

This module provides convenience functions for running the benchmark
suite and the meta-benchmark from the command line.  It supports all
three phases of the project:

  Phase 1 — Calculator only, 50 problems across 5 tiers
  Phase 2 — Multiple tools, 30 problems (tier 6)
  Phase 3 — Incomplete information, 30 problems (20 solvable + 10 unsolvable)

When using ``--llm`` mode, problems are solved by the LLM-powered agent.
Otherwise the pre-defined solution plans are used (Phase 1 only).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Any, Dict, List

from problems import (
    PHASE1_PROBLEMS,
    PHASE2_PROBLEMS,
    PHASE3_PROBLEMS,
    PHASE3_SOLVABLE,
    PHASE3_UNSOLVABLE,
    PROBLEMS,
    Problem,
)
from solver import run_benchmark, run_meta_benchmark, solve_problem


def run_llm_benchmark(
    phase: int = 1,
    model_name: str = "claude-haiku-4-5-20251001",
) -> None:
    """Run the LLM-powered benchmark for a given phase.

    Args:
        phase: Which phase to benchmark (1, 2, or 3).
        model_name: The Claude model to use.
    """
    from solver import solve_problem_llm

    if phase == 1:
        problems = PHASE1_PROBLEMS
        label = "Phase 1 (Calculator Only)"
    elif phase == 2:
        problems = PHASE2_PROBLEMS
        label = "Phase 2 (Multiple Tools)"
    else:
        problems = PHASE3_PROBLEMS
        label = "Phase 3 (Incomplete Information)"

    print(f"=== LLM Benchmark: {label} ===\n")

    if phase in (1, 2):
        # Group by tier
        by_tier: Dict[int, List[Problem]] = {}
        for p in problems:
            by_tier.setdefault(p.tier, []).append(p)

        print("| Tier | Problems | Correct | Accuracy | Avg Tool Calls | Avg Tokens In | Avg Tokens Out |")
        print("|------|----------|---------|----------|----------------|---------------|----------------|")

        total_correct = 0
        total_problems = 0
        for t in sorted(by_tier):
            probs = by_tier[t]
            correct = 0
            tc_sum = 0
            ti_sum = 0
            to_sum = 0
            for prob in probs:
                state = solve_problem_llm(prob.problem, phase=phase, model_name=model_name)
                if state["status"] in ("solved", "correctly_rejected"):
                    correct += 1
                tc_sum += state.get("tool_calls", 0)
                ti_sum += state.get("tokens_in", 0)
                to_sum += state.get("tokens_out", 0)
            n = len(probs)
            total_correct += correct
            total_problems += n
            acc = correct / n * 100 if n else 0
            print(
                f"| {t:<4} | {n:<8} | {correct:<7} | {acc:>7.0f}% |"
                f" {tc_sum/n:>14.1f} | {ti_sum/n:>13.0f} | {to_sum/n:>14.0f} |"
            )
        if len(by_tier) > 1:
            acc = total_correct / total_problems * 100 if total_problems else 0
            print(f"| ALL  | {total_problems:<8} | {total_correct:<7} | {acc:>7.0f}% |")

    else:
        # Phase 3: separate solvable and unsolvable
        print("--- Solvable Problems ---")
        correct_solved = 0
        for prob in PHASE3_SOLVABLE:
            state = solve_problem_llm(prob.problem, phase=3, model_name=model_name)
            if state["status"] == "solved":
                correct_solved += 1
            else:
                print(f"  MISS: {prob.problem[:60]}... (got {state['status']})")

        print(f"\nSolvable accuracy: {correct_solved}/{len(PHASE3_SOLVABLE)}")

        print("\n--- Unsolvable Problems ---")
        correct_rejected = 0
        hallucinated = 0
        for prob in PHASE3_UNSOLVABLE:
            state = solve_problem_llm(prob.problem, phase=3, model_name=model_name)
            if state["status"] == "correctly_rejected":
                correct_rejected += 1
            else:
                hallucinated += 1
                print(f"  HALLUCINATION: {prob.problem[:60]}... (answered {state.get('answer_numeric')})")

        print(f"\nCorrect rejection rate: {correct_rejected}/{len(PHASE3_UNSOLVABLE)}")
        print(f"Hallucination rate: {hallucinated}/{len(PHASE3_UNSOLVABLE)}")

        total = len(PHASE3_SOLVABLE) + len(PHASE3_UNSOLVABLE)
        total_correct = correct_solved + correct_rejected
        print(f"\nOverall Phase 3 score: {total_correct}/{total}")


def main(argv: List[str] | None = None) -> int:
    """Entry point for the benchmark CLI."""
    parser = argparse.ArgumentParser(
        description="Run benchmark suites for the Math Word Problems project."
    )
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4, 5], help="Run a specific tier only")
    parser.add_argument("--mock", action="store_true", help="Use mock mode (naive solver)")
    parser.add_argument("--meta", action="store_true", help="Run the meta-benchmark (agent vs Python)")
    parser.add_argument("--llm", action="store_true", help="Use the LLM-powered agent")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=1, help="Phase to benchmark (with --llm)")
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001", help="Claude model (with --llm)")
    args = parser.parse_args(argv)

    if args.llm:
        run_llm_benchmark(phase=args.phase, model_name=args.model)
    elif args.meta:
        run_meta_benchmark(mock=args.mock)
    else:
        run_benchmark(tier=args.tier, mock=args.mock)
    return 0


if __name__ == "__main__":
    sys.exit(main())