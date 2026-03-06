"""
CLI entry point: python -m math_word_problems
"""
from __future__ import annotations

import argparse
import sys
from typing import List

from .benchmarks import run_benchmark, run_meta_benchmark, run_llm_benchmark
from .solver import solve_problem, solve_problem_llm


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Solve math word problems with a think-act-observe agent.",
    )
    parser.add_argument("problem", nargs="?", help="The word problem to solve")
    parser.add_argument("--verbose", action="store_true", help="Show step-by-step reasoning")
    parser.add_argument("--benchmark", action="store_true", help="Run the predefined benchmark")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4, 5], help="Run a specific tier")
    parser.add_argument("--meta", action="store_true", help="Compare agent vs Python baseline")
    parser.add_argument("--mock", action="store_true", help="Use mock mode (naive solver)")
    parser.add_argument("--llm", action="store_true", help="Use the LLM-powered agent")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=1, help="Phase (default: 1)")
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001", help="Claude model")
    args = parser.parse_args(argv)

    if args.benchmark or args.tier:
        if args.llm:
            run_llm_benchmark(phase=args.phase, model_name=args.model)
        else:
            run_benchmark(tier=args.tier, mock=args.mock)
        return 0

    if args.meta:
        run_meta_benchmark(mock=args.mock)
        return 0

    if args.problem:
        if args.llm:
            state = solve_problem_llm(
                args.problem, phase=args.phase, verbose=args.verbose,
                model_name=args.model,
            )
        else:
            state = solve_problem(args.problem, verbose=args.verbose, mock=args.mock)

        if args.verbose:
            print(f"Problem: {state['problem']}\n")
            for i, step in enumerate(state["steps"], 1):
                if step["type"] == "think":
                    print(f"Step {i} -- THINK: {step['content']}")
                elif step["type"] == "act":
                    print(f"Step {i} -- ACT:   {step['content']}")
                elif step["type"] == "observe":
                    print(f"Step {i} -- OBSERVE: {step['content']}")
            print()
            print(state["answer"])
            if state["expected_answer"] is not None:
                correct = "+" if state["status"] == "solved" else "X"
                print(f"Correct: {correct} (expected: {state['expected_answer']})")
        else:
            if state["status"] in ("solved", "correctly_rejected"):
                print(state["answer"])
            else:
                print(f"Error: {state['answer']}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
