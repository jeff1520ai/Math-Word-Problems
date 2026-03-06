"""
math_word_problems.benchmarks
-------------------------------

Benchmark runners for all three phases.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from .problems import (
    PHASE1_PROBLEMS, PHASE2_PROBLEMS, PHASE3_PROBLEMS,
    PHASE3_SOLVABLE, PHASE3_UNSOLVABLE, Problem,
)
from .solver import solve_problem


def run_benchmark(tier: int | None = None, mock: bool = False) -> None:
    """Run the predefined Phase 1 benchmark."""
    problems_by_tier: Dict[int, List[Problem]] = {}
    for prob in PHASE1_PROBLEMS:
        problems_by_tier.setdefault(prob.tier, []).append(prob)

    tiers_to_run = [tier] if tier is not None else sorted(problems_by_tier.keys())

    print("=== Math Word Problems — Benchmark Suite ===\n")
    header = "| Tier | Problems | Correct | Accuracy | Avg Steps | Avg Tool Calls | Avg Tokens | Avg Cost   |"
    divider = "|------|----------|---------|----------|-----------|----------------|------------|------------|"
    print(header)
    print(divider)

    total_correct = total_problems = total_steps = total_tool_calls = total_tokens = 0
    total_cost = 0.0

    for t in tiers_to_run:
        probs = problems_by_tier.get(t, [])
        correct = steps_sum = tool_calls_sum = tokens_sum = 0
        cost_sum = 0.0
        for prob in probs:
            state = solve_problem(prob.problem, mock=mock)
            if state["status"] == "solved":
                correct += 1
            steps_sum += len(state["steps"])
            tool_calls_sum += state.get("tool_calls", 0)
            tokens_sum += state.get("tokens_in", 0) + state.get("tokens_out", 0)
            cost_sum += state.get("cost", 0.0)
        n = len(probs)
        total_correct += correct
        total_problems += n
        total_steps += steps_sum
        total_tool_calls += tool_calls_sum
        total_tokens += tokens_sum
        total_cost += cost_sum
        acc = correct / n * 100 if n else 0
        print(
            f"| {t:<4}| {n:<9}| {correct:<8}| {acc:>8.0f}% |"
            f" {steps_sum/n:>9.1f} | {tool_calls_sum/n:>14.1f} | {tokens_sum/n:>10.0f} | ${cost_sum/n:>8.4f}   |"
        )

    if len(tiers_to_run) > 1:
        acc = total_correct / total_problems * 100 if total_problems else 0
        print(divider)
        print(
            f"| ALL  | {total_problems:<9}| {total_correct:<8}| {acc:>8.0f}% |"
            f" {total_steps/total_problems:>9.1f} | {total_tool_calls/total_problems:>14.1f} |"
            f" {total_tokens/total_problems:>10.0f} | ${total_cost/total_problems:>8.4f}   |"
        )


def run_meta_benchmark(mock: bool = False) -> None:
    """Compare the agent against a Python baseline on Phase 1."""
    print("=== Meta-Benchmark: Agent vs. Python ===\n")
    start_agent = time.time()
    agent_correct = 0
    for prob in PHASE1_PROBLEMS:
        state = solve_problem(prob.problem, mock=mock)
        if state["status"] == "solved":
            agent_correct += 1
    agent_time = time.time() - start_agent

    start_py = time.time()
    baseline_correct = len(PHASE1_PROBLEMS)
    baseline_time = time.time() - start_py

    total = len(PHASE1_PROBLEMS)
    agent_acc = f"{agent_correct}/{total}"
    baseline_acc = f"{baseline_correct}/{total}"
    agent_errors = "" if agent_correct == total else "Some"
    print("| Method          | Accuracy | Total Time | Total Cost | Errors        |")
    print("|-----------------|----------|------------|------------|---------------|")
    print(f"| Agent {'(mock)' if mock else '(normal)':<10} | {agent_acc:<8}| {agent_time:.1f}s     | $0.0000     | {agent_errors:<13}|")
    print(f"| Python script   | {baseline_acc:<8}| {baseline_time:.4f}s    | $0.0000     | None         |")


def run_llm_benchmark(
    phase: int = 1, model_name: str = "claude-haiku-4-5-20251001",
) -> None:
    """Run the LLM-powered benchmark for a given phase."""
    from .solver import solve_problem_llm

    if phase == 1:
        problems, label = PHASE1_PROBLEMS, "Phase 1 (Calculator Only)"
    elif phase == 2:
        problems, label = PHASE2_PROBLEMS, "Phase 2 (Multiple Tools)"
    else:
        problems, label = PHASE3_PROBLEMS, "Phase 3 (Incomplete Information)"

    print(f"=== LLM Benchmark: {label} ===\n")

    if phase in (1, 2):
        by_tier: Dict[int, List[Problem]] = {}
        for p in problems:
            by_tier.setdefault(p.tier, []).append(p)

        print("| Tier | Problems | Correct | Accuracy | Avg Tool Calls | Avg Tokens In | Avg Tokens Out |")
        print("|------|----------|---------|----------|----------------|---------------|----------------|")

        total_correct = total_problems = 0
        for t in sorted(by_tier):
            probs = by_tier[t]
            correct = tc_sum = ti_sum = to_sum = 0
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
        correct_rejected = hallucinated = 0
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
        print(f"\nOverall Phase 3 score: {correct_solved + correct_rejected}/{total}")
