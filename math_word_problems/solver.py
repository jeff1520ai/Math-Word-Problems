"""
math_word_problems.solver
--------------------------

Solve functions for math word problems.  Provides three modes:

* **Predefined** — uses scripted operation plans from ``operations.py``.
* **Mock** — naive strategy that adds the first two numbers found.
* **LLM** — delegates to the LangGraph agent in ``agent.py``.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Union

from .operations import PROBLEM_OPERATIONS
from .problems import PROBLEM_BY_TEXT
from .tools import calculator

Number = Union[int, float]


def _resolve_operand(value: Any, results: List[Any]) -> Any:
    """Resolve an operand that may reference a previous result."""
    if value == "prev":
        if not results:
            raise ValueError("'prev' used with no previous results")
        return results[-1]
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "result":
        return results[value[1]]
    return value


def solve_problem(
    problem_text: str, verbose: bool = False, mock: bool = False,
) -> Dict[str, Any]:
    """Solve a problem using predefined plans or mock mode."""
    state: Dict[str, Any] = {
        "problem": problem_text,
        "steps": [],
        "answer": None,
        "answer_numeric": None,
        "expected_answer": None,
        "status": "solving",
        "tool_calls": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "cost": 0.0,
        "constraint_violations": 0,
    }

    if problem_text in PROBLEM_BY_TEXT:
        state["expected_answer"] = PROBLEM_BY_TEXT[problem_text].expected_answer

    # Mock mode
    if mock:
        numbers = re.findall(r"\d+(?:\.\d+)?", problem_text)
        if len(numbers) < 2:
            state["status"] = "error"
            state["answer"] = "Could not find two numbers to add"
            return state
        a, b = float(numbers[0]), float(numbers[1])
        state["steps"].append({"type": "think", "content": f"I will add {a} and {b}.", "tool": None, "args": None, "result": None})
        args = {"operation": "add", "a": a, "b": b}
        state["steps"].append({"type": "act", "content": f"calculator(add, {a}, {b})", "tool": "calculator", "args": args, "result": None})
        result = calculator("add", a, b)
        state["steps"].append({"type": "observe", "content": str(result), "tool": None, "args": None, "result": result})
        state["tool_calls"] = 1
        state["steps"].append({"type": "think", "content": f"The sum of {a} and {b} is {result}.", "tool": None, "args": None, "result": None})
        state["answer"] = f"Answer: {result}"
        state["answer_numeric"] = result if isinstance(result, (int, float)) else None
        state["status"] = "solved" if isinstance(result, (int, float)) else "error"
        return state

    # Predefined mode
    ops = PROBLEM_OPERATIONS.get(problem_text)
    if not ops:
        state["status"] = "error"
        state["answer"] = "No solution plan available for this problem"
        return state

    results: List[Any] = []
    for idx, operation in enumerate(ops):
        op_name = operation["op"]
        a_val = _resolve_operand(operation["a"], results)
        b_val = _resolve_operand(operation["b"], results)

        intro = "First" if idx == 0 else "Next"
        verb_map = {"add": "add", "subtract": "subtract", "multiply": "multiply", "divide": "divide"}
        if op_name == "add":
            think = f"{intro}, I will add {a_val} and {b_val}."
        elif op_name == "subtract":
            think = f"{intro}, I will subtract {b_val} from {a_val}."
        elif op_name == "multiply":
            think = f"{intro}, I will multiply {a_val} by {b_val}."
        elif op_name == "divide":
            think = f"{intro}, I will divide {a_val} by {b_val}."
        else:
            think = f"{intro}, I will perform {op_name} on {a_val} and {b_val}."

        state["steps"].append({"type": "think", "content": think, "tool": None, "args": None, "result": None})
        args = {"operation": op_name, "a": a_val, "b": b_val}
        state["steps"].append({"type": "act", "content": f"calculator({op_name}, {a_val}, {b_val})", "tool": "calculator", "args": args, "result": None})
        res = calculator(op_name, a_val, b_val)
        state["steps"].append({"type": "observe", "content": str(res), "tool": None, "args": None, "result": res})
        results.append(res)

    state["tool_calls"] = len(ops)

    if results:
        final_val = results[-1]
        state["steps"].append({"type": "think", "content": f"Therefore, the final result is {final_val}.", "tool": None, "args": None, "result": None})
        state["answer"] = f"Answer: {final_val}"
        state["answer_numeric"] = float(final_val) if isinstance(final_val, (int, float)) else None
        if state["expected_answer"] is not None and state["answer_numeric"] is not None:
            if abs(state["answer_numeric"] - state["expected_answer"]) > 0.01:
                state["status"] = "error"
            else:
                state["status"] = "solved"
        else:
            state["status"] = "solved"
    else:
        state["answer"] = "No result"
        state["status"] = "error"

    return state


def solve_problem_llm(
    problem_text: str,
    phase: int = 1,
    verbose: bool = False,
    model_name: str = "claude-haiku-4-5-20251001",
) -> Dict[str, Any]:
    """Solve a problem using the LLM-powered agent."""
    from .agent import solve_with_agent
    return solve_with_agent(
        problem_text, phase=phase, model_name=model_name, verbose=verbose,
    )
