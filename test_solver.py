"""
tests/test_solver.py
---------------------

Unit tests for the problem solver defined in ``solver.py``.  These tests
exercise normal and mock modes, verify that known problems are solved
correctly, and check that the solver reports errors for unknown
problems.
"""
import math

import pytest

from solver import solve_problem, PROBLEM_OPERATIONS, PROBLEM_BY_TEXT


@pytest.mark.parametrize(
    "problem_text,expected_answer",
    [
        ("What is 47 plus 83?", 130),
        ("I have 3 baskets with 12 apples each. I eat 7. How many are left?", 29),
        ("You have $200. You buy a jacket for $60, shoes for $45, and then earn $80 mowing lawns. How much money do you have now?", 175),
        ("A farmer has 3 fields. Each produces 250 bushels. He sells half at $4/bushel and stores the rest. What did he earn?", 1500),
        ("A bakery makes 12 dozen cookies. They sell 40% on Monday and 25% of the remainder on Tuesday. How many are left after Tuesday?", 64.8),
    ],
)
def test_solve_problem_known(problem_text: str, expected_answer: float) -> None:
    """Solve known problems and check that the numeric answer matches the expected value."""
    state = solve_problem(problem_text)
    assert state["status"] == "solved", f"Solver did not solve problem: {problem_text}"
    assert state["answer_numeric"] is not None
    assert math.isclose(state["answer_numeric"], expected_answer, rel_tol=1e-2)
    # Check that the number of tool calls matches the predefined operations length
    if problem_text in PROBLEM_OPERATIONS:
        assert state["tool_calls"] == len(PROBLEM_OPERATIONS[problem_text])


def test_solve_problem_unknown() -> None:
    """The solver should report an error when asked to solve an unknown problem in normal mode."""
    problem_text = "What is the airspeed velocity of an unladen swallow?"
    state = solve_problem(problem_text)
    assert state["status"] == "error"
    assert "No solution plan" in state["answer"] or "no solution plan" in state["answer"].lower()


def test_mock_mode_addition() -> None:
    """In mock mode the solver adds the first two numbers found in the problem text."""
    problem_text = "I have 2 apples and 3 oranges. How many pieces of fruit do I have?"
    state = solve_problem(problem_text, mock=True)
    assert state["status"] == "solved"
    assert state["answer_numeric"] == 5.0
    # In mock mode, there should be exactly one calculator call
    assert state["tool_calls"] == 1