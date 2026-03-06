"""Unit tests for the solver and problem sets."""
import math
import pytest
from math_word_problems.solver import solve_problem
from math_word_problems.operations import PROBLEM_OPERATIONS
from math_word_problems.problems import (
    PROBLEM_BY_TEXT, PHASE1_PROBLEMS, PHASE2_PROBLEMS,
    PHASE3_PROBLEMS, PHASE3_UNSOLVABLE,
)


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
def test_solve_problem_known(problem_text, expected_answer):
    state = solve_problem(problem_text)
    assert state["status"] == "solved"
    assert state["answer_numeric"] is not None
    assert math.isclose(state["answer_numeric"], expected_answer, rel_tol=1e-2)
    if problem_text in PROBLEM_OPERATIONS:
        assert state["tool_calls"] == len(PROBLEM_OPERATIONS[problem_text])


def test_solve_problem_unknown():
    state = solve_problem("What is the airspeed velocity of an unladen swallow?")
    assert state["status"] == "error"
    assert "no solution plan" in state["answer"].lower()


def test_mock_mode_addition():
    state = solve_problem("I have 2 apples and 3 oranges. How many pieces of fruit?", mock=True)
    assert state["status"] == "solved"
    assert state["answer_numeric"] == 5.0
    assert state["tool_calls"] == 1


def test_phase1_has_50_problems():
    assert len(PHASE1_PROBLEMS) == 50


def test_phase2_has_30_problems():
    assert len(PHASE2_PROBLEMS) == 30


def test_phase3_has_30_problems():
    assert len(PHASE3_PROBLEMS) == 30


def test_phase3_unsolvable_has_nan_answers():
    for prob in PHASE3_UNSOLVABLE:
        assert math.isnan(prob.expected_answer)


def test_all_phase1_problems_have_operations():
    for prob in PHASE1_PROBLEMS:
        assert prob.problem in PROBLEM_OPERATIONS
