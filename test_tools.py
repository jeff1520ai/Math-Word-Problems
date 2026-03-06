"""
tests/test_tools.py
--------------------

Unit tests for the calculator tool defined in ``tools.py``.  These
tests verify that the basic arithmetic operations return the expected
results and that error conditions, such as division by zero, are
handled gracefully.
"""
import math

import pytest

from tools import calculator


@pytest.mark.parametrize(
    "operation,a,b,expected",
    [
        ("add", 2, 3, 5),
        ("add", -1, 1, 0),
        ("subtract", 7, 5, 2),
        ("subtract", 0, 4, -4),
        ("multiply", 3, 4, 12),
        ("multiply", -2, 3, -6),
        ("divide", 10, 2, 5),
        ("divide", -9, 3, -3),
    ],
)
def test_calculator_basic_operations(operation, a, b, expected):
    """Test basic arithmetic operations of the calculator tool."""
    result = calculator(operation, a, b)
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_calculator_division_by_zero():
    """Test that division by zero returns an error message rather than raising."""
    result = calculator("divide", 5, 0)
    assert isinstance(result, str)
    assert "zero" in result.lower()