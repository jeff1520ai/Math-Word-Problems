"""Unit tests for all tools."""
import math
import pytest
from math_word_problems.tools import calculator, unit_converter, percentage_calculator, date_calculator


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
    result = calculator(operation, a, b)
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_calculator_division_by_zero():
    result = calculator("divide", 5, 0)
    assert isinstance(result, str)
    assert "zero" in result.lower()


def test_calculator_unsupported_op():
    result = calculator("power", 2, 3)
    assert isinstance(result, str)
    assert "unsupported" in result.lower()


@pytest.mark.parametrize(
    "value,from_unit,to_unit,expected",
    [
        (1, "miles", "km", 1.60934),
        (1, "km", "miles", 1 / 1.60934),
        (1, "feet", "meters", 0.3048),
        (1, "lbs", "kg", 0.453592),
        (100, "fahrenheit", "celsius", (100 - 32) * 5 / 9),
        (0, "celsius", "fahrenheit", 32),
        (2.5, "cups_flour", "grams", 300.0),
    ],
)
def test_unit_converter(value, from_unit, to_unit, expected):
    result = unit_converter(value, from_unit, to_unit)
    assert math.isclose(result, expected, rel_tol=1e-3)


def test_unit_converter_unsupported():
    result = unit_converter(1, "parsecs", "lightyears")
    assert isinstance(result, str)
    assert "unsupported" in result.lower()


def test_percentage_of():
    assert math.isclose(percentage_calculator("of", 200, 15), 30.0)


def test_percentage_change():
    assert math.isclose(percentage_calculator("change", 100, 20), 120.0)


def test_percentage_change_negative():
    assert math.isclose(percentage_calculator("change", 100, -10), 90.0)


def test_percentage_what_percent():
    assert math.isclose(percentage_calculator("what_percent", 25, 200), 12.5)


def test_percentage_what_percent_zero():
    result = percentage_calculator("what_percent", 10, 0)
    assert isinstance(result, str)
    assert "zero" in result.lower()


def test_date_add_days():
    assert date_calculator("add_days", "2025-01-01", days=10) == "2025-01-11"


def test_date_subtract_days():
    assert date_calculator("subtract_days", "2025-03-15", days=15) == "2025-02-28"


def test_date_days_between():
    assert date_calculator("days_between", "2025-01-01", date2="2025-01-31") == "30"


def test_date_day_of_week():
    assert date_calculator("day_of_week", "2025-01-01") == "Wednesday"


def test_date_invalid_format():
    result = date_calculator("add_days", "not-a-date", days=5)
    assert "error" in result.lower()
