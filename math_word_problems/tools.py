"""
math_word_problems.tools
-------------------------

Calculator, unit converter, percentage calculator, and date calculator
tools used by the math word problem solver.
"""
from __future__ import annotations

from typing import Callable, Any

try:
    from langgraph import tool  # type: ignore
except ImportError:
    def tool(func: Callable[..., Any]) -> Callable[..., Any]:
        return func


@tool
def calculator(operation: str, a: float, b: float) -> float | str:
    """Performs a basic arithmetic operation.

    Args:
        operation: One of ``"add"``, ``"subtract"``, ``"multiply"``,
            or ``"divide"``.
        a: The first operand.
        b: The second operand.

    Returns:
        The numerical result, or an error message string.
    """
    op = operation.lower().strip()
    try:
        a_float = float(a)
        b_float = float(b)
    except (TypeError, ValueError):
        return "Error: Operands must be numbers"

    if op == "add":
        return a_float + b_float
    if op == "subtract":
        return a_float - b_float
    if op == "multiply":
        return a_float * b_float
    if op == "divide":
        if b_float == 0:
            return "Error: Division by zero"
        return a_float / b_float
    return f"Error: Unsupported operation '{operation}'"


# --- Conversion tables for unit_converter ---

_DISTANCE_CONVERSIONS = {
    ("miles", "km"): 1.60934,
    ("km", "miles"): 1 / 1.60934,
    ("feet", "meters"): 0.3048,
    ("meters", "feet"): 1 / 0.3048,
    ("inches", "cm"): 2.54,
    ("cm", "inches"): 1 / 2.54,
}

_WEIGHT_CONVERSIONS = {
    ("lbs", "kg"): 0.453592,
    ("kg", "lbs"): 1 / 0.453592,
    ("oz", "grams"): 28.3495,
    ("grams", "oz"): 1 / 28.3495,
    ("cups_flour", "grams"): 120.0,
    ("grams", "cups_flour"): 1 / 120.0,
    ("cups_sugar", "grams"): 200.0,
    ("grams", "cups_sugar"): 1 / 200.0,
    ("cups_water", "grams"): 236.588,
    ("grams", "cups_water"): 1 / 236.588,
}

_ALL_CONVERSIONS = {**_DISTANCE_CONVERSIONS, **_WEIGHT_CONVERSIONS}


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> float | str:
    """Converts between common units.

    Supported conversions:
        Distance: miles <-> km, feet <-> meters, inches <-> cm
        Weight: lbs <-> kg, oz <-> grams, cups_flour/cups_sugar/cups_water <-> grams
        Temperature: fahrenheit <-> celsius

    Args:
        value: The numeric value to convert.
        from_unit: The source unit.
        to_unit: The target unit.

    Returns:
        The converted value, or an error message.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "Error: Value must be a number"

    fu = from_unit.lower().strip()
    tu = to_unit.lower().strip()

    if fu in ("fahrenheit", "f") and tu in ("celsius", "c"):
        return (v - 32) * 5 / 9
    if fu in ("celsius", "c") and tu in ("fahrenheit", "f"):
        return v * 9 / 5 + 32

    key = (fu, tu)
    if key in _ALL_CONVERSIONS:
        return v * _ALL_CONVERSIONS[key]

    return f"Error: Unsupported conversion from '{from_unit}' to '{to_unit}'"


@tool
def percentage_calculator(operation: str, base: float, percent: float) -> float | str:
    """Performs percentage operations.

    Args:
        operation: One of ``"of"``, ``"change"``, or ``"what_percent"``.
        base: The base number.
        percent: The percentage value.

    Returns:
        The numeric result, or an error message.
    """
    try:
        b = float(base)
        p = float(percent)
    except (TypeError, ValueError):
        return "Error: base and percent must be numbers"

    op = operation.lower().strip()

    if op == "of":
        return b * (p / 100)
    if op == "change":
        return b * (1 + p / 100)
    if op == "what_percent":
        if p == 0:
            return "Error: Division by zero"
        return (b / p) * 100

    return f"Error: Unsupported percentage operation '{operation}'"


@tool
def date_calculator(operation: str, date: str, days: int = 0, date2: str = "") -> str:
    """Performs date arithmetic.

    Args:
        operation: One of ``"add_days"``, ``"subtract_days"``,
            ``"days_between"``, or ``"day_of_week"``.
        date: A date string in ``YYYY-MM-DD`` format.
        days: Number of days for add/subtract operations.
        date2: A second date string for ``days_between``.

    Returns:
        The resulting date string or a descriptive answer.
    """
    from datetime import datetime, timedelta

    op = operation.lower().strip()
    try:
        dt = datetime.strptime(date.strip(), "%Y-%m-%d")
    except ValueError:
        return f"Error: Invalid date format '{date}'. Use YYYY-MM-DD."

    if op == "add_days":
        result = dt + timedelta(days=int(days))
        return result.strftime("%Y-%m-%d")
    if op == "subtract_days":
        result = dt - timedelta(days=int(days))
        return result.strftime("%Y-%m-%d")
    if op == "days_between":
        try:
            dt2 = datetime.strptime(date2.strip(), "%Y-%m-%d")
        except ValueError:
            return f"Error: Invalid second date format '{date2}'. Use YYYY-MM-DD."
        delta = abs((dt2 - dt).days)
        return str(delta)
    if op == "day_of_week":
        return dt.strftime("%A")

    return f"Error: Unsupported date operation '{operation}'"
