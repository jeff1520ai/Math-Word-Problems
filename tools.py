"""
tools.py
-----------

This module defines the calculator tool used by the math word problem solver.
The calculator provides four basic arithmetic operations: addition, subtraction,
multiplication and division.  It is intentionally simple — no exponentiation,
roots or advanced mathematics are supported.  Division by zero returns an
error message rather than raising an exception.

The functions in this module are decorated with a ``tool`` decorator if
available from ``langgraph``.  When the ``langgraph`` package is not
installed (as may be the case in a minimal test environment), a fallback
decorator is defined that simply returns the original function untouched.

Usage::

    from tools import calculator
    result = calculator("add", 2, 3)  # returns 5

"""
from __future__ import annotations

from typing import Callable, Any

try:
    # Attempt to import the tool decorator from langgraph.  In production
    # environments this will annotate the function for use inside a LangGraph
    # node.  If langgraph is not available (e.g. offline testing), we
    # gracefully fall back to a no-op decorator.
    from langgraph import tool  # type: ignore
except ImportError:  # pragma: no cover
    def tool(func: Callable[..., Any]) -> Callable[..., Any]:
        """Fallback decorator used when ``langgraph`` is unavailable.

        The fallback decorator simply returns the original function without
        modification.  It preserves the wrapped function's signature and
        docstring so that tooling and documentation remain intact.

        Args:
            func: The function being decorated.

        Returns:
            The original function.
        """
        return func


@tool
def calculator(operation: str, a: float, b: float) -> float | str:
    """Performs a basic arithmetic operation.

    This tool implements four arithmetic operations: addition (``add``),
    subtraction (``subtract``), multiplication (``multiply``) and division
    (``divide``).  All other operations yield a string error message.  When
    dividing by zero the function returns an error message rather than
    raising an exception.

    Args:
        operation: The operation to perform.  Must be one of ``"add"``,
            ``"subtract"``, ``"multiply"`` or ``"divide"``.
        a: The first operand.
        b: The second operand.

    Returns:
        The numerical result of the operation, or an error message string
        if the operation is unsupported or division by zero occurs.

    Examples:

        >>> calculator("add", 1, 2)
        3
        >>> calculator("divide", 4, 2)
        2.0
        >>> calculator("divide", 4, 0)
        'Error: Division by zero'
    """
    op = operation.lower().strip()
    # Ensure the operands are floats for consistency.  The tool still accepts
    # integers, but converting to float avoids accidental integer division in
    # Python 2 semantics (even though Python 3 is used here).
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
        The converted value, or an error message if the conversion is
        unsupported.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "Error: Value must be a number"

    fu = from_unit.lower().strip()
    tu = to_unit.lower().strip()

    # Temperature conversions are special-cased
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
        operation: One of ``"of"`` (what is X% of Y), ``"change"``
            (Y changed by X%), or ``"what_percent"`` (X is what % of Y).
        base: The base number.
        percent: The percentage value.

    Returns:
        The numeric result, or an error message if the operation is
        unsupported.
    """
    try:
        b = float(base)
        p = float(percent)
    except (TypeError, ValueError):
        return "Error: base and percent must be numbers"

    op = operation.lower().strip()

    if op == "of":
        # What is percent% of base?
        return b * (p / 100)
    if op == "change":
        # base changed by percent%
        return b * (1 + p / 100)
    if op == "what_percent":
        # base is what percent of percent (percent acts as the whole)
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
