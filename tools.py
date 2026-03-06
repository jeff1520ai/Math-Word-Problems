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
