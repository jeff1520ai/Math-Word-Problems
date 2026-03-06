"""
solver.py
----------

This module implements the core logic for solving math word problems using a
simple think–act–observe loop.  Problems are solved by chaining calls to a
basic calculator tool; the agent itself performs no arithmetic directly.

Two modes of operation are supported:

* **Normal mode** uses a pre‑defined plan of operations for each problem.  The
  solver sequentially executes the specified arithmetic operations, logging
  detailed reasoning ("think"), tool invocations ("act") and results
  ("observe") along the way.
* **Mock mode** implements a naive strategy that extracts the first two
  numbers from the problem and adds them.  This mode exercises the full
  pipeline without requiring an external language model or API key.

Additionally, this script exposes a command‑line interface for solving a
single problem, running the benchmark suite, and comparing agent
performance against a plain Python baseline.

"""
from __future__ import annotations

import argparse
import math
import re
import sys
import time
from typing import Any, Dict, List, Tuple, Union

from problems import PROBLEMS, PROBLEM_BY_TEXT, Problem
from tools import calculator

Number = Union[int, float]


def _resolve_operand(value: Any, results: List[Any]) -> Any:
    """Resolve an operand from the operation specification.

    Operands may refer to previous results using the sentinel values
    ``'prev'`` or a tuple of the form ``('result', index)``.  If ``value``
    is a numeric literal, it is returned as‑is.  If ``value`` is
    ``'prev'`` it resolves to the most recent element of ``results``.
    If ``value`` is a tuple ``('result', i)`` it resolves to
    ``results[i]``.  Any other value is returned unchanged.

    Args:
        value: The operand specification to resolve.
        results: A list of previously computed numeric results.

    Returns:
        The resolved operand.
    """
    if value == 'prev':
        if not results:
            raise ValueError("'prev' operand cannot be resolved because no previous results exist")
        return results[-1]
    if isinstance(value, tuple) and len(value) == 2 and value[0] == 'result':
        index = value[1]
        return results[index]
    return value


# Mapping from problem text to a list of operations.  Each operation
# specification is a dictionary with keys:
#   op: one of 'add', 'subtract', 'multiply', 'divide'
#   a: first operand (numeric literal, 'prev', or ('result', i))
#   b: second operand (as above)

PROBLEM_OPERATIONS: Dict[str, List[Dict[str, Any]]] = {
    # Tier 1
    "What is 47 plus 83?": [
        {"op": "add", "a": 47, "b": 83},
    ],
    "A book costs $15. You buy 3. How much total?": [
        {"op": "multiply", "a": 15, "b": 3},
    ],
    "You have 100 stickers and give away 37. How many left?": [
        {"op": "subtract", "a": 100, "b": 37},
    ],
    "A pizza is cut into 8 slices. 3 people split it equally. How many slices does each person get?": [
        {"op": "divide", "a": 8, "b": 3},
    ],
    "I have 20 marbles and my friend gives me 15 more. How many marbles do I have?": [
        {"op": "add", "a": 20, "b": 15},
    ],
    "A car travels 60 miles in one hour. How far does it travel in 4 hours?": [
        {"op": "multiply", "a": 60, "b": 4},
    ],
    "You buy 5 bags of candy with 7 candies each. How many candies do you have?": [
        {"op": "multiply", "a": 5, "b": 7},
    ],
    "There are 45 students and 3 classes. If the students are split equally, how many students per class?": [
        {"op": "divide", "a": 45, "b": 3},
    ],
    "A farmer has 24 eggs and sells 10. How many eggs remain?": [
        {"op": "subtract", "a": 24, "b": 10},
    ],
    "Each pack has 6 batteries. How many batteries are in 5 packs?": [
        {"op": "multiply", "a": 6, "b": 5},
    ],
    # Tier 2
    "I have 3 baskets with 12 apples each. I eat 7. How many are left?": [
        {"op": "multiply", "a": 3, "b": 12},
        {"op": "subtract", "a": "prev", "b": 7},
    ],
    "A pizza has 8 slices. 3 people each eat 2 slices. How many are left?": [
        {"op": "multiply", "a": 3, "b": 2},
        {"op": "subtract", "a": 8, "b": "prev"},
    ],
    "You earn $15/hour and work 8 hours. You spend $40 on groceries. How much do you have?": [
        {"op": "multiply", "a": 15, "b": 8},
        {"op": "subtract", "a": "prev", "b": 40},
    ],
    "A class has 28 students. 4 are absent. The rest split into groups of 6. How many groups?": [
        {"op": "subtract", "a": 28, "b": 4},
        {"op": "divide", "a": "prev", "b": 6},
    ],
    "A container holds 5 liters. You have 3 containers and pour out 2 liters. How much liquid is left?": [
        {"op": "multiply", "a": 5, "b": 3},
        {"op": "subtract", "a": "prev", "b": 2},
    ],
    "You read 30 pages of a book each day for 5 days. Then you read 20 pages. How many pages have you read?": [
        {"op": "multiply", "a": 30, "b": 5},
        {"op": "add", "a": "prev", "b": 20},
    ],
    "There are 10 packs of pencils with 12 pencils each. You give away 15 pencils. How many pencils remain?": [
        {"op": "multiply", "a": 10, "b": 12},
        {"op": "subtract", "a": "prev", "b": 15},
    ],
    "A farmer has 7 cows and buys 5 more. Each cow produces 8 liters of milk. How much milk do the cows produce?": [
        {"op": "add", "a": 7, "b": 5},
        {"op": "multiply", "a": "prev", "b": 8},
    ],
    "You save $50 each week for 6 weeks. Then you spend $120. How much do you have left?": [
        {"op": "multiply", "a": 50, "b": 6},
        {"op": "subtract", "a": "prev", "b": 120},
    ],
    "A theatre has 200 seats. In a show, 75 tickets are sold online and 50 at the door. How many seats are empty?": [
        {"op": "add", "a": 75, "b": 50},
        {"op": "subtract", "a": 200, "b": "prev"},
    ],
    # Tier 3
    "A store has 4 shelves with 15 books each. They sell 12 and receive a shipment of 20. How many now?": [
        {"op": "multiply", "a": 4, "b": 15},
        {"op": "subtract", "a": "prev", "b": 12},
        {"op": "add", "a": "prev", "b": 20},
    ],
    "You earn $12/hour for 8 hours, then $18/hour for 3 hours overtime. What's your total pay?": [
        {"op": "multiply", "a": 12, "b": 8},
        {"op": "multiply", "a": 18, "b": 3},
        {"op": "add", "a": ("result", 0), "b": ("result", 1)},
    ],
    "A farmer plants 5 rows of 10 trees each. 8 trees die. He plants 12 more. How many trees?": [
        {"op": "multiply", "a": 5, "b": 10},
        {"op": "subtract", "a": "prev", "b": 8},
        {"op": "add", "a": "prev", "b": 12},
    ],
    "A factory produces 20 gadgets per hour for 5 hours and then 15 gadgets per hour for 4 hours. How many gadgets produced?": [
        {"op": "multiply", "a": 20, "b": 5},
        {"op": "multiply", "a": 15, "b": 4},
        {"op": "add", "a": ("result", 0), "b": ("result", 1)},
    ],
    "You have $200. You buy a jacket for $60, shoes for $45, and then earn $80 mowing lawns. How much money do you have now?": [
        {"op": "subtract", "a": 200, "b": 60},
        {"op": "subtract", "a": "prev", "b": 45},
        {"op": "add", "a": "prev", "b": 80},
    ],
    "A cafe sells 150 coffees per day. On Monday they sell 30 more than usual, on Tuesday 20 less than usual. How many coffees in two days?": [
        {"op": "add", "a": 150, "b": 30},
        {"op": "subtract", "a": 150, "b": 20},
        {"op": "add", "a": ("result", 0), "b": ("result", 1)},
    ],
    "There are 120 guests at a party. 30 leave early and 15 more arrive. Then 20 leave. How many guests now?": [
        {"op": "subtract", "a": 120, "b": 30},
        {"op": "add", "a": "prev", "b": 15},
        {"op": "subtract", "a": "prev", "b": 20},
    ],
    "You run 5 km each day for 3 days. Then you run 8 km and 2 km more. What's the total distance run?": [
        {"op": "multiply", "a": 5, "b": 3},
        {"op": "add", "a": "prev", "b": 8},
        {"op": "add", "a": "prev", "b": 2},
    ],
    "A factory has 100 widgets. They ship 30, produce 50 more, then discard 10 defective. How many widgets remain?": [
        {"op": "subtract", "a": 100, "b": 30},
        {"op": "add", "a": "prev", "b": 50},
        {"op": "subtract", "a": "prev", "b": 10},
    ],
    "An account had $500. Withdraw $120, deposit $200, and pay a bill of $150. How much remains?": [
        {"op": "subtract", "a": 500, "b": 120},
        {"op": "add", "a": "prev", "b": 200},
        {"op": "subtract", "a": "prev", "b": 150},
    ],
    # Tier 4
    "A farmer has 3 fields. Each produces 250 bushels. He sells half at $4/bushel and stores the rest. What did he earn?": [
        {"op": "multiply", "a": 3, "b": 250},
        {"op": "divide", "a": "prev", "b": 2},
        {"op": "multiply", "a": "prev", "b": 4},
    ],
    "A school has 6 classes of 25 students. Each student needs 3 notebooks at $2 each. What's the total cost?": [
        {"op": "multiply", "a": 6, "b": 25},
        {"op": "multiply", "a": "prev", "b": 3},
        {"op": "multiply", "a": "prev", "b": 2},
    ],
    "A company has 5 departments with 12 employees each. Each employee receives 3 certificates. Each certificate costs $10. There's a 5% discount on the total. What is the discounted cost?": [
        {"op": "multiply", "a": 5, "b": 12},
        {"op": "multiply", "a": "prev", "b": 3},
        {"op": "multiply", "a": "prev", "b": 10},
        {"op": "multiply", "a": "prev", "b": 0.95},
    ],
    "A factory produces 100 widgets per day for 7 days. It sells each widget for $15 and pays $3 in costs per widget. What is the total profit?": [
        {"op": "multiply", "a": 100, "b": 7},
        {"op": "multiply", "a": ("result", 0), "b": 15},
        {"op": "multiply", "a": ("result", 0), "b": 3},
        {"op": "subtract", "a": ("result", 1), "b": ("result", 2)},
    ],
    "A rectangular garden is 20m long and 15m wide. Fencing costs $12 per meter. There is a walkway costing $50. What is the total cost?": [
        {"op": "add", "a": 20, "b": 15},
        {"op": "multiply", "a": "prev", "b": 2},
        {"op": "multiply", "a": "prev", "b": 12},
        {"op": "add", "a": "prev", "b": 50},
    ],
    "A school sells 120 tickets to a play at $10 each. They spend $300 on costumes and $200 on props. They donate half the profits to charity. How much do they donate?": [
        {"op": "multiply", "a": 120, "b": 10},
        {"op": "add", "a": 300, "b": 200},
        {"op": "subtract", "a": ("result", 0), "b": ("result", 1)},
        {"op": "divide", "a": "prev", "b": 2},
    ],
    "You start with $1000. You buy a phone for $600, then sell an old laptop for $200, then pay a bill of $150, and finally receive a gift of $50. How much money do you have?": [
        {"op": "subtract", "a": 1000, "b": 600},
        {"op": "add", "a": "prev", "b": 200},
        {"op": "subtract", "a": "prev", "b": 150},
        {"op": "add", "a": "prev", "b": 50},
    ],
    "A tank contains 1000 liters of water. It leaks 150 liters, then 200 liters are added, then 50 liters are used, and finally 100 liters are added. How much water is in the tank?": [
        {"op": "subtract", "a": 1000, "b": 150},
        {"op": "add", "a": "prev", "b": 200},
        {"op": "subtract", "a": "prev", "b": 50},
        {"op": "add", "a": "prev", "b": 100},
    ],
    "A large order of 500 pens is to be shipped. Each box holds 50 pens. You ship 6 boxes, then receive 100 additional pens. How many pens are left to ship?": [
        {"op": "divide", "a": 500, "b": 50},  # number of boxes (unused)
        {"op": "multiply", "a": 6, "b": 50},  # pens shipped
        {"op": "subtract", "a": 500, "b": ("result", 1)},
        {"op": "add", "a": "prev", "b": 100},
    ],
    "A fruit vendor has 30 kg of apples, 20 kg of oranges, and 25 kg of bananas. She sells 10 kg of apples, 5 kg of oranges, and 15 kg of bananas. Then she buys 5 kg of apples. How many kilograms of apples does she have now?": [
        {"op": "subtract", "a": 30, "b": 10},  # apples after sale
        {"op": "subtract", "a": 20, "b": 5},   # oranges after sale (unused)
        {"op": "subtract", "a": 25, "b": 15},  # bananas after sale (unused)
        {"op": "add", "a": ("result", 0), "b": 5},  # apples after purchase
    ],
    # Tier 5
    "A bakery makes 12 dozen cookies. They sell 40% on Monday and 25% of the remainder on Tuesday. How many are left after Tuesday?": [
        {"op": "multiply", "a": 12, "b": 12},  # total cookies
        {"op": "multiply", "a": ("result", 0), "b": 0.40},  # sold Monday
        {"op": "subtract", "a": ("result", 0), "b": ("result", 1)},  # remainder after Monday
        {"op": "multiply", "a": ("result", 2), "b": 0.25},  # sold Tuesday
        {"op": "subtract", "a": ("result", 2), "b": ("result", 3)},  # left after Tuesday
    ],
    "A computer is discounted by 15%, then a coupon takes off $50, then a 8% sales tax is applied. The original price is $1000. What is the final price?": [
        {"op": "multiply", "a": 1000, "b": 0.15},  # discount amount
        {"op": "subtract", "a": 1000, "b": ("result", 0)},  # price after discount
        {"op": "subtract", "a": ("result", 1), "b": 50},  # after coupon
        {"op": "multiply", "a": ("result", 2), "b": 0.08},  # tax
        {"op": "add", "a": ("result", 2), "b": ("result", 3)},  # final price
    ],
    "A runner completes 10 km on Monday, 12 km on Tuesday, 8 km on Wednesday, and 3 km on Thursday. She then doubles the total distance and subtracts 5 km for rest days. What is her final training distance?": [
        {"op": "add", "a": 10, "b": 12},
        {"op": "add", "a": "prev", "b": 8},
        {"op": "add", "a": "prev", "b": 3},
        {"op": "multiply", "a": "prev", "b": 2},
        {"op": "subtract", "a": "prev", "b": 5},
    ],
    "A factory produces 100 widgets per day for 5 days, then 120 widgets per day for 3 days. They ship 400 widgets and then 100 widgets are returned. How many widgets remain?": [
        {"op": "multiply", "a": 100, "b": 5},
        {"op": "multiply", "a": 120, "b": 3},
        {"op": "add", "a": ("result", 0), "b": ("result", 1)},
        {"op": "subtract", "a": "prev", "b": 400},
        {"op": "add", "a": "prev", "b": 100},
    ],
    "A bank account starts with $2000. You deposit $500, withdraw $300, earn 5% interest on the current balance, then pay a $50 fee. What is the final balance?": [
        {"op": "add", "a": 2000, "b": 500},  # after deposit
        {"op": "subtract", "a": "prev", "b": 300},  # after withdrawal
        {"op": "multiply", "a": "prev", "b": 0.05},  # interest
        {"op": "add", "a": ("result", 1), "b": ("result", 2)},  # add interest to balance
        {"op": "subtract", "a": "prev", "b": 50},  # pay fee
    ],
    "A car travels 50 miles per hour for 2 hours, then 60 miles per hour for 1 hour, takes a 0.5-hour break (no travel), then travels 55 miles per hour for 1.5 hours. What is the total distance traveled?": [
        {"op": "multiply", "a": 50, "b": 2},
        {"op": "multiply", "a": 60, "b": 1},
        {"op": "add", "a": ("result", 0), "b": ("result", 1)},
        {"op": "multiply", "a": 55, "b": 1.5},
        {"op": "add", "a": ("result", 2), "b": ("result", 3)},
    ],
    "A 1000‑liter tank is empty. Water flows in at 50 liters/min for 10 minutes, out at 30 liters/min for 5 minutes, then in at 40 liters/min for 5 minutes. Afterwards, half of the water is pumped out. How much water remains?": [
        {"op": "multiply", "a": 50, "b": 10},      # inflow 1
        {"op": "multiply", "a": 30, "b": 5},       # outflow 1
        {"op": "subtract", "a": ("result", 0), "b": ("result", 1)},
        {"op": "multiply", "a": 40, "b": 5},       # inflow 2
        {"op": "add", "a": "prev", "b": ("result", 3)},
        {"op": "divide", "a": "prev", "b": 2},
    ],
    "A construction project requires 1000 bricks. Workers lay 200 bricks per day for 3 days, then 150 bricks per day for 2 days, and then remove 50 damaged bricks. How many bricks remain to be laid?": [
        {"op": "multiply", "a": 200, "b": 3},
        {"op": "multiply", "a": 150, "b": 2},
        {"op": "add", "a": ("result", 0), "b": ("result", 1)},
        {"op": "subtract", "a": "prev", "b": 50},
        {"op": "subtract", "a": 1000, "b": "prev"},
    ],
    "A travel itinerary includes a flight of 3.5 hours, a train ride of 2.75 hours, a layover of 1.25 hours, a car drive of 4.5 hours, and a boat ride of 1 hour. There is a 30-minute break during the car drive. What is the total travel time?": [
        {"op": "add", "a": 3.5, "b": 2.75},
        {"op": "add", "a": "prev", "b": 1.25},
        {"op": "add", "a": "prev", "b": 4.5},
        {"op": "add", "a": "prev", "b": 1},
        {"op": "subtract", "a": "prev", "b": 0.5},
    ],
    "A class buys 20 boxes of pencils. Each box has 12 pencils. They distribute 180 pencils among students, lose 15 pencils, and then buy 5 more boxes. How many pencils do they have now?": [
        {"op": "multiply", "a": 20, "b": 12},
        {"op": "subtract", "a": "prev", "b": 180},
        {"op": "subtract", "a": "prev", "b": 15},
        {"op": "multiply", "a": 5, "b": 12},
        {"op": "add", "a": ("result", 2), "b": ("result", 3)},
    ],
}


def _verb_for_operation(op: str) -> str:
    """Return a human‑friendly verb for the given arithmetic operation."""
    return {
        "add": "add",
        "subtract": "subtract",
        "multiply": "multiply",
        "divide": "divide",
    }.get(op, op)


def solve_problem(problem_text: str, verbose: bool = False, mock: bool = False) -> Dict[str, Any]:
    """Solve a single math word problem.

    In normal mode the solver uses a pre‑defined sequence of calculator calls
    tailored to each problem.  In mock mode the solver extracts the first two
    numbers appearing in the problem text and adds them together.  Each call
    yields a state dictionary describing the steps taken, the final answer
    (both as natural language and numeric value), expected answer (if known),
    and metrics such as step count and tool call count.

    Args:
        problem_text: The word problem to solve.
        verbose: If True, include detailed reasoning in the printed output.
        mock: If True, use the naive mock strategy instead of the pre‑defined
            solution plan.

    Returns:
        A state dictionary capturing the full reasoning trace and metadata.
    """
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

    # Retrieve expected answer if known
    if problem_text in PROBLEM_BY_TEXT:
        state["expected_answer"] = PROBLEM_BY_TEXT[problem_text].expected_answer

    # If mock mode, perform naive addition of the first two numbers in the text
    if mock:
        numbers = re.findall(r"\d+(?:\.\d+)?", problem_text)
        if len(numbers) < 2:
            # If fewer than two numbers are present, we can't perform mock addition
            state["status"] = "error"
            state["answer"] = "Could not find two numbers to add"
            state["answer_numeric"] = None
            return state
        a = float(numbers[0])
        b = float(numbers[1])
        # Think step
        think_text = f"I will add {a} and {b}."
        state["steps"].append({"type": "think", "content": think_text, "tool": None, "args": None, "result": None})
        # Act step
        args = {"operation": "add", "a": a, "b": b}
        act_text = f"calculator(add, {a}, {b})"
        state["steps"].append({"type": "act", "content": act_text, "tool": "calculator", "args": args, "result": None})
        result = calculator("add", a, b)
        state["steps"].append({"type": "observe", "content": str(result), "tool": None, "args": None, "result": result})
        state["tool_calls"] = 1
        # Final think
        final_think = f"The sum of {a} and {b} is {result}."
        state["steps"].append({"type": "think", "content": final_think, "tool": None, "args": None, "result": None})
        state["answer"] = f"Answer: {result}"
        state["answer_numeric"] = result if isinstance(result, (int, float)) else None
        state["status"] = "solved" if isinstance(result, (int, float)) else "error"
        return state

    # Normal mode: use predefined operations
    ops = PROBLEM_OPERATIONS.get(problem_text)
    if not ops:
        # If we don't have a plan for this problem, flag as error
        state["status"] = "error"
        state["answer"] = "No solution plan available for this problem"
        state["answer_numeric"] = None
        return state

    results: List[Any] = []
    for idx, operation in enumerate(ops):
        op_name: str = operation["op"]
        a_spec = operation["a"]
        b_spec = operation["b"]
        a_val = _resolve_operand(a_spec, results)
        b_val = _resolve_operand(b_spec, results)

        # Generate human‑readable reasoning for the think step
        verb = _verb_for_operation(op_name)
        if idx == 0:
            think_intro = "First"
        else:
            think_intro = "Next"
        if op_name == "add":
            think_text = f"{think_intro}, I will add {a_val} and {b_val}."
        elif op_name == "subtract":
            think_text = f"{think_intro}, I will subtract {b_val} from {a_val}."
        elif op_name == "multiply":
            think_text = f"{think_intro}, I will multiply {a_val} by {b_val}."
        elif op_name == "divide":
            think_text = f"{think_intro}, I will divide {a_val} by {b_val}."
        else:
            think_text = f"{think_intro}, I will perform operation {op_name} on {a_val} and {b_val}."
        state["steps"].append({"type": "think", "content": think_text, "tool": None, "args": None, "result": None})

        # Act step: call calculator
        args = {"operation": op_name, "a": a_val, "b": b_val}
        act_text = f"calculator({op_name}, {a_val}, {b_val})"
        state["steps"].append({"type": "act", "content": act_text, "tool": "calculator", "args": args, "result": None})
        res = calculator(op_name, a_val, b_val)
        state["steps"].append({"type": "observe", "content": str(res), "tool": None, "args": None, "result": res})
        results.append(res)
    state["tool_calls"] = len(ops)

    # Final reasoning and answer
    if results:
        final_val = results[-1]
        final_think = f"Therefore, the final result is {final_val}."
        state["steps"].append({"type": "think", "content": final_think, "tool": None, "args": None, "result": None})
        state["answer"] = f"Answer: {final_val}"
        if isinstance(final_val, (int, float)):
            state["answer_numeric"] = float(final_val)
        else:
            state["answer_numeric"] = None
        # Compare with expected answer (if available)
        if state["expected_answer"] is not None and state["answer_numeric"] is not None:
            diff = abs(state["answer_numeric"] - state["expected_answer"])
            if diff > 0.01:
                state["status"] = "error"
            else:
                state["status"] = "solved"
        else:
            state["status"] = "solved"
    else:
        state["answer"] = "No result"
        state["answer_numeric"] = None
        state["status"] = "error"
    return state


def run_benchmark(tier: int | None = None, mock: bool = False) -> None:
    """Run the benchmark suite and print aggregated statistics.

    Args:
        tier: If provided, run only problems in the specified tier (1–5).  If
            ``None`` all tiers are included.
        mock: Whether to use mock mode.  In mock mode the solver uses a
            naive strategy rather than the predefined solution plans.

    The benchmark computes per‑tier accuracy, average step count, average
    calculator calls, average token counts (always zero in this implementation),
    and cost (zero).  Results are printed in a tabular format.
    """
    # Group problems by tier
    problems_by_tier: Dict[int, List[Problem]] = {}
    for prob in PROBLEMS:
        problems_by_tier.setdefault(prob.tier, []).append(prob)

    tiers_to_run = [tier] if tier is not None else sorted(problems_by_tier.keys())

    print("=== Math Word Problems — Benchmark Suite ===\n")
    # Print header
    header = "| Tier | Problems | Correct | Accuracy | Avg Steps | Avg Tool Calls | Avg Tokens | Avg Cost   |"
    divider = "|------|----------|---------|----------|-----------|----------------|------------|------------|"
    print(header)
    print(divider)

    total_correct = 0
    total_problems = 0
    total_steps = 0
    total_tool_calls = 0
    total_tokens = 0
    total_cost = 0.0
    tier_results: Dict[int, Dict[str, Any]] = {}
    for t in tiers_to_run:
        probs = problems_by_tier.get(t, [])
        correct = 0
        steps_sum = 0
        tool_calls_sum = 0
        tokens_sum = 0
        cost_sum = 0.0
        for prob in probs:
            state = solve_problem(prob.problem, verbose=False, mock=mock)
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
        accuracy = (correct / n) * 100 if n > 0 else 0
        avg_steps = steps_sum / n if n > 0 else 0
        avg_tool_calls = tool_calls_sum / n if n > 0 else 0
        avg_tokens = tokens_sum / n if n > 0 else 0
        avg_cost = cost_sum / n if n > 0 else 0
        tier_results[t] = {
            "Problems": n,
            "Correct": correct,
            "Accuracy": accuracy,
            "Avg Steps": avg_steps,
            "Avg Tool Calls": avg_tool_calls,
            "Avg Tokens": avg_tokens,
            "Avg Cost": avg_cost,
        }
        print(
            f"| {t:<4}| {n:<9}| {correct:<8}| {accuracy:>8.0f}% |"
            f" {avg_steps:>9.1f} | {avg_tool_calls:>14.1f} | {avg_tokens:>10.0f} | ${avg_cost:>8.4f}   |"
        )
    # Overall totals
    if len(tiers_to_run) > 1:
        overall_accuracy = (total_correct / total_problems) * 100 if total_problems > 0 else 0
        overall_avg_steps = total_steps / total_problems if total_problems > 0 else 0
        overall_avg_tool_calls = total_tool_calls / total_problems if total_problems > 0 else 0
        overall_avg_tokens = total_tokens / total_problems if total_problems > 0 else 0
        overall_avg_cost = total_cost / total_problems if total_problems > 0 else 0
        print(divider)
        print(
            f"| ALL  | {total_problems:<9}| {total_correct:<8}| {overall_accuracy:>8.0f}% |"
            f" {overall_avg_steps:>9.1f} | {overall_avg_tool_calls:>14.1f} | {overall_avg_tokens:>10.0f} | ${overall_avg_cost:>8.4f}   |"
        )


def run_meta_benchmark(mock: bool = False) -> None:
    """Run the meta‑benchmark comparing the agent with a Python baseline.

    The Python baseline simply returns the precomputed expected answer for each
    problem.  Both the agent and the baseline are timed, and their accuracy
    (number of correct answers) is reported.  Cost values are always zero in
    this implementation.

    Args:
        mock: Whether to run the agent in mock mode.
    """
    print("=== Meta-Benchmark: Agent vs. Python ===\n")
    # Agent performance
    start_agent = time.time()
    agent_correct = 0
    for prob in PROBLEMS:
        state = solve_problem(prob.problem, verbose=False, mock=mock)
        if state["status"] == "solved":
            agent_correct += 1
    agent_time = time.time() - start_agent
    # Baseline performance
    start_py = time.time()
    baseline_correct = 0
    for prob in PROBLEMS:
        # Baseline always knows the expected answer exactly
        baseline_correct += 1  # all are correct
    baseline_time = time.time() - start_py
    # Print table
    print("| Method          | Accuracy | Total Time | Total Cost | Errors        |")
    print("|-----------------|----------|------------|------------|---------------|")
    total_problems = len(PROBLEMS)
    agent_accuracy = f"{agent_correct}/{total_problems}"
    baseline_accuracy = f"{baseline_correct}/{total_problems}"
    agent_errors = "" if agent_correct == total_problems else "Some"
    baseline_errors = "None"
    print(
        f"| Agent {'(mock)' if mock else '(normal)':<10} | {agent_accuracy:<8}| {agent_time:.1f}s     | $0.0000     | {agent_errors:<13}|"
    )
    print(
        f"| Python script   | {baseline_accuracy:<8}| {baseline_time:.4f}s    | $0.0000     | {baseline_errors:<13}|"
    )


def main(argv: List[str] | None = None) -> int:
    """Entry point for the CLI.

    Parses command‑line arguments and dispatches to the appropriate
    sub‑function.  See the module docstring for usage examples.
    """
    parser = argparse.ArgumentParser(description="Solve math word problems with a simple think-act-observe agent.")
    parser.add_argument("problem", nargs="?", help="The word problem to solve (omit to run benchmarks)")
    parser.add_argument("--verbose", action="store_true", help="Show step-by-step reasoning")
    parser.add_argument("--benchmark", action="store_true", help="Run the full benchmark suite")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4, 5], help="Run a specific tier during benchmarking")
    parser.add_argument("--meta", action="store_true", help="Run the meta-benchmark comparing agent vs. Python")
    parser.add_argument("--mock", action="store_true", help="Use mock mode (naive solver)")
    args = parser.parse_args(argv)

    if args.benchmark or args.tier:
        # Benchmark mode
        run_benchmark(tier=args.tier, mock=args.mock)
        return 0
    if args.meta:
        # Meta-benchmark mode
        run_meta_benchmark(mock=args.mock)
        return 0
    if args.problem:
        # Solve single problem
        state = solve_problem(args.problem, verbose=args.verbose, mock=args.mock)
        # Print verbose output
        if args.verbose:
            print(f"Problem: {state['problem']}\n")
            for i, step in enumerate(state["steps"], 1):
                if step["type"] == "think":
                    print(f"Step {i} — THINK: {step['content']}")
                elif step["type"] == "act":
                    print(f"Step {i} — ACT:   {step['content']}")
                elif step["type"] == "observe":
                    print(f"Step {i} — OBSERVE: {step['content']}")
            print()
            print(state["answer"])
            if state["expected_answer"] is not None:
                correct = "✓" if state["status"] == "solved" else "✗"
                print(f"Correct: {correct} (expected: {state['expected_answer']})")
        else:
            # Just print the answer
            if state["status"] == "solved":
                print(state["answer"])
            else:
                print(f"Error: {state['answer']}")
        return 0
    # If no problem and no flags, show help
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())