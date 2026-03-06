"""
problems.py
-------------

This module defines the word problem set for Phase 1 of the Math Word Problems
project.  Fifty problems are organised across five difficulty tiers.  Each
problem is represented as a dictionary with the following keys:

``problem`` (str):
    The text of the word problem presented to the agent.

``expected_answer`` (float):
    The numeric answer expected for the problem.  Answers are provided
    as floating‑point values and may include fractional components.

``tier`` (int):
    The difficulty tier (1–5) indicating the number of arithmetic
    operations required to solve the problem.

Problems in the higher tiers often involve multiple chained operations,
such as multiplication followed by addition or division.  All arithmetic
has been thoroughly verified; if you modify or extend the problem set,
please double‑check the calculations to avoid false failures during
benchmarking.

The ``PROBLEMS`` list preserves the order of the problems.  A lookup
dictionary ``PROBLEM_BY_TEXT`` provides constant‑time access by problem
string.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Problem:
    """Simple data container for a math word problem."""

    problem: str
    expected_answer: float
    tier: int


# Tier 1 — Single operation problems (10 problems)
_TIER1: List[Problem] = [
    Problem(
        problem="What is 47 plus 83?",
        expected_answer=47 + 83,
        tier=1,
    ),
    Problem(
        problem="A book costs $15. You buy 3. How much total?",
        expected_answer=15 * 3,
        tier=1,
    ),
    Problem(
        problem="You have 100 stickers and give away 37. How many left?",
        expected_answer=100 - 37,
        tier=1,
    ),
    Problem(
        problem=(
            "A pizza is cut into 8 slices. 3 people split it equally. "
            "How many slices does each person get?"
        ),
        expected_answer=8 / 3,
        tier=1,
    ),
    Problem(
        problem="I have 20 marbles and my friend gives me 15 more. How many marbles do I have?",
        expected_answer=20 + 15,
        tier=1,
    ),
    Problem(
        problem="A car travels 60 miles in one hour. How far does it travel in 4 hours?",
        expected_answer=60 * 4,
        tier=1,
    ),
    Problem(
        problem="You buy 5 bags of candy with 7 candies each. How many candies do you have?",
        expected_answer=5 * 7,
        tier=1,
    ),
    Problem(
        problem="There are 45 students and 3 classes. If the students are split equally, how many students per class?",
        expected_answer=45 / 3,
        tier=1,
    ),
    Problem(
        problem="A farmer has 24 eggs and sells 10. How many eggs remain?",
        expected_answer=24 - 10,
        tier=1,
    ),
    Problem(
        problem="Each pack has 6 batteries. How many batteries are in 5 packs?",
        expected_answer=6 * 5,
        tier=1,
    ),
]


# Tier 2 — Two operation problems (10 problems)
_TIER2: List[Problem] = [
    Problem(
        problem="I have 3 baskets with 12 apples each. I eat 7. How many are left?",
        expected_answer=3 * 12 - 7,
        tier=2,
    ),
    Problem(
        problem="A pizza has 8 slices. 3 people each eat 2 slices. How many are left?",
        expected_answer=8 - (3 * 2),
        tier=2,
    ),
    Problem(
        problem="You earn $15/hour and work 8 hours. You spend $40 on groceries. How much do you have?",
        expected_answer=15 * 8 - 40,
        tier=2,
    ),
    Problem(
        problem="A class has 28 students. 4 are absent. The rest split into groups of 6. How many groups?",
        expected_answer=(28 - 4) / 6,
        tier=2,
    ),
    Problem(
        problem=(
            "A container holds 5 liters. You have 3 containers and pour out 2 liters. "
            "How much liquid is left?"
        ),
        expected_answer=5 * 3 - 2,
        tier=2,
    ),
    Problem(
        problem="You read 30 pages of a book each day for 5 days. Then you read 20 pages. How many pages have you read?",
        expected_answer=30 * 5 + 20,
        tier=2,
    ),
    Problem(
        problem="There are 10 packs of pencils with 12 pencils each. You give away 15 pencils. How many pencils remain?",
        expected_answer=10 * 12 - 15,
        tier=2,
    ),
    Problem(
        problem="A farmer has 7 cows and buys 5 more. Each cow produces 8 liters of milk. How much milk do the cows produce?",
        expected_answer=(7 + 5) * 8,
        tier=2,
    ),
    Problem(
        problem="You save $50 each week for 6 weeks. Then you spend $120. How much do you have left?",
        expected_answer=50 * 6 - 120,
        tier=2,
    ),
    Problem(
        problem="A theatre has 200 seats. In a show, 75 tickets are sold online and 50 at the door. How many seats are empty?",
        expected_answer=200 - (75 + 50),
        tier=2,
    ),
]


# Tier 3 — Three operation problems (10 problems)
_TIER3: List[Problem] = [
    Problem(
        problem="A store has 4 shelves with 15 books each. They sell 12 and receive a shipment of 20. How many now?",
        expected_answer=4 * 15 - 12 + 20,
        tier=3,
    ),
    Problem(
        problem="You earn $12/hour for 8 hours, then $18/hour for 3 hours overtime. What's your total pay?",
        expected_answer=12 * 8 + 18 * 3,
        tier=3,
    ),
    Problem(
        problem="A farmer plants 5 rows of 10 trees each. 8 trees die. He plants 12 more. How many trees?",
        expected_answer=5 * 10 - 8 + 12,
        tier=3,
    ),
    Problem(
        problem="A factory produces 20 gadgets per hour for 5 hours and then 15 gadgets per hour for 4 hours. How many gadgets produced?",
        expected_answer=20 * 5 + 15 * 4,
        tier=3,
    ),
    Problem(
        problem="You have $200. You buy a jacket for $60, shoes for $45, and then earn $80 mowing lawns. How much money do you have now?",
        expected_answer=200 - 60 - 45 + 80,
        tier=3,
    ),
    Problem(
        problem="A cafe sells 150 coffees per day. On Monday they sell 30 more than usual, on Tuesday 20 less than usual. How many coffees in two days?",
        expected_answer=(150 + 30) + (150 - 20),
        tier=3,
    ),
    Problem(
        problem="There are 120 guests at a party. 30 leave early and 15 more arrive. Then 20 leave. How many guests now?",
        expected_answer=120 - 30 + 15 - 20,
        tier=3,
    ),
    Problem(
        problem="You run 5 km each day for 3 days. Then you run 8 km and 2 km more. What's the total distance run?",
        expected_answer=5 * 3 + 8 + 2,
        tier=3,
    ),
    Problem(
        problem="A factory has 100 widgets. They ship 30, produce 50 more, then discard 10 defective. How many widgets remain?",
        expected_answer=100 - 30 + 50 - 10,
        tier=3,
    ),
    Problem(
        problem="An account had $500. Withdraw $120, deposit $200, and pay a bill of $150. How much remains?",
        expected_answer=500 - 120 + 200 - 150,
        tier=3,
    ),
]


# Tier 4 — Four operation problems (10 problems)
_TIER4: List[Problem] = [
    Problem(
        problem="A farmer has 3 fields. Each produces 250 bushels. He sells half at $4/bushel and stores the rest. What did he earn?",
        expected_answer=(3 * 250) / 2 * 4,
        tier=4,
    ),
    Problem(
        problem=(
            "A school has 6 classes of 25 students. Each student needs 3 notebooks at $2 each. "
            "What's the total cost?"
        ),
        expected_answer=6 * 25 * 3 * 2,
        tier=4,
    ),
    Problem(
        problem=(
            "A company has 5 departments with 12 employees each. Each employee receives 3 certificates. "
            "Each certificate costs $10. There's a 5% discount on the total. What is the discounted cost?"
        ),
        expected_answer=5 * 12 * 3 * 10 * 0.95,
        tier=4,
    ),
    Problem(
        problem=(
            "A factory produces 100 widgets per day for 7 days. It sells each widget for $15 and pays $3 in costs per widget. "
            "What is the total profit?"
        ),
        expected_answer=(100 * 7) * 15 - (100 * 7) * 3,
        tier=4,
    ),
    Problem(
        problem=(
            "A rectangular garden is 20m long and 15m wide. Fencing costs $12 per meter. "
            "There is a walkway costing $50. What is the total cost?"
        ),
        expected_answer=(20 + 15) * 2 * 12 + 50,
        tier=4,
    ),
    Problem(
        problem=(
            "A school sells 120 tickets to a play at $10 each. They spend $300 on costumes and $200 on props. "
            "They donate half the profits to charity. How much do they donate?"
        ),
        expected_answer=((120 * 10) - (300 + 200)) / 2,
        tier=4,
    ),
    Problem(
        problem=(
            "You start with $1000. You buy a phone for $600, then sell an old laptop for $200, "
            "then pay a bill of $150, and finally receive a gift of $50. How much money do you have?"
        ),
        expected_answer=1000 - 600 + 200 - 150 + 50,
        tier=4,
    ),
    Problem(
        problem=(
            "A tank contains 1000 liters of water. It leaks 150 liters, then 200 liters are added, "
            "then 50 liters are used, and finally 100 liters are added. How much water is in the tank?"
        ),
        expected_answer=1000 - 150 + 200 - 50 + 100,
        tier=4,
    ),
    Problem(
        problem=(
            "A large order of 500 pens is to be shipped. Each box holds 50 pens. You ship 6 boxes, "
            "then receive 100 additional pens. How many pens are left to ship?"
        ),
        expected_answer=500 - (6 * 50) + 100,
        tier=4,
    ),
    Problem(
        problem=(
            "A fruit vendor has 30 kg of apples, 20 kg of oranges, and 25 kg of bananas. "
            "She sells 10 kg of apples, 5 kg of oranges, and 15 kg of bananas. Then she buys 5 kg of apples. "
            "How many kilograms of apples does she have now?"
        ),
        expected_answer=30 - 10 + 5,
        tier=4,
    ),
]


# Tier 5 — Five operation problems (10 problems)
_TIER5: List[Problem] = [
    Problem(
        problem=(
            "A bakery makes 12 dozen cookies. They sell 40% on Monday and 25% of the remainder on Tuesday. "
            "How many are left after Tuesday?"
        ),
        expected_answer=(12 * 12) * (1 - 0.40) * (1 - 0.25),
        tier=5,
    ),
    Problem(
        problem=(
            "A computer is discounted by 15%, then a coupon takes off $50, then a 8% sales tax is applied. "
            "The original price is $1000. What is the final price?"
        ),
        expected_answer=((1000 - (1000 * 0.15) - 50) * 1.08),
        tier=5,
    ),
    Problem(
        problem=(
            "A runner completes 10 km on Monday, 12 km on Tuesday, 8 km on Wednesday, and 3 km on Thursday. "
            "She then doubles the total distance and subtracts 5 km for rest days. What is her final training distance?"
        ),
        expected_answer=((10 + 12 + 8 + 3) * 2) - 5,
        tier=5,
    ),
    Problem(
        problem=(
            "A factory produces 100 widgets per day for 5 days, then 120 widgets per day for 3 days. "
            "They ship 400 widgets and then 100 widgets are returned. How many widgets remain?"
        ),
        expected_answer=((100 * 5 + 120 * 3) - 400 + 100),
        tier=5,
    ),
    Problem(
        problem=(
            "A bank account starts with $2000. You deposit $500, withdraw $300, earn 5% interest on the current balance, "
            "then pay a $50 fee. What is the final balance?"
        ),
        expected_answer=(((2000 + 500) - 300) * 1.05) - 50,
        tier=5,
    ),
    Problem(
        problem=(
            "A car travels 50 miles per hour for 2 hours, then 60 miles per hour for 1 hour, "
            "takes a 0.5‑hour break (no travel), then travels 55 miles per hour for 1.5 hours. "
            "What is the total distance traveled?"
        ),
        expected_answer=(50 * 2) + (60 * 1) + (55 * 1.5),
        tier=5,
    ),
    Problem(
        problem=(
            "A 1000‑liter tank is empty. Water flows in at 50 liters/min for 10 minutes, out at 30 liters/min for 5 minutes, "
            "then in at 40 liters/min for 5 minutes. Afterwards, half of the water is pumped out. How much water remains?"
        ),
        expected_answer=((50 * 10 - 30 * 5 + 40 * 5) / 2),
        tier=5,
    ),
    Problem(
        problem=(
            "A construction project requires 1000 bricks. Workers lay 200 bricks per day for 3 days, then 150 bricks per day for 2 days, "
            "and then remove 50 damaged bricks. How many bricks remain to be laid?"
        ),
        expected_answer=1000 - (((200 * 3) + (150 * 2)) - 50),
        tier=5,
    ),
    Problem(
        problem=(
            "A travel itinerary includes a flight of 3.5 hours, a train ride of 2.75 hours, a layover of 1.25 hours, "
            "a car drive of 4.5 hours, and a boat ride of 1 hour. There is a 30‑minute break during the car drive. "
            "What is the total travel time?"
        ),
        expected_answer=(3.5 + 2.75 + 1.25 + 4.5 + 1) - 0.5,
        tier=5,
    ),
    Problem(
        problem=(
            "A class buys 20 boxes of pencils. Each box has 12 pencils. They distribute 180 pencils among students, lose 15 pencils, "
            "and then buy 5 more boxes. How many pencils do they have now?"
        ),
        expected_answer=((20 * 12) - 180 - 15 + (5 * 12)),
        tier=5,
    ),
]


# Combine all problems into a single list
PROBLEMS: List[Problem] = _TIER1 + _TIER2 + _TIER3 + _TIER4 + _TIER5

# Build a lookup dictionary keyed by the problem text for convenience.  Note
# that the key comparison is case‑sensitive; ensure that the exact problem
# string is used when performing lookups.
PROBLEM_BY_TEXT: Dict[str, Problem] = {p.problem: p for p in PROBLEMS}
