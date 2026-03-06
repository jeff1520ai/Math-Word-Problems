"""
math_word_problems.operations
------------------------------

Predefined operation plans for Phase 1 problems.  Each entry maps a
problem text string to a list of calculator operations that solve it.

Operands may be numeric literals, ``'prev'`` (most recent result), or
``('result', i)`` (result at index *i*).
"""
from __future__ import annotations

from typing import Any, Dict, List

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
        {"op": "divide", "a": 500, "b": 50},
        {"op": "multiply", "a": 6, "b": 50},
        {"op": "subtract", "a": 500, "b": ("result", 1)},
        {"op": "add", "a": "prev", "b": 100},
    ],
    "A fruit vendor has 30 kg of apples, 20 kg of oranges, and 25 kg of bananas. She sells 10 kg of apples, 5 kg of oranges, and 15 kg of bananas. Then she buys 5 kg of apples. How many kilograms of apples does she have now?": [
        {"op": "subtract", "a": 30, "b": 10},
        {"op": "subtract", "a": 20, "b": 5},
        {"op": "subtract", "a": 25, "b": 15},
        {"op": "add", "a": ("result", 0), "b": 5},
    ],
    # Tier 5
    "A bakery makes 12 dozen cookies. They sell 40% on Monday and 25% of the remainder on Tuesday. How many are left after Tuesday?": [
        {"op": "multiply", "a": 12, "b": 12},
        {"op": "multiply", "a": ("result", 0), "b": 0.40},
        {"op": "subtract", "a": ("result", 0), "b": ("result", 1)},
        {"op": "multiply", "a": ("result", 2), "b": 0.25},
        {"op": "subtract", "a": ("result", 2), "b": ("result", 3)},
    ],
    "A computer is discounted by 15%, then a coupon takes off $50, then a 8% sales tax is applied. The original price is $1000. What is the final price?": [
        {"op": "multiply", "a": 1000, "b": 0.15},
        {"op": "subtract", "a": 1000, "b": ("result", 0)},
        {"op": "subtract", "a": ("result", 1), "b": 50},
        {"op": "multiply", "a": ("result", 2), "b": 0.08},
        {"op": "add", "a": ("result", 2), "b": ("result", 3)},
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
        {"op": "add", "a": 2000, "b": 500},
        {"op": "subtract", "a": "prev", "b": 300},
        {"op": "multiply", "a": "prev", "b": 0.05},
        {"op": "add", "a": ("result", 1), "b": ("result", 2)},
        {"op": "subtract", "a": "prev", "b": 50},
    ],
    "A car travels 50 miles per hour for 2 hours, then 60 miles per hour for 1 hour, takes a 0.5\u2011hour break (no travel), then travels 55 miles per hour for 1.5 hours. What is the total distance traveled?": [
        {"op": "multiply", "a": 50, "b": 2},
        {"op": "multiply", "a": 60, "b": 1},
        {"op": "add", "a": ("result", 0), "b": ("result", 1)},
        {"op": "multiply", "a": 55, "b": 1.5},
        {"op": "add", "a": ("result", 2), "b": ("result", 3)},
    ],
    "A 1000\u2011liter tank is empty. Water flows in at 50 liters/min for 10 minutes, out at 30 liters/min for 5 minutes, then in at 40 liters/min for 5 minutes. Afterwards, half of the water is pumped out. How much water remains?": [
        {"op": "multiply", "a": 50, "b": 10},
        {"op": "multiply", "a": 30, "b": 5},
        {"op": "subtract", "a": ("result", 0), "b": ("result", 1)},
        {"op": "multiply", "a": 40, "b": 5},
        {"op": "add", "a": ("result", 2), "b": ("result", 3)},
        {"op": "divide", "a": "prev", "b": 2},
    ],
    "A construction project requires 1000 bricks. Workers lay 200 bricks per day for 3 days, then 150 bricks per day for 2 days, and then remove 50 damaged bricks. How many bricks remain to be laid?": [
        {"op": "multiply", "a": 200, "b": 3},
        {"op": "multiply", "a": 150, "b": 2},
        {"op": "add", "a": ("result", 0), "b": ("result", 1)},
        {"op": "subtract", "a": "prev", "b": 50},
        {"op": "subtract", "a": 1000, "b": "prev"},
    ],
    "A travel itinerary includes a flight of 3.5 hours, a train ride of 2.75 hours, a layover of 1.25 hours, a car drive of 4.5 hours, and a boat ride of 1 hour. There is a 30\u2011minute break during the car drive. What is the total travel time?": [
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
