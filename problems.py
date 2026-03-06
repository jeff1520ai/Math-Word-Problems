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


# Phase 1 problems (all tiers)
PHASE1_PROBLEMS: List[Problem] = _TIER1 + _TIER2 + _TIER3 + _TIER4 + _TIER5


# ---------------------------------------------------------------------------
# Phase 2 — Multiple Tools, Ambiguous Problems (30 problems)
# These problems require 2-3 different tools and the agent must decide which
# tool to use.  Tools: calculator, unit_converter, percentage_calculator,
# date_calculator.
# ---------------------------------------------------------------------------

PHASE2_PROBLEMS: List[Problem] = [
    # calculator + percentage_calculator
    Problem(
        problem="A shirt costs $80 and is 25% off. What is the sale price?",
        expected_answer=80 * (1 - 0.25),
        tier=6,
    ),
    Problem(
        problem="A laptop costs $1200. There's a 10% discount, then you pay 8% sales tax on the discounted price. What do you pay?",
        expected_answer=1200 * 0.90 * 1.08,
        tier=6,
    ),
    Problem(
        problem="You earn $3000 per month. You save 15% of your salary. After 4 months, how much have you saved?",
        expected_answer=3000 * 0.15 * 4,
        tier=6,
    ),
    Problem(
        problem="A restaurant bill is $85. You want to leave a 20% tip. What is the total amount you pay?",
        expected_answer=85 * 1.20,
        tier=6,
    ),
    Problem(
        problem="A store marks up products by 40%. If a product costs $50 to make, what is the selling price?",
        expected_answer=50 * 1.40,
        tier=6,
    ),
    # calculator + unit_converter
    Problem(
        problem="A recipe calls for 2.5 cups of flour for 4 servings. You're making 8 servings. How many grams of flour do you need?",
        expected_answer=(2.5 * (8 / 4)) * 120.0,
        tier=6,
    ),
    Problem(
        problem="You drive 120 miles. Your friend drives 150 km. Who drove farther, and by how many miles?",
        expected_answer=120 - 150 / 1.60934,
        tier=6,
    ),
    Problem(
        problem="A package weighs 5 lbs. You need to ship 3 of them. What is the total weight in kilograms?",
        expected_answer=5 * 3 * 0.453592,
        tier=6,
    ),
    Problem(
        problem="A room is 15 feet long and 12 feet wide. What is the area in square feet?",
        expected_answer=15 * 12,
        tier=6,
    ),
    Problem(
        problem="You run a 10 km race in 50 minutes. What was your pace in minutes per mile?",
        expected_answer=50 / (10 / 1.60934),
        tier=6,
    ),
    # percentage_calculator + calculator
    Problem(
        problem="A town has 25000 people. The population grew by 12% last year. Then 500 people moved away. What is the current population?",
        expected_answer=25000 * 1.12 - 500,
        tier=6,
    ),
    Problem(
        problem="You invest $5000 and earn 7% return. You then withdraw $200. How much remains?",
        expected_answer=5000 * 1.07 - 200,
        tier=6,
    ),
    Problem(
        problem="A class of 40 students took a test. 85% passed. Of those who passed, 6 got an A. How many who passed did not get an A?",
        expected_answer=40 * 0.85 - 6,
        tier=6,
    ),
    Problem(
        problem="A factory produces 2000 units. 3% are defective. Each defective unit costs $25 to fix. What is the total repair cost?",
        expected_answer=2000 * 0.03 * 25,
        tier=6,
    ),
    Problem(
        problem="A house was bought for $250000 and appreciated by 8%. What is it worth now in thousands of dollars?",
        expected_answer=250000 * 1.08 / 1000,
        tier=6,
    ),
    # unit_converter + percentage_calculator
    Problem(
        problem="A European car gets 15 km per liter. Fuel costs $1.20 per liter. You drive 100 miles. How much do you spend on fuel?",
        expected_answer=(100 * 1.60934 / 15) * 1.20,
        tier=6,
    ),
    Problem(
        problem="You weigh 180 lbs and want to lose 10% of your body weight. How many kilograms do you need to lose?",
        expected_answer=180 * 0.10 * 0.453592,
        tier=6,
    ),
    Problem(
        problem="A recipe for 6 people uses 500 grams of sugar. You're cooking for 9 people. How many cups of sugar do you need?",
        expected_answer=(500 * (9 / 6)) / 200.0,
        tier=6,
    ),
    # calculator + date_calculator
    Problem(
        problem="You start a project on 2025-03-01 and it takes 45 days. Then you have a 10-day review. How many total days from start to review end?",
        expected_answer=55,
        tier=6,
    ),
    Problem(
        problem="An event is 120 days away. You need 90 days to prepare. How many days do you have before you must start preparing?",
        expected_answer=120 - 90,
        tier=6,
    ),
    # Three-tool combinations
    Problem(
        problem="You buy 3 items at $40 each with a 15% discount. Convert the total to euros at 0.92 euros per dollar. How many euros do you pay?",
        expected_answer=3 * 40 * 0.85 * 0.92,
        tier=6,
    ),
    Problem(
        problem="A 50 kg bag of rice costs $80. You need 110 lbs of rice. How much will it cost?",
        expected_answer=(110 * 0.453592 / 50) * 80,
        tier=6,
    ),
    Problem(
        problem="A car travels 300 miles on 10 gallons of gas. Gas costs $3.50 per gallon. How many km can you drive per dollar spent?",
        expected_answer=(300 * 1.60934) / (10 * 3.50),
        tier=6,
    ),
    Problem(
        problem="Your salary is $4500. You pay 22% in taxes and then convert the remainder to British pounds at 0.79 per dollar. How many pounds do you take home?",
        expected_answer=4500 * (1 - 0.22) * 0.79,
        tier=6,
    ),
    Problem(
        problem="A pool holds 10000 gallons. You fill it at 5 gallons per minute. After 80% is full, you switch to 3 gallons per minute. How many total minutes to fill it?",
        expected_answer=(10000 * 0.80 / 5) + (10000 * 0.20 / 3),
        tier=6,
    ),
    Problem(
        problem="A 5 km race charges $30 entry. If 200 runners sign up and 15% drop out, how much revenue is collected from those who actually run?",
        expected_answer=200 * (1 - 0.15) * 30,
        tier=6,
    ),
    Problem(
        problem="A farmer has 200 acres. He plants 60% with corn and 25% with wheat. How many acres are unplanted?",
        expected_answer=200 * (1 - 0.60 - 0.25),
        tier=6,
    ),
    Problem(
        problem="You mix 2 lbs of coffee at $12/lb with 3 lbs at $8/lb. What is the average price per kilogram of the blend?",
        expected_answer=(2 * 12 + 3 * 8) / ((2 + 3) * 0.453592),
        tier=6,
    ),
    Problem(
        problem="A marathon is 26.2 miles. You ran 30 km. What percentage of a marathon have you completed?",
        expected_answer=(30 / 1.60934) / 26.2 * 100,
        tier=6,
    ),
    Problem(
        problem="A tank holds 500 liters. It's 40% full. You add 50 gallons of water (1 gallon = 3.785 liters). How many liters are in the tank now?",
        expected_answer=500 * 0.40 + 50 * 3.785,
        tier=6,
    ),
]


# ---------------------------------------------------------------------------
# Phase 3 — Incomplete Information and Failure (30 problems)
# 20 solvable problems mixed with 10 unsolvable ones.  The agent must
# recognise unsolvable problems and say so rather than hallucinate.
# ---------------------------------------------------------------------------

PHASE3_SOLVABLE: List[Problem] = [
    Problem(
        problem="A train travels at 80 mph for 3 hours. How far does it go?",
        expected_answer=80 * 3,
        tier=7,
    ),
    Problem(
        problem="You buy 4 notebooks at $3 each and 2 pens at $1.50 each. The store gives you a $2 discount. How much do you pay?",
        expected_answer=4 * 3 + 2 * 1.50 - 2,
        tier=7,
    ),
    Problem(
        problem="A garden is 10m by 8m. You plant flowers in a 3m by 3m section. How much area is left unplanted?",
        expected_answer=10 * 8 - 3 * 3,
        tier=7,
    ),
    Problem(
        problem="There are 5 red balls, 3 blue balls, and 7 green balls in a bag. You remove all the blue balls. How many balls remain?",
        expected_answer=5 + 7,
        tier=7,
    ),
    Problem(
        problem="A movie is 2 hours and 15 minutes long. You've watched 45 minutes. How many minutes are left?",
        expected_answer=2 * 60 + 15 - 45,
        tier=7,
    ),
    Problem(
        problem="A baker makes 5 dozen muffins and sells them for $2 each. She gives 10 away for free. How much revenue does she earn?",
        expected_answer=(5 * 12 - 10) * 2,
        tier=7,
    ),
    Problem(
        problem="You have $500. You buy a jacket for $120 and shoes for $80. Your friend owes you $50 and pays you back. How much do you have?",
        expected_answer=500 - 120 - 80 + 50,
        tier=7,
    ),
    Problem(
        problem="A parking lot has 200 spaces. 60% are occupied in the morning. By noon, 30 more cars arrive. How many spaces are still empty?",
        expected_answer=200 - (200 * 0.60 + 30),
        tier=7,
    ),
    Problem(
        problem="A pool is 50 meters long. A swimmer does 12 laps (one lap = one length). How many meters did she swim?",
        expected_answer=50 * 12,
        tier=7,
    ),
    Problem(
        problem="A teacher has 120 worksheets. She gives 4 worksheets to each of her 28 students. How many worksheets are left over?",
        expected_answer=120 - 4 * 28,
        tier=7,
    ),
    # Problems with irrelevant/distractor information
    Problem(
        problem="John is 35 years old and weighs 180 lbs. He buys 6 apples at $0.75 each. How much does he spend?",
        expected_answer=6 * 0.75,
        tier=7,
    ),
    Problem(
        problem="A blue car and a red car are in a parking lot. The blue car has 4 doors. The red car has 2 doors. There are also 3 motorcycles. How many doors are there on the cars?",
        expected_answer=4 + 2,
        tier=7,
    ),
    Problem(
        problem="Sarah has 3 cats, 2 dogs, and a goldfish. Each cat eats 200g of food per day and each dog eats 400g. How much food do the cats and dogs eat per day in grams?",
        expected_answer=3 * 200 + 2 * 400,
        tier=7,
    ),
    Problem(
        problem="A rectangular room is 15 feet long, 12 feet wide, and 9 feet tall. Paint costs $25 per gallon and covers 350 square feet. You only need to paint the floor. How many square feet is the floor?",
        expected_answer=15 * 12,
        tier=7,
    ),
    Problem(
        problem="There are 52 cards in a deck. You remove the 4 aces and 4 kings. The remaining cards weigh 1.5 grams each. How many cards remain?",
        expected_answer=52 - 4 - 4,
        tier=7,
    ),
    # Trick questions that are solvable
    Problem(
        problem="If you have 3 quarters, 2 dimes, and 5 nickels, how much money do you have in cents?",
        expected_answer=3 * 25 + 2 * 10 + 5 * 5,
        tier=7,
    ),
    Problem(
        problem="A snail climbs 3 feet during the day but slips back 1 foot at night. After 5 full day-night cycles, how many feet has it gained?",
        expected_answer=(3 - 1) * 5,
        tier=7,
    ),
    Problem(
        problem="You have two ropes. One is 8 meters long and the other is 5 meters long. You cut 2 meters off the longer rope and discard it, then tie the two ropes together. What is the total length?",
        expected_answer=(8 - 2) + 5,
        tier=7,
    ),
    Problem(
        problem="A clock strikes every hour. How many times does it strike between 1:00 PM and 5:00 PM inclusive?",
        expected_answer=1 + 2 + 3 + 4 + 5,
        tier=7,
    ),
    Problem(
        problem="Three friends split a $90 dinner bill equally, then each leaves a $5 tip. How much does each person pay in total?",
        expected_answer=90 / 3 + 5,
        tier=7,
    ),
]

PHASE3_UNSOLVABLE: List[Problem] = [
    Problem(
        problem="Two trains leave cities 400 miles apart, heading toward each other. Train A goes 60 mph. When do they meet?",
        expected_answer=float("nan"),  # unsolvable — Train B's speed is missing
        tier=8,
    ),
    Problem(
        problem="A store sells apples and oranges. Apples cost $2 each. You buy some fruit and pay $14. How many apples did you buy?",
        expected_answer=float("nan"),  # unsolvable — could be a mix of apples and oranges
        tier=8,
    ),
    Problem(
        problem="A rectangle has a perimeter of 30 meters. What is its area?",
        expected_answer=float("nan"),  # unsolvable — need both length and width
        tier=8,
    ),
    Problem(
        problem="A car drives from City A to City B. The trip takes 3 hours. How fast was the car going?",
        expected_answer=float("nan"),  # unsolvable — distance is missing
        tier=8,
    ),
    Problem(
        problem="You mix red and blue paint to make purple. You used 3 cups of red paint. How much purple paint do you have?",
        expected_answer=float("nan"),  # unsolvable — amount of blue paint is missing
        tier=8,
    ),
    Problem(
        problem="A factory produces widgets. Each widget sells for $15. What is the factory's monthly profit?",
        expected_answer=float("nan"),  # unsolvable — production count and costs are missing
        tier=8,
    ),
    Problem(
        problem="Three siblings share some candy. The oldest gets twice as many as the youngest. How many pieces does each child get?",
        expected_answer=float("nan"),  # unsolvable — total candy and middle child's share unknown
        tier=8,
    ),
    Problem(
        problem="A tank is filling with water at some rate. After 2 hours it's half full. When will it be completely full?",
        expected_answer=float("nan"),  # unsolvable — can't assume constant rate without knowing it
        tier=8,
    ),
    Problem(
        problem="A plane flies east at 500 mph. Another plane flies west. When will they be 2000 miles apart?",
        expected_answer=float("nan"),  # unsolvable — second plane's speed is missing
        tier=8,
    ),
    Problem(
        problem="You earn a certain salary. After a 10% raise, you can afford a new car. How much does the car cost?",
        expected_answer=float("nan"),  # unsolvable — salary is unknown
        tier=8,
    ),
]

PHASE3_PROBLEMS: List[Problem] = PHASE3_SOLVABLE + PHASE3_UNSOLVABLE


# Combine all problems into a single list
PROBLEMS: List[Problem] = PHASE1_PROBLEMS + PHASE2_PROBLEMS + PHASE3_PROBLEMS

# Build a lookup dictionary keyed by the problem text for convenience.  Note
# that the key comparison is case-sensitive; ensure that the exact problem
# string is used when performing lookups.
PROBLEM_BY_TEXT: Dict[str, Problem] = {p.problem: p for p in PROBLEMS}
