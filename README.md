# Math Word Problems — Single-Agent Hello World

## The canonical "hello world" for single-agent LLM systems

An agent that can't do math. A calculator it can call. Word problems it has to solve.

The simplest possible demonstration of the think-act-observe loop: the agent reads a word problem, reasons about which operations to perform, calls a calculator tool, observes the result, and chains operations together until it reaches the answer.

Built with LangGraph and Claude.

---

## Why This Exists

`print("Hello, World!")` teaches you that your compiler works.

Math Word Problems teaches you that your agent works — that it can reason about a task, select a tool, execute an action, observe the result, and decide what to do next. The math is trivial. The pattern is the point.

---

## Quick Start

```bash
git clone https://github.com/violethawk/math-word-problems.git
cd math-word-problems
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here

# Run a single problem
python solver.py "I have 3 baskets with 12 apples each. I eat 7. How many are left?"

# Run the benchmark suite
python solver.py --benchmark

# Run with step-by-step trace visible
python solver.py --verbose "A shirt costs $25. It's 20% off. What's the sale price?"
```

---

## Sample Output

```
Problem: I have 3 baskets with 12 apples each. I eat 7. How many are left?

Step 1 — THINK: I need to find the total number of apples first.
         Multiply 3 baskets × 12 apples.
Step 2 — ACT:   calculator(multiply, 3, 12)
Step 3 — OBSERVE: 36
Step 4 — THINK: Now I subtract the 7 apples that were eaten.
Step 5 — ACT:   calculator(subtract, 36, 7)
Step 6 — OBSERVE: 29

Answer: 29 apples
Steps: 3 think, 2 tool calls
Correct: ✓
```

---

## The Three Phases

### Phase 1 — One Tool, Clear Problems

The agent has one tool: a basic calculator with four operations (add, subtract, multiply, divide). Problems have unambiguous answers and require 1-4 operations chained together.

The agent cannot do math in its head. It must use the calculator for every arithmetic operation — even obvious ones. This constraint is enforced by the system prompt and validated by checking that every numeric result in the answer trace came from a tool call, not from the LLM's own output.

**Key concepts:** Think-act-observe loop, tool calling, chain-of-thought reasoning.

**Benchmark:** 50 word problems across 5 difficulty tiers (1-operation through 5-operation). Measure: accuracy, tool calls per problem, total tokens, cost.

### Phase 2 — Multiple Tools, Ambiguous Problems

Add tools: unit converter (miles↔km, lbs↔kg, °F↔°C), percentage calculator, date/time calculator. Problems don't specify which tool to use — the agent must decide.

"A recipe serves 4 and needs 2.5 cups of flour. I'm cooking for 7 people and I only have a kitchen scale. How many grams of flour do I need?"

The agent has to: scale the recipe (multiply), then convert cups to grams (unit converter). Tool selection, not just tool use.

**Key concepts:** Tool selection, multi-tool chaining, implicit requirements.

**Benchmark:** 30 problems requiring 2-3 different tools. Measure: accuracy, correct tool selection rate, unnecessary tool calls.

### Phase 3 — Incomplete Information and Failure

Problems that can't be solved with the available information. The agent must recognize the gap and say so rather than hallucinate an answer.

"Two trains leave cities 400 miles apart, heading toward each other. Train A goes 60 mph. When do they meet?"

Train B's speed is missing. The correct answer is "I can't solve this — Train B's speed isn't given." The agent should ask for the missing information rather than guess.

Also: problems with trick questions, irrelevant information included as distractors, and problems where the arithmetic is simple but the reasoning about what to compute is hard.

**Key concepts:** Knowing when you can't solve something, asking clarifying questions, filtering irrelevant information, reasoning vs. computing.

**Benchmark:** 20 solvable problems mixed with 10 unsolvable ones. Measure: accuracy on solvable, correct rejection rate on unsolvable, hallucination rate.

---

## Architecture

```
User (word problem)
  → Agent (Claude Haiku — cheap, fast, sufficient for math reasoning)
    → THINK: decompose the problem
    → ACT: call calculator/converter tool
    → OBSERVE: receive result
    → THINK: what's next?
    → [repeat until solved]
  → Answer + full reasoning trace
```

The agent is a single LangGraph node with a tool-calling loop. State tracks the conversation history, tool call results, and the agent's reasoning chain.

```python
{
    "problem": str,                    # the word problem text
    "steps": [
        {
            "type": "think" | "act" | "observe",
            "content": str,            # reasoning text or tool result
            "tool": str | None,        # tool name if type == "act"
            "args": dict | None        # tool arguments if type == "act"
        }
    ],
    "answer": str | None,              # final answer
    "answer_numeric": float | None,    # extracted numeric answer for verification
    "status": "solving" | "solved" | "unsolvable",
    "tool_calls": int,
    "tokens_used": int
}
```

---

## Tools

### calculator

```python
@tool
def calculator(operation: str, a: float, b: float) -> float:
    """
    Performs basic arithmetic.
    
    Args:
        operation: one of "add", "subtract", "multiply", "divide"
        a: first number
        b: second number
    
    Returns:
        Result of the operation.
    """
```

Four operations. That's it. No exponents, no roots, no trig. If the agent needs something fancier, it has to compose from these primitives (e.g., squaring is `multiply(x, x)`). This constraint is deliberate — it forces multi-step reasoning.

### unit_converter (Phase 2)

```python
@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> float:
    """
    Converts between common units.
    
    Supported conversions:
        Distance: miles ↔ km, feet ↔ meters, inches ↔ cm
        Weight: lbs ↔ kg, oz ↔ grams, cups ↔ grams (flour/sugar/water)
        Temperature: fahrenheit ↔ celsius
    """
```

### percentage_calculator (Phase 2)

```python
@tool
def percentage_calculator(operation: str, base: float, percent: float) -> float:
    """
    Percentage operations.
    
    Args:
        operation: "of" (what is X% of Y), "change" (Y changed by X%), 
                   "what_percent" (X is what % of Y)
        base: the base number
        percent: the percentage value
    """
```

### date_calculator (Phase 2)

```python
@tool
def date_calculator(operation: str, date: str, days: int = 0) -> str:
    """
    Date arithmetic.
    
    Args:
        operation: "add_days", "subtract_days", "days_between", "day_of_week"
        date: date string in YYYY-MM-DD format
        days: number of days (for add/subtract operations)
    """
```

---

## Benchmark Suite

### Problem Difficulty Tiers (Phase 1)

**Tier 1 — Single operation (10 problems)**
"What is 47 plus 83?"
"A book costs $15. You buy 3. How much total?"

**Tier 2 — Two operations (10 problems)**
"I have 3 baskets with 12 apples each. I eat 7. How many are left?"
"A pizza has 8 slices. 3 people each eat 2 slices. How many are left?"

**Tier 3 — Three operations (10 problems)**
"A store has 4 shelves with 15 books each. They sell 12 and receive a shipment of 20. How many now?"
"You earn $12/hour for 8 hours, then $18/hour for 3 hours overtime. What's your total pay?"

**Tier 4 — Four operations (10 problems)**
"A farmer has 3 fields. Each produces 250 bushels. He sells half at $4/bushel and stores the rest. What did he earn?"

**Tier 5 — Five operations (10 problems)**
"A road trip is 340 miles. You drive 65mph for 3 hours, stop for gas, then 55mph for the rest. How long is the second leg?"

### Benchmark Output

```
| Tier | Problems | Accuracy | Avg Steps | Avg Tool Calls | Avg Tokens | Avg Cost   |
|------|----------|----------|-----------|----------------|------------|------------|
| 1    | 10       | 10/10    | 3.2       | 1.0            | 180        | $0.0001    |
| 2    | 10       | 10/10    | 5.1       | 2.0            | 310        | $0.0002    |
| 3    | 10       | 9/10     | 7.4       | 3.1            | 480        | $0.0004    |
| 4    | 10       | 8/10     | 9.8       | 4.2            | 620        | $0.0005    |
| 5    | 10       | 7/10     | 12.1      | 5.5            | 810        | $0.0007    |
```

The expected pattern: accuracy drops and cost rises as problem complexity increases. The interesting question is where it breaks — which is the tier where chaining tool calls becomes unreliable.

### The Meta-Benchmark

After the agent benchmark, run the same 50 problems through plain Python arithmetic. Report:

```
| Method          | Accuracy | Time     | Cost    |
|-----------------|----------|----------|---------|
| Agent (Haiku)   | 44/50    | 23.4s    | $0.019  |
| Python script   | 50/50    | 0.001s   | $0.00   |
```

This is the math equivalent of Phase 1 vs. Phase 3 in 52 Card Pickup. The agent is slower, more expensive, and less accurate than a script that just does the math. The point isn't that agents are good at arithmetic. The point is that you now understand how agents work — and you know when not to use one.

---

## Project Structure

```
solver.py              Core agent: LangGraph state machine, tool loop, answer extraction
tools.py               Calculator, unit converter, percentage, date tools
problems.py            Problem sets for all tiers and phases
benchmarks.py          Benchmark runner and reporting
tests/                 Unit tests for tools and agent behavior
prompts/               Tier 2 implementation prompts for each phase
docs/
  tutorial_phase_1.md  Tutorial: one tool, clear problems
  tutorial_phase_2.md  Tutorial: multiple tools, ambiguous problems
  tutorial_phase_3.md  Tutorial: incomplete information and failure
  blog_post.md         "Math Word Problems: The Single-Agent Hello World"
```

---

## What This Project Teaches

1. **The think-act-observe loop** — the fundamental pattern underlying all LLM agents
2. **Tool calling** — how agents interact with external capabilities
3. **Chain-of-thought reasoning** — how agents decompose complex tasks into steps
4. **Tool selection** — how agents decide which tool to use (Phase 2)
5. **Knowing what you don't know** — how agents handle missing information (Phase 3)
6. **When not to use an agent** — the meta-benchmark proves a Python script beats the agent on every dimension for this task

---

## Where This Leads

Math Word Problems is Level 1 of a three-part curriculum for agentic coding:

**Level 1 — Math Word Problems** (this project)
You understand what an agent *is*. One agent, one tool, think-act-observe.

**Level 2 — Maze Solver** *(coming soon)*
You understand what an agent is *good at*. One agent, spatial reasoning, memory, backtracking. Benchmarked against A* and wall-following.

**Level 3 — [52 Card Pickup](https://github.com/violethawk/Fifty-Two-Card-Pickup)**
You understand how agents *work together* — and when they shouldn't. Multiple agents, coordination, conflict resolution, scaling experiments.

Nobody wants to do math homework. Nobody wants to solve a maze. Nobody wants to pick up 52 cards. That's why we build agents.

---

## Dependencies

- langgraph — state machine and tool loop
- anthropic — Claude Haiku for agent reasoning
- pytest — unit tests

Python 3.11+ stdlib (argparse, json, datetime).

---

## License

MIT
