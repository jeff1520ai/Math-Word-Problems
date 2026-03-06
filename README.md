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

# Solve a problem with the LLM agent
python -m math_word_problems --llm "I have 3 baskets with 12 apples each. I eat 7. How many are left?"

# Solve with step-by-step trace
python -m math_word_problems --llm --verbose "I have 3 baskets with 12 apples each. I eat 7. How many are left?"

# Use Phase 2 tools (unit converter, percentage calculator, date calculator)
python -m math_word_problems --llm --phase 2 "A recipe calls for 2.5 cups of flour for 4 servings. How many grams for 8 servings?"

# Use Phase 3 (detects unsolvable problems)
python -m math_word_problems --llm --phase 3 "A car drives from A to B in 3 hours. How fast was it going?"

# Run the predefined solver (no API key needed)
python -m math_word_problems --benchmark

# Run LLM benchmarks
python -m math_word_problems --llm --benchmark --phase 1
python -m math_word_problems --llm --benchmark --phase 2
python -m math_word_problems --llm --benchmark --phase 3
```

---

## Sample Output

```
Problem: I have 3 baskets with 12 apples each. I eat 7. How many are left?

Step 1 -- THINK: I need to find the total number of apples first. Multiply 3 baskets by 12 apples.
Step 2 -- ACT:   calculator({'operation': 'multiply', 'a': 3, 'b': 12})
Step 3 -- OBSERVE: 36.0
Step 4 -- THINK: Now I subtract the 7 apples that were eaten.
Step 5 -- ACT:   calculator({'operation': 'subtract', 'a': 36, 'b': 7})
Step 6 -- OBSERVE: 29.0
Step 7 -- THINK: FINAL ANSWER: 29

Answer: 29
Correct: + (expected: 29.0)
```

---

## The Three Phases

### Phase 1 — One Tool, Clear Problems

The agent has one tool: a basic calculator with four operations (add, subtract, multiply, divide). Problems have unambiguous answers and require 1-5 operations chained together.

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

The agent is a LangGraph state machine with two nodes — an `agent` node (calls Claude) and a `tool_node` (executes tool calls) — connected by a conditional edge that loops until the model stops requesting tools. State tracks the conversation history, tool call results, and the agent's reasoning chain.

```python
{
    "problem": str,                    # the word problem text
    "steps": [
        {
            "type": "think" | "act" | "observe",
            "content": str,            # reasoning text or tool result
            "tool": str | None,        # tool name if type == "act"
            "args": dict | None,       # tool arguments if type == "act"
            "result": Any | None       # numeric result if type == "observe"
        }
    ],
    "answer": str | None,              # final answer
    "answer_numeric": float | None,    # extracted numeric answer for verification
    "status": "solving" | "solved" | "unsolvable",
    "tool_calls": int,
    "tokens_in": int,
    "tokens_out": int
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

### Benchmark Results (Claude Haiku 4.5)

#### Phase 1 — Calculator Only

```
| Tier | Problems | Correct | Accuracy | Avg Tool Calls | Avg Tokens In | Avg Tokens Out |
|------|----------|---------|----------|----------------|---------------|----------------|
| 1    | 10       | 10      |     100% |            1.0 |          1654 |            134 |
| 2    | 10       | 10      |     100% |            2.0 |          2382 |            264 |
| 3    | 10       | 10      |     100% |            3.0 |          2741 |            371 |
| 4    | 10       | 10      |     100% |            3.7 |          4544 |            470 |
| 5    | 10       | 10      |     100% |            5.1 |          6370 |            634 |
| ALL  | 50       | 50      |     100% |                |               |                |
```

Perfect accuracy across all tiers. Token usage scales linearly with problem complexity.

#### Phase 2 — Multiple Tools

```
| Tier | Problems | Correct | Accuracy | Avg Tool Calls | Avg Tokens In | Avg Tokens Out |
|------|----------|---------|----------|----------------|---------------|----------------|
| 6    | 30       | 30      |     100% |            2.7 |          5402 |            381 |
```

The agent correctly selects between calculator, unit converter, percentage calculator, and date calculator across all 30 problems.

#### Phase 3 — Incomplete Information

```
Solvable problems:     20/20 (100%)
Unsolvable problems:   10/10 correctly rejected (100%)
Hallucination rate:    0/10 (0%)
Overall:               30/30 (100%)
```

The agent correctly identified all 10 unsolvable problems with zero hallucinations, and solved all 20 solvable problems including those with distractor information.

#### The Meta-Benchmark

The predefined solver runs all 50 Phase 1 problems through scripted operation plans — no LLM needed:

```
| Method          | Accuracy | Time      | Cost    |
|-----------------|----------|-----------|---------|
| Agent (Haiku)   | 50/50    | 168.2s    | ~$0.02  |
| Python script   | 50/50    | 0.001s    | $0.00   |
```

The agent gets the same answers but is 168,000x slower and costs money. The point isn't that agents are good at arithmetic. The point is that you now understand how agents work — and you know when not to use one.

---

## Project Structure

```
math_word_problems/        Python package
  __init__.py              Package exports
  __main__.py              CLI entry point (python -m math_word_problems)
  agent.py                 LLM-powered solver: LangGraph state machine, Claude tool loop
  benchmarks.py            Benchmark runners for predefined and LLM modes
  operations.py            Predefined operation plans for Phase 1 problems
  problems.py              110 problems across 3 phases and 8 tiers
  solver.py                Predefined solver and mock mode
  tools.py                 Calculator, unit converter, percentage, date tools
tests/
  test_tools.py            Unit tests for all four tools
  test_solver.py           Unit tests for solver and problem set structure
pyproject.toml             Python packaging and pytest config
requirements.txt           Python dependencies
```

### CLI Reference

```
python -m math_word_problems [problem]
  --llm                   Use the LLM-powered agent (requires ANTHROPIC_API_KEY)
  --phase {1,2,3}         Tool set and system prompt to use (default: 1)
  --model MODEL           Claude model name (default: claude-haiku-4-5-20251001)
  --verbose               Show step-by-step reasoning trace
  --mock                  Use naive mock solver (adds first two numbers)
  --benchmark             Run predefined Phase 1 benchmark
  --tier {1,2,3,4,5}      Run a specific tier only
  --meta                  Compare agent vs. Python baseline
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

40 tests covering all four tools, the predefined solver, mock mode, and problem set structure. No API key required.

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

**Level 2 — Maze Solver**
You understand what an agent is *good at*. One agent, spatial reasoning, memory, backtracking. Benchmarked against A* and wall-following.

**Level 3 — 52 Card Pickup**
You understand how agents *work together* — and when they shouldn't. Multiple agents, coordination, conflict resolution, scaling experiments.

Nobody wants to do math homework. Nobody wants to solve a maze. Nobody wants to pick up 52 cards. That's why we build agents.

---

## Dependencies

- **langgraph** — state machine and tool-calling loop
- **langchain-anthropic** — Claude model integration for LangChain
- **langchain-core** — base message types and tool abstractions
- **anthropic** — Anthropic API client
- **pytest** — unit tests

Python 3.11+ required. All other imports are stdlib (`argparse`, `math`, `re`, `datetime`).

---

## License

MIT
