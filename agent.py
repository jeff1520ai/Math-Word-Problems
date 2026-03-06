"""
agent.py
--------

LLM-powered math word problem solver using LangGraph and Claude.

This module implements the actual agent that reasons about word problems,
selects and calls tools, and chains operations together in a
think-act-observe loop.  It replaces the pre-defined operation plans in
solver.py with genuine LLM reasoning.

The agent supports three phases:
  Phase 1 — single tool (calculator), clear problems
  Phase 2 — multiple tools, agent decides which to use
  Phase 3 — incomplete information, agent must recognise unsolvable problems
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Literal, Sequence, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph

from tools import (
    calculator,
    date_calculator,
    percentage_calculator,
    unit_converter,
)

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

PHASE1_TOOLS = [calculator]
PHASE2_TOOLS = [calculator, unit_converter, percentage_calculator, date_calculator]
PHASE3_TOOLS = PHASE2_TOOLS  # same tools, different system prompt

# Build a lookup by function name
_TOOL_FNS = {
    "calculator": calculator,
    "unit_converter": unit_converter,
    "percentage_calculator": percentage_calculator,
    "date_calculator": date_calculator,
}


# ---------------------------------------------------------------------------
# LangChain tool schemas (for binding to the model)
# ---------------------------------------------------------------------------

def _lc_tool_schemas(tools: list) -> list:
    """Build LangChain-compatible tool schemas from our tool functions."""
    schemas = []
    for fn in tools:
        if fn is calculator:
            schemas.append({
                "name": "calculator",
                "description": (
                    "Performs basic arithmetic. You MUST use this for ALL arithmetic — "
                    "never compute numbers in your head."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide"],
                            "description": "The arithmetic operation to perform.",
                        },
                        "a": {"type": "number", "description": "First operand."},
                        "b": {"type": "number", "description": "Second operand."},
                    },
                    "required": ["operation", "a", "b"],
                },
            })
        elif fn is unit_converter:
            schemas.append({
                "name": "unit_converter",
                "description": (
                    "Converts between common units. "
                    "Distance: miles<->km, feet<->meters, inches<->cm. "
                    "Weight: lbs<->kg, oz<->grams, cups_flour/cups_sugar/cups_water<->grams. "
                    "Temperature: fahrenheit<->celsius."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number", "description": "The value to convert."},
                        "from_unit": {"type": "string", "description": "Source unit."},
                        "to_unit": {"type": "string", "description": "Target unit."},
                    },
                    "required": ["value", "from_unit", "to_unit"],
                },
            })
        elif fn is percentage_calculator:
            schemas.append({
                "name": "percentage_calculator",
                "description": (
                    "Percentage operations. "
                    "'of': what is X% of Y. "
                    "'change': Y changed by X%. "
                    "'what_percent': X is what % of Y."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["of", "change", "what_percent"],
                            "description": "The percentage operation.",
                        },
                        "base": {"type": "number", "description": "The base number."},
                        "percent": {"type": "number", "description": "The percentage value."},
                    },
                    "required": ["operation", "base", "percent"],
                },
            })
        elif fn is date_calculator:
            schemas.append({
                "name": "date_calculator",
                "description": (
                    "Date arithmetic. Operations: add_days, subtract_days, "
                    "days_between, day_of_week. Dates in YYYY-MM-DD format."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add_days", "subtract_days", "days_between", "day_of_week"],
                            "description": "The date operation.",
                        },
                        "date": {"type": "string", "description": "Date in YYYY-MM-DD format."},
                        "days": {"type": "integer", "description": "Number of days (for add/subtract).", "default": 0},
                        "date2": {"type": "string", "description": "Second date (for days_between).", "default": ""},
                    },
                    "required": ["operation", "date"],
                },
            })
    return schemas


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

PHASE1_SYSTEM = """\
You are a math word problem solver. You solve problems step by step using the calculator tool.

CRITICAL RULES:
1. You MUST use the calculator tool for EVERY arithmetic operation. Never compute numbers in your head.
2. Think step by step: identify what operations are needed, then execute them one at a time.
3. After getting all results, state the final answer clearly as: "FINAL ANSWER: <number>"
4. Keep your reasoning concise.
"""

PHASE2_SYSTEM = """\
You are a math word problem solver with access to multiple tools:
- calculator: basic arithmetic (add, subtract, multiply, divide)
- unit_converter: convert between units (distance, weight, temperature)
- percentage_calculator: percentage operations (of, change, what_percent)
- date_calculator: date arithmetic (add_days, subtract_days, days_between, day_of_week)

CRITICAL RULES:
1. You MUST use tools for ALL computations. Never compute numbers in your head.
2. Choose the right tool for each step. You may need multiple tools for one problem.
3. Think step by step, then execute tool calls.
4. State the final answer clearly as: "FINAL ANSWER: <number>" (or a date/string if appropriate).
"""

PHASE3_SYSTEM = """\
You are a math word problem solver with access to multiple tools:
- calculator: basic arithmetic (add, subtract, multiply, divide)
- unit_converter: convert between units (distance, weight, temperature)
- percentage_calculator: percentage operations (of, change, what_percent)
- date_calculator: date arithmetic (add_days, subtract_days, days_between, day_of_week)

CRITICAL RULES:
1. You MUST use tools for ALL computations. Never compute numbers in your head.
2. Before solving, carefully check if the problem provides enough information.
3. If information is MISSING and the problem CANNOT be solved, respond with:
   "UNSOLVABLE: <explanation of what information is missing>"
4. Do NOT guess or assume missing values. Do NOT hallucinate an answer.
5. If the problem IS solvable, solve it step by step and state: "FINAL ANSWER: <number>"
6. Ignore irrelevant/distractor information that isn't needed for the solution.
"""


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: List[BaseMessage]
    steps: List[Dict[str, Any]]
    tool_calls_count: int
    tokens_in: int
    tokens_out: int
    done: bool


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

_MAX_ITERATIONS = 15


def _make_agent_node(model: ChatAnthropic, tool_schemas: list):
    """Create the agent node that calls the LLM."""

    def agent_node(state: AgentState) -> dict:
        messages = state["messages"]
        response = model.invoke(messages)

        # Track token usage from response metadata
        usage = getattr(response, "usage_metadata", None) or {}
        tokens_in = usage.get("input_tokens", 0)
        tokens_out = usage.get("output_tokens", 0)

        new_steps = list(state.get("steps", []))

        # Record think step from the text content
        if response.content:
            text_parts = []
            if isinstance(response.content, str):
                text_parts.append(response.content)
            elif isinstance(response.content, list):
                for block in response.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block["text"])
                    elif isinstance(block, str):
                        text_parts.append(block)
            if text_parts:
                think_text = " ".join(text_parts).strip()
                if think_text:
                    new_steps.append({
                        "type": "think",
                        "content": think_text,
                        "tool": None,
                        "args": None,
                        "result": None,
                    })

        # Record act steps from tool calls
        tool_calls = getattr(response, "tool_calls", []) or []
        for tc in tool_calls:
            new_steps.append({
                "type": "act",
                "content": f"{tc['name']}({tc['args']})",
                "tool": tc["name"],
                "args": tc["args"],
                "result": None,
            })

        new_messages = list(messages) + [response]

        return {
            "messages": new_messages,
            "steps": new_steps,
            "tokens_in": state.get("tokens_in", 0) + tokens_in,
            "tokens_out": state.get("tokens_out", 0) + tokens_out,
            "tool_calls_count": state.get("tool_calls_count", 0),
            "done": state.get("done", False),
        }

    return agent_node


def tool_node(state: AgentState) -> dict:
    """Execute tool calls from the last AI message."""
    messages = state["messages"]
    last_msg = messages[-1]
    tool_calls = getattr(last_msg, "tool_calls", []) or []

    new_messages = list(messages)
    new_steps = list(state.get("steps", []))
    tc_count = state.get("tool_calls_count", 0)

    for tc in tool_calls:
        fn_name = tc["name"]
        fn_args = tc["args"]
        fn = _TOOL_FNS.get(fn_name)
        if fn is None:
            result = f"Error: Unknown tool '{fn_name}'"
        else:
            try:
                result = fn(**fn_args)
            except Exception as e:
                result = f"Error: {e}"

        new_steps.append({
            "type": "observe",
            "content": str(result),
            "tool": fn_name,
            "args": fn_args,
            "result": result,
        })
        new_messages.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"])
        )
        tc_count += 1

    return {
        "messages": new_messages,
        "steps": new_steps,
        "tool_calls_count": tc_count,
        "tokens_in": state.get("tokens_in", 0),
        "tokens_out": state.get("tokens_out", 0),
        "done": state.get("done", False),
    }


def _should_continue(state: AgentState) -> Literal["tool_node", "end"]:
    """Decide whether to call tools or finish."""
    messages = state["messages"]
    last_msg = messages[-1]
    tool_calls = getattr(last_msg, "tool_calls", []) or []

    if tool_calls and state.get("tool_calls_count", 0) < _MAX_ITERATIONS:
        return "tool_node"
    return "end"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(phase: int = 1, model_name: str = "claude-haiku-4-5-20251001"):
    """Build and compile the LangGraph agent.

    Args:
        phase: Which phase (1, 2, or 3) determines available tools and
            system prompt.
        model_name: The Claude model to use.

    Returns:
        A compiled LangGraph that can be invoked with an AgentState.
    """
    if phase == 1:
        tools = PHASE1_TOOLS
        system_prompt = PHASE1_SYSTEM
    elif phase == 2:
        tools = PHASE2_TOOLS
        system_prompt = PHASE2_SYSTEM
    else:
        tools = PHASE3_TOOLS
        system_prompt = PHASE3_SYSTEM

    tool_schemas = _lc_tool_schemas(tools)

    model = ChatAnthropic(
        model=model_name,
        max_tokens=1024,
        temperature=0,
    )
    # Bind tools to the model
    model = model.bind_tools(tool_schemas)

    agent_node = _make_agent_node(model, tool_schemas)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tool_node", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _should_continue, {
        "tool_node": "tool_node",
        "end": END,
    })
    graph.add_edge("tool_node", "agent")

    return graph.compile()


# ---------------------------------------------------------------------------
# Public solve function
# ---------------------------------------------------------------------------

def _extract_final_answer(steps: List[Dict[str, Any]]) -> tuple[float | None, str | None, str]:
    """Extract the final answer from the agent's reasoning steps.

    Returns:
        (numeric_answer, text_answer, status)
    """
    # Look through think steps for FINAL ANSWER or UNSOLVABLE
    for step in reversed(steps):
        if step["type"] != "think":
            continue
        content = step["content"]
        if "UNSOLVABLE:" in content:
            reason = content.split("UNSOLVABLE:", 1)[1].strip()
            return None, reason, "unsolvable"
        if "FINAL ANSWER:" in content:
            answer_str = content.split("FINAL ANSWER:", 1)[1].strip()
            import re
            cleaned = answer_str.replace("*", "").replace(",", "")
            # Strip unicode fraction characters that can corrupt number
            # parsing (e.g. "2⅔" would match as just "2").
            cleaned = re.sub(r"[\u2150-\u215E\u00BC-\u00BE]", " ", cleaned)
            # Take the first line / sentence only, but keep parenthetical
            # alternatives like "(or 2.67)".
            first_line = cleaned.split("\n")[0]
            for sep in ["(Note", "(note", "Here's"]:
                first_line = first_line.split(sep)[0]
            all_matches = re.findall(r"-?\d+\.\d+|-?\d+", first_line)
            if all_matches:
                # If there's a decimal match anywhere on the line, prefer
                # it over a bare integer (handles "2⅔ (or 2.67)" cases).
                decimals = [m for m in all_matches if "." in m]
                val = float(decimals[0] if decimals else all_matches[0])
                return val, answer_str, "solved"
            # Fallback: search the full cleaned text
            all_matches = re.findall(r"-?\d+\.\d+|-?\d+", cleaned)
            if all_matches:
                val = float(all_matches[0])
                return val, answer_str, "solved"
            # Also check for date answers (YYYY-MM-DD)
            date_match = re.search(r"\d{4}-\d{2}-\d{2}", answer_str)
            if date_match:
                return None, date_match.group(), "solved"
            return None, answer_str, "solved"
    return None, None, "error"


def solve_with_agent(
    problem_text: str,
    phase: int = 1,
    model_name: str = "claude-haiku-4-5-20251001",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Solve a math word problem using the LLM agent.

    Args:
        problem_text: The word problem to solve.
        phase: Which phase (1, 2, or 3).
        model_name: The Claude model to use.
        verbose: If True, print step-by-step reasoning.

    Returns:
        A state dictionary with the full reasoning trace.
    """
    from problems import PROBLEM_BY_TEXT

    graph = build_graph(phase=phase, model_name=model_name)

    if phase == 1:
        system_prompt = PHASE1_SYSTEM
    elif phase == 2:
        system_prompt = PHASE2_SYSTEM
    else:
        system_prompt = PHASE3_SYSTEM

    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=problem_text),
        ],
        "steps": [],
        "tool_calls_count": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "done": False,
    }

    final_state = graph.invoke(initial_state)

    steps = final_state.get("steps", [])
    numeric_answer, text_answer, status = _extract_final_answer(steps)

    # Look up expected answer
    expected = None
    if problem_text in PROBLEM_BY_TEXT:
        expected = PROBLEM_BY_TEXT[problem_text].expected_answer

    # Validate against expected answer
    if status == "solved" and expected is not None and numeric_answer is not None:
        if not math.isnan(expected):
            if abs(numeric_answer - expected) > max(0.01, abs(expected) * 0.01):
                status = "wrong"
    elif status == "unsolvable" and expected is not None:
        if not math.isnan(expected):
            status = "wrong"  # incorrectly said unsolvable
    elif status == "solved" and expected is not None and math.isnan(expected):
        status = "wrong"  # should have been unsolvable

    # Check correct rejection of unsolvable
    if status == "unsolvable" and expected is not None and math.isnan(expected):
        status = "correctly_rejected"

    result = {
        "problem": problem_text,
        "steps": steps,
        "answer": text_answer or f"Answer: {numeric_answer}",
        "answer_numeric": numeric_answer,
        "expected_answer": expected,
        "status": status,
        "tool_calls": final_state.get("tool_calls_count", 0),
        "tokens_in": final_state.get("tokens_in", 0),
        "tokens_out": final_state.get("tokens_out", 0),
        "cost": 0.0,  # could be computed from token counts
    }

    if verbose:
        print(f"Problem: {problem_text}\n")
        for i, step in enumerate(steps, 1):
            if step["type"] == "think":
                print(f"Step {i} -- THINK: {step['content']}")
            elif step["type"] == "act":
                print(f"Step {i} -- ACT:   {step['content']}")
            elif step["type"] == "observe":
                print(f"Step {i} -- OBSERVE: {step['content']}")
        print()
        print(result["answer"])
        if expected is not None:
            mark = "+" if status in ("solved", "correctly_rejected") else "X"
            print(f"Correct: {mark} (expected: {expected})")

    return result
