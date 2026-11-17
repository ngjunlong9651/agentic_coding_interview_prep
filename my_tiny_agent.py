"""Practice: write the agent loop yourself.

Run:  .venv/bin/python my_tiny_agent.py
Needs ANTHROPIC_API_KEY. Each run costs about 1 to 3 US cents.

Everything except the heart of the loop is done for you. Compare with `tiny_agent` in notebook section 1.4
only AFTER you have tried it.
"""

import ast
import operator

import anthropic

MODEL = "claude-opus-5-5"

# ---- The environment and tools (done for you) ------------------------------------------------

PRICES = {"pen": 1.50, "notebook": 4.00, "stapler": 7.25}


def get_price(item: str) -> str:
    if item not in PRICES:
        raise ValueError(f"Unknown item '{item}'. Known items: {', '.join(PRICES)}.")
    return f"{PRICES[item]:.2f}"


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def calculate(expression: str) -> str:
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        raise ValueError("Only numbers and + - * / ( ) are allowed.")
    return f"{ev(ast.parse(expression, mode='eval').body):.4f}"


SHOP_TOOLS = [
    {"name": "get_price",
     "description": "Get the unit price in SGD of one item sold in our shop, e.g. 'pen'. Use this for any price question.",
     "input_schema": {"type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]}},
    {"name": "calculate",
     "description": "Evaluate an arithmetic expression with + - * / and parentheses, e.g. '(3*1.5)*1.09'. Use it for all arithmetic.",
     "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}},
]


def run_shop_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    """Runs one tool. Returns (output_text, is_error). Never raises."""
    functions = {"get_price": get_price, "calculate": calculate}
    try:
        return functions[name](**tool_input), False
    except Exception as e:
        return f"Error: {e}", True


def text_of(response) -> str:
    return "".join(block.text for block in response.content if block.type == "text")


# ---- The agent loop ----------------------------------------------------------------------------

def tiny_agent(client, question: str, max_turns: int = 10) -> str:
    messages = [{"role": "user", "content": question}]

    for turn in range(1, max_turns + 1):
        response = client.messages.create(model=MODEL, max_tokens=4000, tools=SHOP_TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": response.content})
        print(f"turn {turn}: stop_reason={response.stop_reason}")

        # TODO(human): decide what happens next, using `response`, `messages` and `run_shop_tool`.
        #   1. If the model did NOT ask for a tool, return its answer (text_of(response)).
        #   2. Otherwise run EVERY tool_use block in response.content, and append the results to
        #      `messages` so the next call can read them. Then the loop goes round again.
        # Each tool_use block has .id, .name and .input. A result looks like:
        #   {"type": "tool_result", "tool_use_id": ..., "content": ..., "is_error": ...}

    return "Stopped: reached max_turns without an answer."


if __name__ == "__main__":
    client = anthropic.Anthropic()
    question = "How much do 3 pens and 2 notebooks from your shop cost in total, including 9% GST?"
    print("\nANSWER:", tiny_agent(client, question))
