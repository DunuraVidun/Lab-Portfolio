# ============================================================
# LAB 06 - PART B: GIVE YOUR AI AGENT MEMORY
# SQLite long-term memory + Calculator Tool + Memory Search Tool
# ============================================================

import ast
import operator
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_NAME = "llama3.2:3b"

# The database is created beside this Python file.
# It remains available after the program is closed.
DATABASE_PATH = Path(__file__).resolve().with_name("agent_memory.db")

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0
)

print(f"Ollama LLM initialized successfully: {MODEL_NAME}")


# ============================================================
# 2. CREATE SQLITE MEMORY DATABASE
# ============================================================

def initialise_database():
    """Create the SQLite database and memory table if necessary."""

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_key TEXT NOT NULL,
                memory_value TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        connection.commit()


initialise_database()


# ============================================================
# 3. STORE EXPLICIT USER INFORMATION
# ============================================================

def save_memory(memory_key: str, memory_value: str) -> str:
    """Save one user fact in SQLite."""

    memory_key = memory_key.strip().lower()
    memory_value = memory_value.strip()

    if not memory_key or not memory_value:
        return "Memory was not saved because the information was empty."

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            INSERT INTO memories (memory_key, memory_value, created_at)
            VALUES (?, ?, ?)
            """,
            (
                memory_key,
                memory_value,
                datetime.now().astimezone().isoformat(timespec="seconds"),
            ),
        )
        connection.commit()

    return f"Saved memory: {memory_key} = {memory_value}"


def extract_and_save_user_information(user_input: str):
    """
    Save clearly stated personal facts.

    This intentionally uses simple patterns rather than asking the
    language model to invent or infer personal information.
    """

    text = user_input.strip()

    patterns = [
        (r"\bmy name is\s+(.+)", "name"),
        (r"\bi live in\s+(.+)", "location"),
        (r"\bi am from\s+(.+)", "origin"),
        (r"\bi work as\s+(.+)", "occupation"),
        (r"\bi study\s+(.+)", "study"),
        (r"\bi am studying\s+(.+)", "study"),
        (r"\bi prefer\s+(.+)", "preference"),
        (r"\bi like\s+(.+)", "likes"),
        (r"\bi dislike\s+(.+)", "dislikes"),
        (r"\bremember that\s+(.+)", "note"),
    ]

    for pattern, memory_key in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            value = match.group(1).strip().rstrip(".!?")

            if value:
                # Avoid storing very long accidental text.
                if len(value) > 300:
                    return None

                return save_memory(memory_key, value)

    return None


# ============================================================
# 4. SAFE CALCULATOR HELPER
# ============================================================

def calculate_expression(expression: str):
    """Safely evaluate basic arithmetic using Python AST."""

    expression = expression.strip()
    expression = expression.replace("×", "*")
    expression = expression.replace("÷", "/")
    expression = expression.replace("^", "**")

    if len(expression) > 100:
        raise ValueError("Expression is too long.")

    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
    }

    allowed_unary_operators = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def evaluate(node):

        if isinstance(node, ast.Expression):
            return evaluate(node.body)

        if isinstance(node, ast.Constant):
            if type(node.value) in (int, float):
                return node.value
            raise ValueError("Only numeric values are allowed.")

        if isinstance(node, ast.BinOp):
            operator_type = type(node.op)

            if operator_type not in allowed_operators:
                raise ValueError("Unsupported mathematical operator.")

            left = evaluate(node.left)
            right = evaluate(node.right)

            if operator_type == ast.Pow and abs(right) > 100:
                raise ValueError("Exponent is too large.")

            result = allowed_operators[operator_type](left, right)

            if isinstance(result, (int, float)) and abs(result) > 1e100:
                raise ValueError("Result is too large.")

            return result

        if isinstance(node, ast.UnaryOp):
            operator_type = type(node.op)

            if operator_type not in allowed_unary_operators:
                raise ValueError("Unsupported unary operator.")

            return allowed_unary_operators[operator_type](
                evaluate(node.operand)
            )

        raise ValueError("Invalid mathematical expression.")

    syntax_tree = ast.parse(expression, mode="eval")

    return evaluate(syntax_tree)


# ============================================================
# 5. TOOL 1 - CALCULATOR TOOL
# ============================================================

@tool
def calculator_tool(expression: str) -> str:
    """
    Perform mathematical calculations.

    Args:
        expression: Arithmetic expression, for example 25 * 4.
    """

    try:
        result = calculate_expression(expression)
        return f"Calculation successful. Result: {result}"

    except Exception as error:
        return f"Calculator error: {error}"


# ============================================================
# 6. TOOL 2 - MEMORY SEARCH TOOL
# ============================================================

@tool
def memory_search_tool(query: str) -> str:
    """
    Search previously stored user information in the SQLite database.

    Args:
        query: Information to search for, such as name, location,
               occupation, study, preferences or likes.
    """

    try:
        query = query.strip().lower()

        if not query:
            return "Please provide something to search for."

        # Map common natural-language questions to memory categories.
        category_map = {
            "name": ["name"],
            "location": ["location"],
            "live": ["location"],
            "where": ["location", "origin"],
            "origin": ["origin"],
            "from": ["origin"],
            "work": ["occupation"],
            "job": ["occupation"],
            "occupation": ["occupation"],
            "study": ["study"],
            "studying": ["study"],
            "prefer": ["preference"],
            "preference": ["preference"],
            "like": ["likes"],
            "likes": ["likes"],
            "dislike": ["dislikes"],
        }

        search_categories = set()

        for word, categories in category_map.items():
            if re.search(rf"\b{re.escape(word)}\b", query):
                search_categories.update(categories)

        with sqlite3.connect(DATABASE_PATH) as connection:
            connection.row_factory = sqlite3.Row

            if search_categories:
                placeholders = ",".join("?" for _ in search_categories)

                rows = connection.execute(
                    f"""
                    SELECT memory_key, memory_value, created_at
                    FROM memories
                    WHERE memory_key IN ({placeholders})
                    ORDER BY id DESC
                    LIMIT 20
                    """,
                    tuple(search_categories),
                ).fetchall()

            else:
                # Search both keys and values for other queries.
                words = [
                    word for word in re.findall(r"[a-zA-Z0-9]+", query)
                    if len(word) > 2
                ]

                if not words:
                    return "Please provide a more specific memory search."

                conditions = " OR ".join(
                    "(LOWER(memory_key) LIKE ? OR LOWER(memory_value) LIKE ?)"
                    for _ in words
                )

                parameters = []
                for word in words:
                    parameters.extend([f"%{word}%", f"%{word}%"])

                rows = connection.execute(
                    f"""
                    SELECT memory_key, memory_value, created_at
                    FROM memories
                    WHERE {conditions}
                    ORDER BY id DESC
                    LIMIT 20
                    """,
                    parameters,
                ).fetchall()

        if not rows:
            return f"No matching memories found for: {query}"

        results = [
            f"- {row['memory_key']}: {row['memory_value']}"
            for row in rows
        ]

        return (
            f"Found {len(rows)} matching memory record(s):\n"
            + "\n".join(results)
        )

    except Exception as error:
        return f"Memory search error: {error}"


# ============================================================
# 7. REGISTER THE TWO TOOLS
# ============================================================

tools = [
    calculator_tool,
    memory_search_tool,
]

tool_map = {
    registered_tool.name: registered_tool
    for registered_tool in tools
}

llm_with_tools = llm.bind_tools(tools)

print("\nTwo tools registered successfully!")
print("1. calculator_tool")
print("2. memory_search_tool")


# ============================================================
# 8. DETECT MEMORY-RELATED QUESTIONS
# ============================================================

def is_memory_question(user_input: str) -> bool:
    """Identify questions that ask about previously shared information."""

    text = user_input.lower()

    memory_question_patterns = [
        r"\bwhat is my name\b",
        r"\bwhat's my name\b",
        r"\bwhere do i live\b",
        r"\bwhere am i from\b",
        r"\bwhat do i study\b",
        r"\bwhat am i studying\b",
        r"\bwhat is my job\b",
        r"\bwhat do i work as\b",
        r"\bwhat do i prefer\b",
        r"\bwhat do i like\b",
        r"\bwhat did i tell you\b",
        r"\bwhat do you remember about me\b",
        r"\bremember anything about me\b",
    ]

    return any(
        re.search(pattern, text)
        for pattern in memory_question_patterns
    )


def memory_query_from_question(user_input: str) -> str:
    """Convert common questions into searchable memory categories."""

    text = user_input.lower()

    if "name" in text:
        return "name"
    if "live" in text:
        return "location"
    if "from" in text:
        return "origin"
    if "study" in text or "studying" in text:
        return "study"
    if "job" in text or "work" in text:
        return "occupation"
    if "prefer" in text:
        return "preference"
    if "like" in text:
        return "likes"

    return user_input


# ============================================================
# 9. AGENT EXECUTION
# ============================================================

SYSTEM_PROMPT = """
You are an intelligent AI assistant with persistent SQLite memory.

You have exactly two tools:

1. calculator_tool:
   Use this for every mathematical calculation.

2. memory_search_tool:
   Use this whenever the user asks about previously shared personal
   information or something they told you earlier.

IMPORTANT RULES:

- Never invent personal information.
- If memory_search_tool returns no matching memories, clearly say
  that the information is not stored in memory.
- Use the actual memory search result when answering memory questions.
- Use calculator_tool for mathematical calculations.
- For normal conversation, answer naturally.
"""


def run_agent(user_input: str, conversation_history: list):
    """Process one user message and preserve conversation context."""

    print("\n" + "=" * 60)
    print("USER:", user_input)
    print("=" * 60)

    # Save explicitly stated information before answering.
    saved_result = extract_and_save_user_information(user_input)

    if saved_result:
        print("\n[MEMORY SAVED]", saved_result)

    # Keep the current conversation context.
    conversation_history.append(HumanMessage(content=user_input))

    # For common memory questions, explicitly invoke the memory tool.
    # This also makes the lab's name-recall demonstration reliable.
    if is_memory_question(user_input):
        search_result = memory_search_tool.invoke({
            "query": memory_query_from_question(user_input)
        })

        print("\n[TOOL SELECTED] memory_search_tool")
        print("[TOOL ARGUMENTS]", {
            "query": memory_query_from_question(user_input)
        })
        print("[TOOL RESULT]", search_result)

        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            *conversation_history,
            HumanMessage(
                content=(
                    "Use this SQLite memory search result to answer "
                    "the user's question. Do not invent information.\n\n"
                    f"MEMORY SEARCH RESULT:\n{search_result}"
                )
            ),
        ])

        print("\nAGENT RESPONSE:")
        print(response.content)

        conversation_history.append(response)
        return

    # Normal tool-calling loop for calculator and other conversation.
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *conversation_history,
    ]

    for step in range(5):

        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print("\nAGENT RESPONSE:")
            print(response.content)
            conversation_history.append(response)
            return

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            print("\n[TOOL SELECTED]", tool_name)
            print("[TOOL ARGUMENTS]", tool_args)

            if tool_name in tool_map:
                try:
                    tool_result = tool_map[tool_name].invoke(tool_args)
                except Exception as error:
                    tool_result = f"Tool execution failed: {error}"
            else:
                tool_result = f"Unknown tool: {tool_name}"

            print("[TOOL RESULT]", tool_result)

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_id,
                )
            )

    print("\nAgent stopped: maximum execution steps reached.")


# ============================================================
# 10. RUN THE MEMORY-ENABLED AI AGENT
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("          MEMORY-ENABLED AI AGENT")
    print("=" * 60)

    print("\nMemory storage: SQLite")
    print(f"Database location: {DATABASE_PATH}")
    print("\nAvailable tools: 2")
    print("1. Calculator Tool")
    print("2. Memory Search Tool")
    print("\nType 'exit' to stop the agent.")

    conversation_history = []

    while True:

        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("AI Agent terminated.")
                break

            if not user_input:
                continue

            run_agent(user_input, conversation_history)

        except KeyboardInterrupt:
            print("\nAI Agent terminated.")
            break

        except Exception as error:
            print(f"\nAgent error: {error}")
