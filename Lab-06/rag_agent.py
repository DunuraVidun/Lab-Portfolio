# ============================================================
# LAB 06 - PART C: RAG-POWERED AI AGENT (OPTIMIZED)
#
# Integrates:
# 1. Calculator Tool       - From Part B
# 2. Memory Search Tool    - From Part B (SQLite)
# 3. RAG Search Tool       - From Lab 05
#
# Design:
#   - A fast, deterministic ROUTER picks the tool for clear requests
#     (no unreliable 3B-model tool guessing, no extra LLM call).
#   - Tool results are formatted in plain Python, so answers are
#     exact and RAG sources are never lost.
#   - Anything the router does not recognise goes to the LLM, which
#     can still call tools through a proper tool-calling loop.
# ============================================================

import re
import sqlite3
import sys
from pathlib import Path

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool


# ============================================================
# 1. IMPORT EXISTING PART B TOOLS
# ============================================================

from memory_agent import (
    DATABASE_PATH,
    MODEL_NAME,
    calculator_tool,
    extract_and_save_user_information,
    is_memory_question,
    llm,                      # reuse the same Ollama client
    memory_query_from_question,
    memory_search_tool,
    save_memory,  
)


# ============================================================
# 2. CONNECT TO EXISTING LAB 05 RAG PIPELINE
# ============================================================
# Expected structure:
#   Lab-Portfolio/
#       Lab-05/  rag_pipeline.py, ingest.py, chroma_db/
#       Lab-06/  memory_agent.py, rag_agent.py

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAB05_PATH = PROJECT_ROOT / "Lab-05"

if not LAB05_PATH.exists():
    raise FileNotFoundError(f"Lab-05 folder not found: {LAB05_PATH}")

if str(LAB05_PATH) not in sys.path:
    sys.path.insert(0, str(LAB05_PATH))

try:
    from rag_pipeline import ask_university
except ImportError as error:
    raise ImportError(
        f"Could not import ask_university from {LAB05_PATH / 'rag_pipeline.py'}: "
        f"{error}"
    ) from error

print(f"Ollama initialized: {MODEL_NAME}")
print(f"Lab 05 RAG pipeline connected: {LAB05_PATH}")


# ============================================================
# 3. CONFIGURATION
# ============================================================

MAX_HISTORY_MESSAGES = 12     # keeps the prompt small and fast
MAX_TOOL_STEPS = 3            # limit for the LLM tool-calling loop
ALL_MEMORIES = "__all__"      # internal token: "list everything you know"


# ============================================================
# 4. RAG SEARCH TOOL
# ============================================================

@tool
def rag_search_tool(question: str) -> str:
    """
    Search university documents using the existing Lab 05 RAG pipeline.

    Use this tool for questions about:
    - University information and academic policies
    - Assessments and assignments
    - Modules and courses
    - Lab sheets and lecture materials
    - Examinations and academic documents

    Args:
        question: The question to search the university documents for.
    """

    try:
        result = ask_university(question)
        answer = str(result.get("answer", "No answer was returned.")).strip()
        sources = result.get("sources", [])

        # Remove duplicate (document, page) pairs while keeping order.
        unique_sources = list(dict.fromkeys(
            (
                source.get("document", "Unknown document"),
                source.get("page", "Unknown"),
            )
            for source in sources
        ))

        if not unique_sources:
            return (
                f"RAG Search Result:\n{answer}\n\n"
                "Sources: No source documents returned."
            )

        source_lines = "\n".join(
            f"- {document} (Page {page})"
            for document, page in unique_sources
        )

        return f"RAG Search Result:\n{answer}\n\nSources:\n{source_lines}"

    except Exception as error:
        return f"RAG search error: {error}"


# ============================================================
# 5. REGISTER ALL THREE TOOLS
# ============================================================

tools = [calculator_tool, memory_search_tool, rag_search_tool]
tool_map = {registered_tool.name: registered_tool for registered_tool in tools}
llm_with_tools = llm.bind_tools(tools)

print("\nThree tools registered successfully:")
for registered_tool in tools:
    print(f"- {registered_tool.name}")


# ============================================================
# 6. INPUT CLEANING
# ============================================================

def normalise_input(text: str) -> str:
    """
    Clean the raw input.
    Fixes terminal double-paste such as 'Calculate 2*3Calculate 2*3'.
    """

    text = text.strip()
    half = len(text) // 2

    if len(text) % 2 == 0 and half > 3 and text[:half] == text[half:]:
        text = text[:half].strip()

    return text


# ============================================================
# 7. CALCULATOR ROUTING
# ============================================================

# Natural-language operators -> symbols (order matters).
WORD_OPERATORS = [
    (r"\bto the power of\b|\braised to\b|\bpower of\b", "**"),
    (r"\bmultiplied by\b|\bmultiply by\b|\btimes\b", "*"),
    (r"\bdivided by\b|\bdivide by\b|\bover\b", "/"),
    (r"\bplus\b|\badded to\b", "+"),
    (r"\bminus\b|\bsubtract(?:ed)?\b", "-"),
    (r"\bmodulo\b|\bmodulus\b|\bmod\b", "%"),
]

NUMBER = r"-?\d+(?:\.\d+)?"

# number (operator number)+  e.g. 250*4, (2+3)*4, 2**8
MATH_PATTERN = re.compile(
    rf"\(*{NUMBER}\)*(?:\s*(?:\*\*|[-+*/%^])\s*\(*{NUMBER}\)*)+"
)

DATE_PATTERN = re.compile(
    r"\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b"      # 2026-11-14
    r"|\b(?:19|20)\d{2}\s*[-/]\s*(?:19|20)?\d{2}\b"  # 2024-2025, 2024/25
)


def extract_math_expression(text: str):
    """Return a clean arithmetic expression, or None if there is none."""

    lowered = text.lower()

    if DATE_PATTERN.search(lowered):
        return None

    for pattern, symbol in WORD_OPERATORS:
        lowered = re.sub(pattern, f" {symbol} ", lowered)

    lowered = lowered.replace("×", "*").replace("÷", "/")

    match = MATH_PATTERN.search(lowered)
    if not match:
        return None

    # The message should be mostly maths, not a long sentence that
    # happens to contain numbers.
    leftover = lowered[:match.start()] + " " + lowered[match.end():]
    if len(re.findall(r"[a-z]{2,}", leftover)) > 6:
        return None

    return re.sub(r"\s+", "", match.group(0))


# ============================================================
# 8. MEMORY ROUTING
# ============================================================

GENERAL_MEMORY_PATTERNS = [
    r"\bremember about me\b",
    r"\bknow about me\b",
    r"\btell you about me\b",
    r"\bwhat do you know about me\b",
    r"\bwhat information do you (?:have|remember|know)\b",
    r"\bwho am i\b",
]


def is_general_memory_question(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in GENERAL_MEMORY_PATTERNS)


# ============================================================
# 9. RAG ROUTING
# ============================================================

RAG_KEYWORDS = [
    "university", "module", "course", "assessment", "assignment",
    "lecture", "lab sheet", "lab", "policy", "document", "pdf",
    "academic", "semester", "exam", "examination", "credit",
    "degree", "programme", "program", "handbook", "admission",
    "attendance", "award", "curriculum", "syllabus", "prerequisite",
]

RAG_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(word) for word in RAG_KEYWORDS) + r")(?:s|es)?\b"
)


# ============================================================
# 10. ROUTER (SINGLE ENTRY POINT)
# ============================================================

CAPABILITY_PATTERN = re.compile(
    r"\bwhat can you do\b|\bhow can you help\b|\bwhat are your (?:tools|capabilities)\b"
    r"|\bwhat tools\b|^help\b"
)


def route(user_input: str):
    """
    Decide which tool (if any) should handle the request.

    Returns:
        ("calculator_tool", {...}) | ("memory_search_tool", {...}) |
        ("rag_search_tool", {...}) | ("capabilities", {}) | None
    """

    text = user_input.lower().strip()

    # 1. Capability / help question (static answer, no LLM needed)
    if CAPABILITY_PATTERN.search(text):
        return "capabilities", {}

    # 2. Calculator
    expression = extract_math_expression(user_input)
    if expression:
        return "calculator_tool", {"expression": expression}

    # 3. Memory
    if is_general_memory_question(text):
        return "memory_search_tool", {"query": ALL_MEMORIES}

    if is_memory_question(text):
        return "memory_search_tool", {
            "query": memory_query_from_question(user_input)
        }

    # 4. University documents (RAG)
    if RAG_PATTERN.search(text):
        return "rag_search_tool", {"question": user_input}

    # 5. Everything else -> LLM decides
    return None


# ============================================================
# 11. MEMORY HELPERS & FORMATTING
# ============================================================

SINGLE_VALUE_KEYS = {"name", "location", "origin", "occupation", "study", "preference"}

MEMORY_SENTENCES = {
    "name": "Your name is {v}.",
    "location": "You live in {v}.",
    "origin": "You are from {v}.",
    "occupation": "You work as {v}.",
    "study": "You study {v}.",
    "preference": "You prefer {v}.",
    "likes": "You like {v}.",
    "dislikes": "You dislike {v}.",
}

MEMORY_LABELS = {
    "name": "Name",
    "location": "Lives in",
    "origin": "From",
    "occupation": "Occupation",
    "study": "Studies",
    "preference": "Prefers",
    "likes": "Likes",
    "dislikes": "Dislikes",
    "note": "Notes",
}

MEMORY_HINTS = {
    "name": "My name is ...",
    "location": "I live in ...",
    "origin": "I am from ...",
    "occupation": "I work as ...",
    "study": "I study ...",
    "preference": "I prefer ...",
    "likes": "I like ...",
    "dislikes": "I dislike ...",
}

def fetch_all_memories() -> str:
    """Return every stored memory (newest first) in the tool-result format."""

    with sqlite3.connect(DATABASE_PATH) as connection:
        rows = connection.execute(
            "SELECT memory_key, memory_value FROM memories ORDER BY id DESC"
        ).fetchall()

    if not rows:
        return "No matching memories found."

    lines = "\n".join(f"- {key}: {value}" for key, value in rows)
    return f"Found {len(rows)} matching memory record(s):\n{lines}"


def group_memories(tool_result: str) -> dict:
    """
    Parse '- key: value' lines (newest first) into {key: [values]}.
    Single-value keys keep only the NEWEST value, so 'Alex' replaces
    the older 'Dunura' instead of confusing the answer.
    """

    grouped = {}

    for key, value in re.findall(r"^- ([^:]+): (.*)$", tool_result, flags=re.M):
        key, value = key.strip(), value.strip()
        values = grouped.setdefault(key, [])

        if key in SINGLE_VALUE_KEYS:
            if not values:
                values.append(value)
        elif value not in values:
            values.append(value)

    return grouped


def format_memory_answer(query: str, tool_result: str) -> str:
    grouped = group_memories(tool_result)

    if not grouped:
        hint = MEMORY_HINTS.get(query, "My name is ...")
        return f'I don\'t have that stored yet. You can tell me, e.g. "{hint}"'

    # "What do you remember about me?" -> full profile
    if query == ALL_MEMORIES:
        lines = ["Here is what I remember about you:"]
        for key, values in grouped.items():
            label = MEMORY_LABELS.get(key, key.capitalize())
            lines.append(f"- {label}: {', '.join(values)}")
        return "\n".join(lines)

    # Specific question -> natural sentence
    sentences = []
    for key, values in grouped.items():
        template = MEMORY_SENTENCES.get(key)
        value_text = ", ".join(values)
        sentences.append(
            template.format(v=value_text) if template else f"{key}: {value_text}"
        )

    return " ".join(sentences)


def format_calculator_answer(expression: str, tool_result: str) -> str:
    match = re.search(r"Result:\s*(.+)$", tool_result)

    if not match:
        return f"Sorry, I couldn't calculate that. ({tool_result})"

    return f"{expression} = {match.group(1).strip()}"

# ============================================================
# MEMORY UPDATES ("update my name as ...", "change my city to ...")
# ============================================================

UPDATE_PATTERNS = [
    (r"\b(?:update|change|set|correct)\s+my\s+name\s+(?:to|as|into)\s+(.+)", "name"),
    (r"\bmy\s+(?:new|real|actual|correct)\s+name\s+is\s+(.+)", "name"),   # <-- new
    (r"\bmy\s+name\s+is\s+now\s+(.+)", "name"),                           # <-- new
    (r"\b(?:i am|i'm)\s+called\s+(.+)", "name"),                          # <-- new
    (r"\bcall me\s+(.+)", "name"),
    (r"\b(?:update|change|set)\s+my\s+(?:location|city)\s+(?:to|as)\s+(.+)", "location"),
    (r"\b(?:update|change|set)\s+my\s+(?:job|occupation)\s+(?:to|as)\s+(.+)", "occupation"),
    (r"\b(?:update|change|set)\s+my\s+(?:study|course|degree)\s+(?:to|as)\s+(.+)", "study"),
]


def prune_old_memories(memory_key: str):
    """Keep only the newest row for a single-value key (name, location...)."""

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            DELETE FROM memories
            WHERE memory_key = ?
              AND id < (SELECT MAX(id) FROM memories WHERE memory_key = ?)
            """,
            (memory_key, memory_key),
        )
        connection.commit()


def save_user_information(user_input: str):
    """Save new or updated personal facts; old single-value facts are replaced."""

    text = user_input.strip()
    result = None

    for pattern, key in UPDATE_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip().rstrip(".!?")
            if value and len(value) <= 300:
                result = save_memory(key, value)
            break

    if result is None:
        result = extract_and_save_user_information(text)

    if result:
        saved_key = re.match(r"Saved memory: (\w+) =", result)
        if saved_key and saved_key.group(1) in SINGLE_VALUE_KEYS:
            prune_old_memories(saved_key.group(1))

    return result

def prune_all_single_value_memories():
    """One-time cleanup: keep only the newest row for name, location, etc."""
    for key in SINGLE_VALUE_KEYS:
        prune_old_memories(key)

CAPABILITIES_TEXT = """\
I can help with:
1. Calculations   - e.g. "Calculate 250 multiplied by 4"
2. Memory         - I remember things you tell me, e.g. "My name is Alex",
                    then ask "What is my name?"
3. University docs - e.g. "What are the main modules in the Computer
                    Science degree programme?"
I can also chat about general topics."""


# ============================================================
# 12. TOOL EXECUTION
# ============================================================

_rag_cache = {}


def execute_tool(tool_name: str, arguments: dict) -> str:
    """Run a tool safely, with logging and RAG caching."""

    print("\n[TOOL SELECTED]", tool_name)
    print("[TOOL ARGUMENTS]", arguments)

    try:
        if tool_name == "memory_search_tool" and arguments.get("query") == ALL_MEMORIES:
            result = fetch_all_memories()

        elif tool_name == "rag_search_tool":
            key = arguments.get("question", "").strip().lower()
            if key in _rag_cache:
                result = _rag_cache[key]
            else:
                result = tool_map[tool_name].invoke(arguments)
                if not result.startswith("RAG search error"):
                    _rag_cache[key] = result

        elif tool_name in tool_map:
            result = tool_map[tool_name].invoke(arguments)

        else:
            result = f"Unknown tool: {tool_name}"

    except Exception as error:
        result = f"Tool execution failed: {error}"

    print("\n[TOOL RESULT]")
    print(result)
    return str(result)


# ============================================================
# 13. LLM PATH (for requests the router does not recognise)
# ============================================================

SYSTEM_PROMPT = """
You are a helpful assistant.
Answer general questions and chat directly, clearly and concisely.
Never invent personal information about the user.
Never invent facts about the university or its documents.
"""


def run_llm_with_tools(conversation_history: list) -> str:
    """General chat: plain LLM, no tools (the router already handles tools)."""
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        *conversation_history,
    ])
    return response.content


# ============================================================
# 14. AGENT EXECUTION
# ============================================================

def respond(answer: str, conversation_history: list):
    """Print the answer, store it in history and trim old messages."""

    print("\nAGENT RESPONSE:")
    print(answer)

    conversation_history.append(AIMessage(content=answer))

    if len(conversation_history) > MAX_HISTORY_MESSAGES:
        del conversation_history[:-MAX_HISTORY_MESSAGES]


def run_agent(user_input: str, conversation_history: list):

    user_input = normalise_input(user_input)

    print("\n" + "=" * 65)
    print("USER:", user_input)
    print("=" * 65)

    # Save personal facts (e.g. "My name is Alex") using Part B logic.
    saved_result = save_user_information(user_input)
    if saved_result:
        print("\n[MEMORY SAVED]", saved_result)

    conversation_history.append(HumanMessage(content=user_input))

    # A pure statement ("My name is Alex.") only needs an acknowledgement.
    if saved_result and "?" not in user_input:
        respond(f"Got it! {saved_result.replace('Saved memory: ', 'I saved: ')}.",
                conversation_history)
        return

    routed = route(user_input)

    # ---------------- Router-handled requests ----------------
    if routed:
        tool_name, arguments = routed

        if tool_name == "capabilities":
            respond(CAPABILITIES_TEXT, conversation_history)
            return

        result = execute_tool(tool_name, arguments)

        if tool_name == "calculator_tool":
            answer = format_calculator_answer(arguments["expression"], result)

        elif tool_name == "memory_search_tool":
            answer = format_memory_answer(arguments["query"], result)

        else:  # rag_search_tool: already an LLM answer + sources
            answer = result.replace("RAG Search Result:\n", "", 1)

        respond(answer, conversation_history)
        return

    # ---------------- LLM-handled requests ----------------
    respond(run_llm_with_tools(conversation_history), conversation_history)


# ============================================================
# 15. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 65)
    print("             LAB 06 - PART C")
    print("              RAG-POWERED AI AGENT")
    print("=" * 65)

    print("\nAvailable Tools:")
    print("1. Calculator Tool")
    print("2. Memory Search Tool (SQLite)")
    print("3. RAG Search Tool (Lab 05 ChromaDB)")
    print("\nType 'exit' to stop the agent.")
    prune_all_single_value_memories()

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