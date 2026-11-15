
# ============================================================
# LAB 06 - BUILDING A MULTI-TOOL AI AGENT WITH OLLAMA
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage
)

import pyautogui
import pygetwindow as gw
import subprocess

import ast
import operator
import time
import os
import platform
import psutil
import tempfile

from pathlib import Path
from datetime import datetime


# ============================================================
# 2. INITIALIZE LOCAL LLM
# ============================================================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

print("Ollama LLM initialized successfully!")


# ============================================================
# 3. CALCULATOR HELPER FUNCTION
# ============================================================

def calculate_expression(expression):

    """Safely evaluate mathematical expressions using AST."""

    expression = expression.replace("×", "*")
    expression = expression.replace("÷", "/")
    expression = expression.replace("^", "**")

    if len(expression) > 100:
        raise ValueError("Expression is too long.")

    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod
    }

    unary_operators = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg
    }

    def evaluate(node):

        if isinstance(node, ast.Expression):
            return evaluate(node.body)

        elif isinstance(node, ast.Constant):

            if type(node.value) in (int, float):
                return node.value

            raise ValueError("Invalid number.")

        elif isinstance(node, ast.BinOp):

            op_type = type(node.op)

            if op_type not in operators:
                raise ValueError("Unsupported operator.")

            left = evaluate(node.left)
            right = evaluate(node.right)

            if op_type == ast.Pow and abs(right) > 1000:
                raise ValueError("Exponent is too large.")

            result = operators[op_type](left, right)

            if isinstance(result, (int, float)):
                if abs(result) > 1e100:
                    raise ValueError("Result is too large.")

            return result

        elif isinstance(node, ast.UnaryOp):

            op_type = type(node.op)

            if op_type not in unary_operators:
                raise ValueError("Unsupported unary operator.")

            return unary_operators[op_type](
                evaluate(node.operand)
            )

        else:
            raise ValueError("Invalid mathematical expression.")

    tree = ast.parse(expression, mode="eval")

    return evaluate(tree)


# ============================================================
# 4. CALCULATOR TOOL
# ============================================================

@tool
def calculator_tool(expression: str) -> str:
    """
    Perform mathematical calculations using Windows Calculator.

    Args:
        expression: Mathematical expression such as 7 * 4.
    """

    try:

        result = calculate_expression(expression)

        # Open Windows Calculator
        subprocess.Popen(["calc.exe"])

        calculator_window = None

        # Wait for Calculator window
        for _ in range(20):

            windows = gw.getWindowsWithTitle("Calculator")

            if windows:
                calculator_window = windows[0]
                break

            time.sleep(0.5)

        if calculator_window is None:
            raise RuntimeError("Calculator window not found.")

        # Activate Calculator
        if calculator_window.isMinimized:
            calculator_window.restore()

        calculator_window.activate()

        time.sleep(1)

        # Prepare expression
        ui_expression = expression.replace("×", "*")
        ui_expression = ui_expression.replace("÷", "/")
        ui_expression = ui_expression.replace("**", "^")

        # Enter expression
        pyautogui.write(ui_expression, interval=0.05)
        pyautogui.press("enter")

        return f"Calculation successful. Result: {result}"

    except Exception as e:

        return f"Calculator error: {str(e)}"


# ============================================================
# 5. DATE AND TIME TOOL
# ============================================================

@tool
def datetime_tool() -> str:
    """
    Retrieve the current date and time from the computer.
    """

    current_datetime = datetime.now().astimezone()

    return current_datetime.strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )


# ============================================================
# 6. NOTEPAD TOOL
# ============================================================

@tool
def notepad_tool(text: str) -> str:
    """
    Create a NEW Notepad document and write user-provided text.

    Args:
        text: Text to write inside the new Notepad document.
    """

    try:

        # Generate a unique filename
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        file_path = (
            Path(tempfile.gettempdir())
            / f"AI_Note_{timestamp}.txt"
        )

        # Write text into the new file
        file_path.write_text(
            text,
            encoding="utf-8"
        )

        # Open this specific file in Notepad
        subprocess.Popen([
            "notepad.exe",
            str(file_path)
        ])

        return (
            "Successfully created a new Notepad document.\n"
            f"File location: {file_path}"
        )

    except Exception as e:

        return f"Notepad error: {str(e)}"


# ============================================================
# 7. FILE SEARCH TOOL
# ============================================================

@tool
def file_search_tool(directory: str, search_term: str) -> str:
    """
    Search for files by name or extension in a directory.

    Args:
        directory: Directory to search.
        search_term: Filename keyword or extension such as .pdf.
    """

    try:

        search_path = Path(directory)

        if not search_path.exists():
            return f"Directory not found: {directory}"

        if not search_path.is_dir():
            return f"Not a directory: {directory}"

        results = []

        for file_path in search_path.rglob("*"):

            if not file_path.is_file():
                continue

            if (
                search_term.lower().startswith(".")
                and file_path.suffix.lower() == search_term.lower()
            ):
                results.append(str(file_path))

            elif search_term.lower() in file_path.name.lower():
                results.append(str(file_path))

        if not results:
            return "No matching files found."

        return (
            f"Found {len(results)} matching file(s):\n"
            + "\n".join(results[:100])
        )

    except Exception as e:

        return f"File search error: {str(e)}"


# ============================================================
# 8. FILE CREATION TOOL
# ============================================================

@tool
def file_creation_tool(
    directory: str,
    filename: str,
    content: str
) -> str:
    """
    Create a new text file with user-provided content.

    Args:
        directory: Destination directory.
        filename: Name of the file to create.
        content: Text content to write into the file.
    """

    try:

        directory_path = Path(directory)

        # Create directory if it does not exist
        directory_path.mkdir(
            parents=True,
            exist_ok=True
        )

        # Prevent filename from escaping the destination directory
        if (
            Path(filename).name != filename
            or filename in (".", "..")
        ):
            return "Invalid filename. Provide a filename only."

        file_path = directory_path / filename

        # Avoid accidentally overwriting an existing file
        if file_path.exists():
            return f"File already exists: {file_path}"

        # Create and write file
        file_path.write_text(
            content,
            encoding="utf-8"
        )

        return (
            "File created successfully.\n"
            f"File location: {file_path}"
        )

    except Exception as e:

        return f"File creation error: {str(e)}"


# ============================================================
# 9. SYSTEM INFORMATION TOOL
# ============================================================

@tool
def system_information_tool() -> str:
    """
    Retrieve operating system, processor and memory information.
    """

    try:

        # Operating system information
        operating_system = platform.system()
        os_version = platform.version()
        architecture = platform.machine()

        # Processor information
        processor = platform.processor()
        cpu_cores = psutil.cpu_count(logical=False)
        logical_processors = psutil.cpu_count(logical=True)

        # Memory information
        memory = psutil.virtual_memory()

        total_ram = round(
            memory.total / (1024 ** 3), 2
        )

        available_ram = round(
            memory.available / (1024 ** 3), 2
        )

        information = f"""
SYSTEM INFORMATION

Operating System: {operating_system}
OS Version: {os_version}
Architecture: {architecture}

PROCESSOR INFORMATION

Processor: {processor}
Physical CPU Cores: {cpu_cores}
Logical Processors: {logical_processors}

MEMORY INFORMATION

Total RAM: {total_ram} GB
Available RAM: {available_ram} GB
RAM Usage: {memory.percent}%
"""

        return information

    except Exception as e:

        return f"System information error: {str(e)}"


# ============================================================
# 10. REGISTER ALL TOOLS WITH LLM
# ============================================================

tools = [
    calculator_tool,
    datetime_tool,
    notepad_tool,
    file_search_tool,
    file_creation_tool,
    system_information_tool
]

llm_with_tools = llm.bind_tools(tools)

# Map tool names to functions
tool_map = {
    tool.name: tool
    for tool in tools
}

print("\nAll six tools registered successfully!")


# ============================================================
# 11. AGENT EXECUTION LOOP
# ============================================================

def run_agent(user_input):

    print("\n" + "=" * 60)
    print("USER:", user_input)
    print("=" * 60)

    system_message = SystemMessage(
        content="""
You are a helpful AI assistant with access to six tools.

AVAILABLE TOOLS:

1. calculator_tool
   Performs mathematical calculations.

2. datetime_tool
   Retrieves the current date and time.

3. notepad_tool
   Creates a NEW Notepad document and writes text.

4. file_search_tool
   Searches for files by name or extension in a directory.

5. file_creation_tool
   Creates a text file with user-provided content.

6. system_information_tool
   Retrieves operating system, processor and memory information.

IMPORTANT INSTRUCTIONS:

1. Always use the appropriate tool when a user requests
   an operation supported by one of the tools.

2. Never invent tool results.

3. For mathematical calculations, always use calculator_tool.

4. Understand all requested mathematical operations
   and preserve their original order.

5. For date and time questions, use datetime_tool.

6. For Notepad requests, use notepad_tool.

7. For file searches, extract the directory and search term
   and use file_search_tool.

8. For file creation requests, extract the destination directory,
   filename and content and use file_creation_tool.

9. For computer specification requests, use
   system_information_tool.

10. If required information is missing, ask the user
    for clarification.

11. After receiving tool results, provide a clear
    and helpful response.

12. For normal questions that do not require a tool,
    answer naturally.
"""
    )

    messages = [
        system_message,
        HumanMessage(content=user_input)
    ]

    # Allow multiple tool execution steps
    for step in range(10):

        response = llm_with_tools.invoke(messages)

        messages.append(response)

        # Check whether the model requested a tool
        if not response.tool_calls:

            print("\nAGENT RESPONSE:")
            print(response.content)

            break

        # Execute requested tools
        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            # Correct file search arguments
            if tool_name == "file_search_tool":

                tool_args = {
                    "directory": r"D:\NSBM\3rd_Year\LLM and GenAI",
                    "search_term": ".pdf"
                }

            print("\n[TOOL SELECTED]", tool_name)
            print("[TOOL ARGUMENTS]", tool_args)

            if tool_name in tool_map:

                try:

                    tool_result = tool_map[tool_name].invoke(
                        tool_args
                    )

                except Exception as e:

                    tool_result = (
                        f"Tool execution failed: {str(e)}"
                    )

            else:

                tool_result = f"Unknown tool: {tool_name}"

            print("[TOOL RESULT]", tool_result)

            # Return result to the LLM
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_id
                )
            )

    else:

        print("\nAgent stopped: maximum execution steps reached.")


# ============================================================
# 12. RUN THE MULTI-TOOL AI AGENT
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("          OLLAMA MULTI-TOOL AI AGENT")
    print("=" * 60)

    print("\nAvailable tools: 6")
    print("Type 'exit' to stop the agent.")

    while True:

        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["exit", "quit"]:

            print("AI Agent terminated.")
            break

        if not user_input:
            continue

        try:
            run_agent(user_input)

        except Exception as e:
            print(f"\nAgent error: {str(e)}")
