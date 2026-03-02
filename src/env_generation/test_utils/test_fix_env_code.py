import subprocess
import sys
import tempfile
import json
import re
import os
import ast
from pathlib import Path
from typing import Dict, Any

from src.utils.llm import call_llm,extract_code

def extract_mcp_tools_from_code(code_str: str) -> list[str]:
    """
    Given a Python source code string, return a list of tool names 
    decorated with @mcp.tool(...). When the decorator has no parameters, 
    the function name is used by default; if the first parameter of the 
    decorator is a string, the alias is used first.
    """
    tree = ast.parse(code_str)
    tools: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if (
                    isinstance(dec, ast.Call)            # @xxx(...)
                    and isinstance(dec.func, ast.Attribute)
                    and dec.func.attr == "tool"          # .tool
                ):
                    # @mcp.tool()              
                    if not dec.args:
                        tools.append(node.name)
                    # @mcp.tool("alias")       
                    elif (
                        len(dec.args) == 1
                        and isinstance(dec.args[0], ast.Constant)
                        and isinstance(dec.args[0].value, str)
                    ):
                        tools.append(dec.args[0].value)
                    else:
                        tools.append(node.name)         
    return tools

def extract_registry_from_plan(plan: Dict[str, Any]) ->Dict[str, Any]:
    if "files_path" not in plan:
        return {}
    registry_file_path = plan["files_path"][0]
    json_file = json.load(open(registry_file_path))
    return json_file

# Directory to save code and tests for inspection
DEBUG_SAVE_DIR = Path("debug_env")
DEBUG_SAVE_DIR.mkdir(exist_ok=True)

# Read the UnifiedBaseEnv template
_TEMPLATE_DIR = Path(__file__).resolve().parent
base_env_path = _TEMPLATE_DIR / "unified_base_env_test.py"
with open(base_env_path, "r", encoding="utf-8") as f:
    base_env_template = f.read()

# Read the UnifiedBaseEnv template
base_single_env_path = _TEMPLATE_DIR / "unified_single_env_test.py"
with open(base_single_env_path, "r", encoding="utf-8") as f:
    base_single_env_template = f.read()

# Improved system prompt for dynamic pytest generation
TEST_GEN_SYSTEM_PROMPT = f"""
You are a code-gen agent that automatically writes pytest files for any MCP-based
environment class.

TEMPLATE TEST CODE TO EXTEND:
```python
{base_env_template}
```

MCP server test code integration requirements
• Keep the same file headers and fixtures as defined in the base template.
• Use the same import statements and DummyMCP patch fixtures without modification.
• Do not add or remove any auxiliary functions: `_get_underlying_env`, `_exec`, `_state`, and `reset_state`.
• Generate a test function named `test_<tool_name>` for each tool in the environment tool list.
• Each test function must follow the following three-step pattern:
    1. Get a random state through `s = _state(env)`.
    2. Construct the `args` dictionary with the values ​​required by the tool-scheme and the functions implemented in the Environment code.
    3. Call the tool using `_exec(env, "<tool_name>", args)` and assert the expected keys. The tool_name here must be the same as the tool function in the Environment code
    4. Only can use _exec to call <tool_name> like: out = _exec(env, <tool_name>, rgs)
• Use double quotes only for strings and JSON payloads.
• Leave exactly two blank lines after the auxiliary fixture section and before the first test.
• Leave exactly one blank line between subsequent test functions.
• Do not include any extra imports, comments, or text outside of boilerplate code and test definitions.

**Additional result validation rules (apply inside each test function)**
• If the tool returns a scalar (str/int/bool), assert that it is exactly equal to the documented literal.
• If the tool returns a dictionary:
    – Assert that all required top-level keys are present.
    – For each key, assert its value type (e.g. list/str/int).
    – **Use** `out.get("<key>") == <expected>` **rather than** `out == <expected dict>`  
    – When the spec defines a deterministic value (e.g. "status": "OK"), assert equality.
    – When the spec allows multiple items (e.g. "members": [...]):
* When the input is guaranteed to match, assert that the list is non-empty.
* Assert that each element satisfies the filter implied by the input status or arguments.
    • Never query external state to verify side effects; assertions are strictly limited to the returned payload.
    • Use only built-in `assert` and `isinstance`, no need to use additional libraries.
    • All identifiers, comments, and docstrings must be written in English only.
    • File usage rule – Any argument representing a file ID or path **must**
    come from the list provided in EVAL FILES PATH JSON.  
    Never invent new filenames or directories that are not in that list.

"""


def generate_test_module(env_code: str, eval_files: Dict[str, Any], mcp_data: Dict[str, Any]) -> str:
    """
    Use LLM to generate a pytest-compatible test module for env_module.py based on eval_data.
    """    
    tools_name = extract_mcp_tools_from_code(env_code)
    
    prompt = f"""
    
    ```python
    {env_code}
    ```
    
    EVAL FILES PATH JSON:
    ```json
    {json.dumps(eval_files, indent=2)}
    ```
    
    MCP data JSON:
    ```json
    {json.dumps(mcp_data, indent=2)}
    ```


IMPLEMENTATION GUIDELINES:
•	Preserve the boilerplate fixtures and helpers exactly as defined in the system prompt.
•	For each tool in the environment ({','.join(tools_name)}), generate one pytest function named test_<tool_name>.
•	Use only double quotes for strings and JSON.
    
    Generate test_env.py as pytest module following the system prompt instructions.
    """
    # Call LLM
    raw = call_llm(
        text=prompt,
        system_prompt=TEST_GEN_SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=4000,
    )
    tests_code = extract_code(raw)
    return tests_code



# Improved system prompt for dynamic pytest generation
TEST_GEN_SINGLE_SYSTEM_PROMPT = f"""
You are a code-gen agent that automatically writes pytest files for any MCP-based
environment class.

TEMPLATE TEST CODE TO EXTEND:
```python
{base_single_env_template}
```

MCP server test code integration requirements
• Keep the same file headers and fixtures as defined in the base template.
• Use the same import statements and DummyMCP patch fixtures without modification.
• Do not add or remove any auxiliary functions: `_get_underlying_env`, `_exec`, `_state`, and `reset_state`.
• Generate a test function named `test_<tool_name>` for each tool in the environment tool list.
• Each test function must follow the following three-step pattern:
    1. Get a random state through `s = _state(env)`.
    2. Construct the `args` dictionary with the values ​​required by the tool-scheme and the functions implemented in the Environment code.
    3. Call the tool using `_exec(env, "<tool_name>", args)` and assert the expected keys. The tool_name here must be the same as the tool function in the Environment code
    4. Only can use _exec to call <tool_name> like: out = _exec(env, <tool_name>, rgs)
• Use double quotes only for strings and JSON payloads.
• Leave exactly two blank lines after the auxiliary fixture section and before the first test.
• Leave exactly one blank line between subsequent test functions.
• Do not include any extra imports, comments, or text outside of boilerplate code and test definitions.

**Additional result validation rules (apply inside each test function)**
• If the tool returns a scalar (str/int/bool), assert that it is exactly equal to the documented literal.
• If the tool returns a dictionary:
    – Assert that all required top-level keys are present.
    – For each key, assert its value type (e.g. list/str/int).
    – **Use** `out.get("<key>") == <expected>` **rather than** `out == <expected dict>`  
    – When the spec defines a deterministic value (e.g. "status": "OK"), assert equality.
    – When the spec allows multiple items (e.g. "members": [...]):
* When the input is guaranteed to match, assert that the list is non-empty.
* Assert that each element satisfies the filter implied by the input status or arguments.
    • Never query external state to verify side effects; assertions are strictly limited to the returned payload.
    • Use only built-in `assert` and `isinstance`, no need to use additional libraries.
    • All identifiers, comments, and docstrings must be written in English only.
    • File usage rule – Any argument representing a file ID or path **must**
    come from the list provided in EVAL FILES PATH JSON.
    Never invent new filenames or directories that are not in that list.

"""

def gen_single_test_module(env_code: str, mcp_data: Dict[str, Any]) -> str:
    """
    Use LLM to generate a pytest-compatible test module for env_module.py based on eval_data.
    """    
    tools_name = extract_mcp_tools_from_code(env_code)
    
    prompt = f"""
    
    ```python
    {env_code}
    ```
    
    MCP data JSON:
    ```json
    {json.dumps(mcp_data, indent=2)}
    ```


IMPLEMENTATION GUIDELINES:
•	Preserve the boilerplate fixtures and helpers exactly as defined in the system prompt.
•	For each tool in the environment ({','.join(tools_name)}), generate one pytest function named test_<tool_name>.
•	Use only double quotes for strings and JSON.
    
    Generate test_env.py as pytest module following the system prompt instructions.
    """
    # Call LLM
    raw = call_llm(
        text=prompt,
        system_prompt=TEST_GEN_SINGLE_SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=4000,
    )
    tests_code = extract_code(raw)
    return tests_code

def run_pytest(test_module_path: Path) -> Dict[str, Any]:
    """
    Run pytest on test_module_path, capture success and console output.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_module_path), "--maxfail=1", "-q"],
        capture_output=True,
        text=True
    )
    # print(f"stdout: {proc.stdout}")
    return {"success": proc.returncode == 0, "stdout": proc.stdout, "stderr": proc.stderr}


def evaluate_and_patch(
    env_code: str,
    gen_code_system_prompt: str,
    test_results: Dict[str, Any],
) -> str:
    """
    Ask the LLM to repair env_module.py until all pytest cases pass.

    Parameters
    ----------
    env_code : str
        Current, **failing** environment source code.
    gen_code_system_prompt : str
        The original system-prompt that was used to generate the file.
        It contains style guides and functional constraints that we
        still have to respect during the fix-up stage.
    test_results : Dict[str, Any]
        The parsed pytest output returned by run_pytest().

    Returns
    -------
    str
        A fully-corrected `env_module.py` source string.  The caller is
        responsible for saving it and re-running the test loop.
    """
    # ------------------------------------------------------------------ #
    # 1) Build an amplified, self-contained system prompt                #
    # ------------------------------------------------------------------ #
    fix_system_prompt = f"""
You are a senior simulation architect and code fixer.

Goal:
----
Ensure that the provided `env_module.py` passes all its pytest tests without modifying any test code.

Context:
-------
Below is the original code-generation system prompt that defined style, structure, and requirements for the environment module. **You must preserve every rule, naming convention, import structure, and docstring style it specifies.**

{gen_code_system_prompt}

Inputs:
-------
1. **Pytest failure report** (stdout and stderr) showing which tests failed and why.  
2. **Current `env_module.py` source** that failed those tests.

Instructions:
-------------
1. **Analyze the pytest output** to pinpoint each failure (assertion errors, JSONDecodeError, missing attributes, etc.).  
2. **Read the full source** of `env_module.py` to understand existing logic, file-system usage, and MCP tool implementations.  
3. **Apply minimal, targeted fixes** so that every test passes:
   - Correct mis-uses of `json.load` vs. `json.loads` when parsing strings.
   - Ensure all MCPFileSystem calls consume the right parameters (`save_text`, `save_binary`, `update_meta`, `delete_file`, `load_text`, `load_info`).
   - Add any missing methods or adjust return types to satisfy test assertions.
   - Fix state-reset logic to work with `tools` passed as dicts if necessary.
4. **Do not alter or remove any pytest files or test functions**—only change `env_module.py`.
5. **Maintain original imports, class names, decorator usage, and coding style.**  
6. **If you see `JSONDecodeError` in the failures,** replace string-based `json.load(...)` calls with `json.loads(...)` or add safe defaults (`or "[]"` / `or "{{}}"`) as appropriate.
7. **Output requirement:** Return exactly one Python fenced code block containing the entire corrected `env_module.py` source. Do not include any explanatory text, file paths, or markdown beyond the single ```python block.```
"""


    report_json = json.dumps(test_results, indent=2)
    user_prompt = (
    "Pytest results:\nstdout\n" + report_json + "\n"
    "\nCurrent env_module.py code:\npython\n" + env_code + "\n"
    "\nGenerate the full corrected env_module.py inside a "
    "python fenced code block so tests pass."
    )


    response = call_llm(
    text=user_prompt,
    system_prompt=fix_system_prompt,
    temperature=0.0,
    max_tokens=4000,
    )

    corrected_code = extract_code(response)
    return corrected_code




def validate_and_refine(
    env_code: str,
    gen_code_system_prompt: str,
    plan: Dict[str, Any],
    mcp_data: Dict[str, Any],
    tests_code: str = None,
    max_rounds: int = 4,
    print_info: bool = True
) -> str:
    """
    1) If tests_code is None, generate and save initial test_env.py via LLM.
    2) Loop for up to max_rounds:
       - Write env_module.py (initial or corrected) and test_env.py to temp dir
       - Run pytest
       - If tests pass, save final env_module.py and return env_code
       - Else, call evaluate_and_patch to get new env_code and iterate
    3) Raise if still failing after max_rounds.

    Args:
        env_code: Current environment source code.
        gen_code_system_prompt: Original system prompt used for generation.
        plan: Simulation plan dict.
        mcp_data: MCP server metadata and tools.
        tests_code: Optional pre-generated test module code.
        max_rounds: Maximum number of refine iterations.
        print_info: If True, print progress and pytest output.

    Returns:
        The final, passing env_code string.

    Raises:
        RuntimeError: If tests still fail after max_rounds.
    """
    # Generate and save initial test module if needed
    # save_inputs = False
    # if save_inputs:
    #     (DEBUG_SAVE_DIR / "env_module_initial.py").write_text(env_code, encoding='utf-8')
    #     (DEBUG_SAVE_DIR / "plan.json").write_text(json.dumps(plan, indent=2), encoding='utf-8')
    #     (DEBUG_SAVE_DIR / "mcp_data.json").write_text(json.dumps(mcp_data, indent=2), encoding='utf-8')

    server_name = mcp_data["server_name"].replace(" ", "_").replace("-", "_")
    server_path = Path(DEBUG_SAVE_DIR / server_name)
    server_path.mkdir(parents=True, exist_ok=True)
    eval_files_map = extract_registry_from_plan(plan)
    tests_code = tests_code or generate_test_module(env_code, eval_files_map, mcp_data)
    
    for round_idx in range(1, max_rounds + 1):
        if print_info:
            print(f"[Round {round_idx}/{max_rounds}] Validating...")

        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            env_path = workdir / "env_module_test.py"
            test_path = workdir / "test_env.py"

            # Write environment module and tests
            env_path.write_text(env_code, encoding='utf-8')
            test_path.write_text(tests_code, encoding='utf-8')

            # Save working copies for inspection
            (server_path / f"env_module_test.py").write_text(env_code, encoding='utf-8')
            (server_path / f"test_env.py").write_text(tests_code, encoding='utf-8')

            # Adjust import path
            sys.path.insert(0, str(workdir))

            try:
                results = run_pytest(test_path)
                if print_info:
                    print(results["stdout"])

                if results["success"]:
                    if print_info:
                        print("✅ Tests passed.")
                    final_path = server_path / "env_module_final.py"
                    final_path.write_text(env_code, encoding='utf-8')
                    return env_code

                if print_info:
                    print("❌ Tests failed. Requesting fix...")
                env_code = evaluate_and_patch(env_code, gen_code_system_prompt, results)
            finally:
                # Clean up import path
                if str(workdir) in sys.path:
                    sys.path.remove(str(workdir))

    # After exhausting rounds without success
    (server_path / "env_module_failed.py").write_text(env_code, encoding='utf-8')
    raise RuntimeError("Validation failed after multiple rounds: tests still failing.")

def only_validate(
    env_code: str,
    mcp_data: Dict[str, Any],
) -> None:
    """
    1) If tests_code is None, generate and save initial test_env.py via LLM.
    2) Loop for up to max_rounds:
       - Write env_module.py (initial or corrected) and test_env.py to temp dir
       - Run pytest
       - If tests pass, save final env_module.py and return env_code
       - Else, call evaluate_and_patch to get new env_code and iterate
    3) Raise if still failing after max_rounds.

    Args:
        env_code: Current environment source code.
        mcp_data: MCP server metadata and tools.

    Returns:
        None

    Raises:
        RuntimeError: If tests still fail after max_rounds.
    """
    debug_single_save_dir = Path("debug_single_env")
    server_name = mcp_data["server_name"].replace(" ", "_").replace("-", "_")
    server_path = Path(debug_single_save_dir / server_name)
    server_path.mkdir(parents=True, exist_ok=True)
    tests_code =  gen_single_test_module(env_code, mcp_data)
    
    (server_path / f"env_module_test.py").write_text(env_code, encoding='utf-8')
    (server_path / f"test_env.py").write_text(tests_code, encoding='utf-8')




# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Validate and refine environment code.")
#     parser.add_argument(
#         "--mode", choices=["normal", "test"], default="normal",
#         help="normal: generate code, save debug inputs, and validate; test: rerun validation using saved debug inputs and existing tests"
#     )
#     args = parser.parse_args()

#     if args.mode == "normal":
#         env_code = open(DEBUG_SAVE_DIR / "env_module_initial.py", encoding='utf-8').read()
#         plan = json.loads(open(DEBUG_SAVE_DIR / "plan.json", encoding='utf-8').read())
#         mcp_data = json.loads(open(DEBUG_SAVE_DIR / "mcp_data.json", encoding='utf-8').read())
#         validate_and_refine(env_code, plan, mcp_data)
#     else:
#         # Test mode: reuse initial test module without regenerating
#         env_code = open(DEBUG_SAVE_DIR / "env_module_initial.py", encoding='utf-8').read()
#         plan = json.loads(open(DEBUG_SAVE_DIR / "plan.json", encoding='utf-8').read())
#         mcp_data = json.loads(open(DEBUG_SAVE_DIR / "mcp_data.json", encoding='utf-8').read())
#         tests_code = open(DEBUG_SAVE_DIR / "test_env_initial.py", encoding='utf-8').read()
#         validate_and_refine(env_code, plan, mcp_data, tests_code=tests_code)
