"""
env_eval.py  —  Unified evaluator for MCP environment code.

Metrics
-------
1. Executability        (syntax + import)
2. Schema Fidelity      (function + signature match)
3. Functionality        (LLM audit of each tool)
4. Semantic Fidelity    (schema vs. code embedding cosine)
5. Truthfulness / Realism
   (run dummy call for each tool, let LLM judge “像真 API”)

Author: ChatGPT 2025-07-05
"""
from __future__ import annotations

import importlib.util
import json
import math
import py_compile
import re
import subprocess
import textwrap
import sys
import traceback
from pathlib import Path
import ast
import os
from typing import Dict, List, Tuple, Any,Union

# === external helpers (your project) ========================================

from src.utils.llm import call_llm, extract_code, call_embedding
from src.env_generation.utils import extract_tool_defs,extract_method
# =============================================================================
# Weights (adjust as needed)
# =============================================================================
WEIGHTS = {
    "executability":      0.15,
    "schema_fidelity":    0.10,
    "functionality":      0.30,
    "semantic_fidelity":  0.15,
    "truthfulness":       0.30,
}

# -----------------------------------------------------------------------------
# General-purpose sanitizers
# -----------------------------------------------------------------------------
def sanitise_tool_name(name: str) -> str:
    """
    Normalize a tool name into snake_case, handling Case, acronyms, and non-alphanumeric separators.

    Examples:
      'Standard I/O Transport Mode' -> 'standard_io_transport'
    """
    name = name.replace("/", "")
    # 1) Replace any sequence of non-word characters with underscore
    s = re.sub(r"[^\w]+", "_", name)
    # 2) Collapse multiple underscores and strip leading/trailing
    s = re.sub(r"_+", "_", s).strip("_")

    # 3) Build result, capturing uppercase acronyms vs. regular letters
    res: List[str] = []
    i = 0
    while i < len(s):
        # At start of string or after underscore, check for acronym
        if i == 0 or s[i-1] == "_":
            # Look ahead for consecutive uppercase letters
            j = i
            while j < len(s) and s[j].isupper():
                j += 1
            # If we found an acronym of length ≥2, lowercase it as a chunk
            if j - i > 1:
                res.append(s[i:j].lower())
                i = j
                continue
            # Else treat single uppercase letter as lowercase
            res.append(s[i].lower())
            i += 1
            continue

        # Otherwise lowercase the character
        res.append(s[i].lower())
        i += 1

    result = "".join(res)

    # 4) Remove trailing "_mode" if present
    if result.endswith("_mode"):
        result = result[: -len("_mode")]

    return result


def sanitise_server_key(name: str) -> str:
    return re.sub(r"[^\w]+", "_", name).strip("_")

# =============================================================================
# 1) Executability
# =============================================================================

def _infer_project_root(code_path: Path) -> Path:
    """
    Heuristic: return the first ancestor that contains a ``src`` folder or
    ``pyproject.toml``. If neither is found, default to two levels above
    the current script (assumed repository root).
    """
    for parent in code_path.parents:
        if (parent / "src").exists() or (parent / "pyproject.toml").exists():
            return parent
    # Fallback: <repo_root>/scripts/...
    return Path(__file__).resolve().parents[2]

def check_executability(code_path: Union[str, Path]) -> Tuple[int, str]:
    """
    Launch the MCP-environment module in a subprocess and judge success by
    whether it stays alive for at least 30 seconds (or exits with code 0).

    Parameters
    ----------
    code_path : str | Path
        Path to a .py file or a package directory containing __init__.py.

    Returns
    -------
    Tuple[int, str]
        * score: 1 if the process booted successfully, 0 otherwise
        * msg  : stderr/stdout snippet or success message
    """
    path = Path(code_path).resolve()
    target = path if path.is_file() else path / "__init__.py"

    # Determine project root and patch cwd / PYTHONPATH
    project_root = _infer_project_root(path)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root}{os.pathsep}{env.get('PYTHONPATH', '')}"

    # Spawn the subprocess
    proc = subprocess.Popen(
        [sys.executable, str(target)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=project_root,          # set working directory to project root
        env=env,                   # ensure project root is on PYTHONPATH
    )

    try:
        stdout, stderr = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        # Process is still running → assume healthy boot
        proc.kill()
        return 1, "Process is alive after 30 s — considered healthy."
    else:
        if proc.returncode == 0:
            return 1, "Process exited normally within 30 s."
        msg = stderr.strip() or stdout.strip() or "Process exited prematurely."
        return 0, msg
# =============================================================================
# 2) Schema Fidelity
# =============================================================================
def _parse_function_defs(src: str) -> Dict[str, List[str]]:
    tree = ast.parse(src)
    out: Dict[str, List[str]] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef):
            out[sanitise_tool_name(n.name)] = [a.arg for a in n.args.args]
    return out

def schema_fidelity_score(code_path: Path,
                          entry: Dict[str, Any]) -> Tuple[float, List[str]]:
    src = code_path.read_text(encoding="utf-8", errors="ignore")
    parsed = _parse_function_defs(src)

    missing: List[str] = []
    total = len(entry.get("tools", []))
    for tool in entry.get("tools", []):
        key = sanitise_tool_name(tool["name"])
        expected = [p["name"] for p in tool.get("parameters", [])]
        actual   = parsed.get(key)
        if actual != expected:
            missing.append(tool["name"])

    score = 1.0 - len(missing) / total if total else 1.0
    return score, missing

# =============================================================================
# 3) Functionality Correctness  (single LLM call for all tools)
# =============================================================================
FUNCTIONALITY_SYSTEM_PROMPT = """
You are a **Functionality-Auditor-GPT**, tasked with rigorously evaluating how thoroughly each MCP tool implementation satisfies its declared business logic.

────────────────────────────────────────  
⚠️ **Important: Tool Name Usage**  
When filling the `"tool"` field in your JSON output:

- **Always use the exact tool name from the TOOL-SCHEME provided** (the headers labeled "<<< TOOL: ToolName >>>").  
- Do **not** use the exact function names from code snippets, even if slightly different.

Correct Example:
{
  "tool": "getBaziDetail",
  "score": 2,
  "reason": "Implements basic data loading but does not use inputs meaningfully."
}

Incorrect Example (uses function name from code):
{
  "tool": "get_bazi_detail",
  "score": 2,
  "reason": "Implements basic data loading but does not use inputs meaningfully."
}

────────────────────────────────────────  

### Scoring Rubric (strict)  
Assign a single integer score (0–5) based solely on **Functionality Correctness**, ignoring side-effects or state mutations:

0 — **Uncallable / Spec Violation**
	•	Code fails to import or execute (syntax/runtime error)
	•	Function signature does not match declared pattern
	•	Call throws exception or returns wrong type

1 — **Stub Implementation (callable but void)**
	•	Contains no meaningful logic or processing beyond basic input validation or trivial formatted returns.
	•	Performs no environment changes (no file creation, no state updates).
	•	Returned data is either hard-coded, fixed strings, or unrelated placeholders, not influenced by actual input.

2 — **Basic Functionality Simulation (incomplete logic)**
	•	Performs initial or superficial environment interactions (e.g., loading files, accessing variables),
but does not utilize or process loaded data meaningfully.
	•	Returned content is superficially formatted and lacks genuine association with actual loaded or processed data.
	•	Environment state updates occur superficially (e.g., assigns values without real processing logic), logic incomplete or placeholder.

3 — **Partially Correct Implementation (≥50% but <100%)**
	•	Implements majority of declared critical business steps correctly, clearly involving meaningful logic and input processing.
	•	At least one important business step or logical branch remains incorrectly implemented or missing.
	•	Produces partially correct or inconsistent results due to incomplete logic.

4 — **Fully Implemented Core Logic (100%)**
	•	Implements **all** declared critical business steps fully and correctly.
	•	Accurately handles normal, typical scenarios; all described logic fully implemented.
	•	Handles typical and boundary inputs correctly; idempotent and consistent results.

5 — **Robust & Hardened Implementation**
	•	Meets all criteria for "4" **and** includes robustness measures:
		- Comprehensive input validation (formats, ranges, permissions).
		- Clear, descriptive error handling and exceptions.
		- Idempotent and consistent results across normal and boundary inputs.
		- Logging, monitoring hooks, or extensive test coverage.

────────────────────────────────────────  

⚠️ **Evaluation Guidance (Critical)**:

When assigning scores, **do NOT rely solely on returned text strings in the code** (e.g., phrases containing "Simulated", "placeholder", "dummy result"). Instead, carefully analyze the underlying logic, considering these factors:

✅ **Indicators of genuine functionality (higher scores)**:
- Actual computation, parsing, or logical processing based on inputs or loaded data.
- Explicit and logical environment-state or file/database updates based on input.
- Robust error handling, input validation, conditional branching, or idempotency mechanisms.
- **Explicit calls to external LLM (`call_llm`) for processing are dynamic and meaningful operations, deserving a minimum score of 3 or higher. However, simply using external LLM calls without additional robustness does NOT merit the highest score of 5.**

❌ **Stub indicators (lower scores)**:
- Purely hard-coded or trivial returns (e.g., directly returning input or formatted placeholder responses).
- Missing or superficial environment-state changes despite explicit implication by the tool description.
- Presence of keywords such as "Simulated", "placeholder", "dummy" strongly suggest absence of real implementation logic.

**Always prioritize evaluating the actual implemented logic and real data-flow demonstrated in the code, rather than plausible-sounding strings or returned texts.**

────────────────────────────────────────  

**Output Format (strict)**  
Return exactly one JSON array, each object containing exactly these keys in order:

[
  {
    "tool": "<ToolName from TOOL-SCHEME>",
    "score": <0–5 integer>,
    "reason": "<≤25 words clearly explaining the primary limitation>"
  },
  ...
]

**Do not output any text beyond this JSON array. Begin your evaluation now.**
"""
def functionality_score_and_details(
    code_path: Path,
    entry: Dict[str, Any],
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Ask the LLM to judge *all* tools in one shot.

    Prompt format:
        <<< TOOL 1 >>>
        description: ...
        code:
        <code block>

        <<< TOOL 2 >>>
        ...

    Expected LLM reply:
    [
      {"tool": "<tool name>", "score": <0–5 integer>, "reason": "<≤25 words on main limitation>"},
      ...
    ]
    """
    src = code_path.read_text(encoding="utf-8", errors="ignore")
    init_block = extract_tool_defs(src)
    # -------- build the mega prompt --------
    blocks: List[str] = [
        f"### _initialize_mcp_server implementation:\n```python\n{init_block}\n```"
    ]
    for idx, tool in enumerate(entry.get("tools", []), 1):
        desc = tool.get("description", "")
        parameters = tool.get("parameters", [])
        blocks.append(
            f"<<< TOOL {idx}: {tool['name']} >>>\n"
            f"Description:\n{desc}\n\n"
            f"Parameters:\n{parameters}\n\n"
        )
        
    joined_sections = "\n\n".join(blocks)
    user_prompt = f'''
    You are a rigorous code auditor. Below you will find several MCP tool implementations,
    each introduced by a header of the form <<< TOOL: ToolName >>>.

    Important reminders:
    - Always use the exact tool names provided in the TOOL-SCHEME ("<<< TOOL: ToolName >>>" headers).
    - Do NOT use exact function names directly from the code snippet.
    - Do NOT rely solely on returned textual outputs (e.g., "Simulated", "placeholder"). Always evaluate the actual logic implemented.

    Below are the tool implementations and their descriptions:

    {joined_sections}

    Evaluate each tool strictly according to the FUNCTIONALITY_SYSTEM_PROMPT provided.
    Return exactly one JSON array. Do NOT provide extra explanations or text.
    '''
    # -------- single LLM call --------
    resp = call_llm(
        text=user_prompt,
        system_prompt=FUNCTIONALITY_SYSTEM_PROMPT,
        max_tokens=2048,
        temperature=0.0,
    )

    # Parse the LLM’s JSON reply
    try:
        raw = extract_code(resp)
        results: List[Dict[str, Any]] = json.loads(raw)
    except Exception:
        # Fallback: mark all tools with zero score if parsing fails
        results = [
            {"tool": t["name"], "score": 0.0, "reason": "ParseError: " + resp}
            for t in entry.get("tools", [])
        ]

    # Ensure every tool has an entry
    details: List[Dict[str, Any]] = []
    seen = {r["tool"] for r in results}
    for t in entry.get("tools", []):
        if t["name"] in seen:
            rec = next(r for r in results if r["tool"] == t["name"])
        else:
            rec = {"tool": t["name"], "score": 0, "reason": "No judgement from LLM"}
        try:
            rec["score"] = max(0, min(5, int(rec.get("score", 0)))) / 5
        except Exception:
            rec["score"] = 0
        details.append(rec)

    # Compute the average score
    avg_score = sum(r["score"] for r in details) / len(details) if details else 0.0

    return avg_score, details

# =============================================================================
# 4) Semantic Fidelity (embedding cosine)
# =============================================================================
def _flatten_scheme_text(entry: Dict[str, Any]) -> str:
    parts: List[str] = []
    meta = entry.get("metadata", {}).get("description", "")
    if meta:
        parts.append(meta.strip())
    for tool in entry.get("tools", []):
        tn = tool.get("name", "")
        td = tool.get("description", "")
        parts.append(f"Tool «{tn}»: {td}")
        for p in tool.get("parameters", []):
            parts.append(f"Param «{p['name']}»: {p.get('description','')}")
    return "\n".join(parts)


def semantic_fidelity_score(entry: Dict[str, Any],
                            code_path: Path,
                            model: str = "text-embedding-3-small",
                            ) -> float:
    text_a = _flatten_scheme_text(entry)
    text_b = code_path.read_text(encoding="utf-8", errors="ignore")
    text_b = extract_tool_defs(text_b)
    emb_a, emb_b = call_embedding([text_a, text_b], model=model)
    dot = sum(a*b for a, b in zip(emb_a, emb_b))
    norm_a = math.sqrt(sum(a*a for a in emb_a))
    norm_b = math.sqrt(sum(b*b for b in emb_b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

# =============================================================================
# 5) Truthfulness / Realism  – single LLM call for all tools (fine-grained scores)
# =============================================================================

# -----------------------------------------------------------------------------
# System prompt for the fine-grained API‐realism audit
# -----------------------------------------------------------------------------
REALISM_SYSTEM_PROMPT = """
You are a **serious API/code reviewer**.
Your task: Refer to the scoring criteria below. For each item provided, assign **an integer score between 0 and 5**.
Return the score in **a JSON array** (without anything else).

────────────────────────────────────────
Scoring criteria – **A single integer score (0–5) is applied only if all conditions in that row are met** 
(Evaluate from top to bottom; select the first matching row)

0 — **Uncallable / Spec Violation**
	•	Code fails to import or execute (syntax/runtime error)
	•	Function signature does not match declared pattern
	•	Call throws exception or returns wrong type

1 — **Stub Implementation (callable but void)**
	•	Contains no meaningful logic or processing beyond basic input validation or trivial formatted returns.
	•	Performs no environment changes (no file creation, no state updates).
	•	Returned data is either hard-coded, fixed strings, or unrelated placeholders, not influenced by actual input.

2 — **Basic Functionality Simulation (incomplete logic)**
	•	Performs initial or superficial environment interactions (e.g., loading files, accessing variables),
but does not utilize or process loaded data meaningfully.
	•	Returned content is superficially formatted and lacks genuine association with actual loaded or processed data.
	•	Environment state updates occur superficially (e.g., assigns values without real processing logic), logic incomplete or placeholder.

3 — **Input-driven but incorrect**
	•	Actually uses the input values or environment data to perform state changes or data processing,
but applies incorrect transformations or produces wrong or inconsistent results.
	•	Core logic is present and nontrivial, but output does not match expected results.
	•	May raise errors for some inputs or produce silently incorrect outputs.

4 — **Correct and Production-Ready**
	•	Implements all declared business steps correctly using the provided inputs or environment data.
	•	File/database state changes fully reflect inputs and intended behavior.
	•	Handles typical and boundary inputs correctly; idempotent and consistent results.

5 — **Robust & Hardened**
	•	Meets all criteria for “4” and adds safety and reliability features:
	•	Comprehensive input validation (formats, ranges, permissions).
	•	Clear, descriptive exceptions for invalid or malicious inputs.
	•	Robust error handling, retries/backoffs, or transaction support for atomic state changes.
	•	Extensive logging/monitoring and comprehensive test coverage.


────────────────────────────────────────
# Read vs. Write Classification (Scheme + Code)

First, use the TOOL SCHEME (name, description, parameters) to infer which environment fields or files **should** be updated. Then inspect the CODE to see if those mutations actually occur.

1. **Infer Expected Mutations**  
   • Look at the tool’s verb and parameters:  
     – “convert”, “save”, “place”, “dig”, “move” ⇒ expect modifications to `self.<field>` or writes to disk (e.g. output files).  
     – “get”, “list”, “look”, “find”, “detect” ⇒ expect **no** mutations.  
   • Map common verbs to fields in `__init__` / `_reset_environment_state`:  
     – convert ⇒ `execution_history`, output file paths  
     – move ⇒ `position`  
     – step ⇒ `step_idx`  
     – …etc.  

2. **Detect Actual Mutations**  
   In the code snippet, search for:  
   • **Assignments** to the inferred `self.<field>` names  
   • **File-write calls** (`fs.save_*`, `open(...,'w')`, `write(...)`, `os.remove`, etc.)  
   • **Method calls** known to produce side-effects (e.g. `load_text`+`save_text`)  

   If you find at least one matching mutation, classify as a **write operation**; otherwise it’s **read-only**.

3. **Apply Scoring**  
   - **Write operations:** apply the 0–5 state-mutation rubric. In the “reason,” explicitly cite which field or API call you detected (or didn’t).  
   - **Read-only operations:**  
     • **3** = correct data returned for typical inputs.  
     • **4** = correct for both typical *and* boundary cases.  
     • **5** = plus extra safeguards (input validation, error handling, logging).  
     In “reason,” prefix with `read-only:` and note any edge-case or safeguard observations.

────────────────────────────────────────
⚠️ Evaluation Guidance (Important):

When assigning scores, **do not rely solely on the text strings returned by the code** 
(e.g., phrases containing "Simulated", "placeholder", "dummy result", or similar).
Instead, carefully inspect the underlying logic by considering these factors:

✅ **Correctness indicators (higher scores)**:
- Actual computation or logical processing of inputs (e.g., parsing, real calculations, meaningful conditionals).
- Explicit environment-state updates (e.g., assignment to known env-state fields, file/database writes).
- Robust error handling, input validation, or idempotency mechanisms.
- **Explicit calls to external LLM (`call_llm`) for processing are dynamic and meaningful operations, deserving a minimum score of 3 or higher. However, simply using external LLM calls without additional robustness does NOT merit the highest score of 5.**

❌ **Stub indicators (lower scores)**:
- Hard-coded or trivial returns (e.g., directly returning the input, formatted string responses without real computation).
- Missing environment-state changes, especially if implied by tool description.
- The presence of keywords like "Simulated", "placeholder", "dummy", strongly suggests lack of genuine implementation.

Always prioritize evaluating the actual logic and data-flow demonstrated in the code, 
rather than merely judging from plausible-sounding returned strings.

────────────────────────────────────────
OUTPUT FORMAT (strict)
Return **exactly** one JSON array.  
Each element **must** contain three keys in this order:

```json
{
  "tool":   "<ToolName>",
  "score":  <0–5 integer>,
  "reason": "<≤30 words explaining the main limitation>"
}
```
No additional fields, comments, markdown, or prose.
If uncertain, err toward the lower score and state why.

────────────────────────────────────────
⚠️ **Important: Tool Name Usage**
When filling in the `"tool"` field in your JSON output:

- **Always use the exact name from the TOOL-SCHEME provided** (the "<<< TOOL: ToolName >>>" header from the scheme), 
- Do **not** use the function name or any other variations extracted directly from code blocks.

Even if the function name in the provided code slightly differs (e.g., contains underscores `_`, hyphens `-`, or spaces), always prefer the TOOL-SCHEME name provided in the header line:

Correct Example:
```json
{
  "tool": "ref-search-documentation",
  "score": 2,
  "reason": "Placeholder environment interaction; loaded content not actually processed."
}

────────────────────────────────────────
ITEMS TO EVALUATE  ↓
(Each block starts with “<<< ITEM-START” and ends with “<<< ITEM-END”.)
…
“””

**Key changes**:

- **Score 1** now explicitly forbids any environment change.  
- **Score 2** allows only *placeholder* state mutations.  
- **Score 3+** requires *real* state changes, with increasing robustness.

"""
def truthfulness_score_and_details(
    code_path: Path,
    server_entry: Dict[str, Any],
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Read the source at `code_path`, extract all @mcp.tool() defs, build a single
    prompt listing each tool’s name, description, signature & code, then ask the LLM
    to assign each a 0.0–1.0 score. Parse the JSON array and return (average_score, details).
    """
    # 1) Read and extract the init/tool definitions
    src = code_path.read_text(encoding="utf-8", errors="ignore")
    init_tools_block = extract_tool_defs(src) or "# <no @mcp.tool() definitions found>"
    environment_state =  extract_method(src,"_reset_environment_state")
    init_code = extract_method(src,"__init__")

    # 2) Assemble the user prompt
    sections: List[str] = [
        f"### MCP _initialize_mcp_server block:\n```python\n{init_tools_block}\n```"
        f"### MCP environment_state block:\n```python\n{environment_state}\n```"
        f"### MCP __init__ block:\n```python\n{init_code}\n```"

    ]
    for tool in server_entry.get("tools", []):
        name       = tool["name"]
        description= tool.get("description", "").strip()
        params     = tool.get("parameters", [])
        sections.append(
            f"<<< TOOL: {name} >>>\n"
            f"Description:\n{description}\n\n"
            f"Signature parameters:\n{params}\n"
        )

    # First, build the body outside the f-string:
    joined_sections = "\n\n".join(sections)

    # Then interpolate it into your prompt:
    user_prompt = f'''
    You are a rigorous code auditor. Below you will find several MCP tool implementations,
    each introduced by a header of the form <<< TOOL: ToolName >>>.

    Important reminder:  
    - When providing the "tool" name in your JSON response, strictly use the names listed in the TOOL-SCHEME section (the headers labeled "<<< TOOL: ToolName >>>").  
    - Do NOT use the exact function names directly from the provided code snippet.

    Do **not** output anything else (no prose, no extra fields).  

    Below are the implementations and their descriptions:

    {joined_sections}
    '''
    # 3) Single LLM call
    resp = call_llm(
        text=user_prompt,
        system_prompt=REALISM_SYSTEM_PROMPT,
        max_tokens=4096,
        temperature=0.0,
    )

    # 4) Parse the JSON array reply
    try:
        judgements: List[Dict[str, Any]] = json.loads(extract_code(resp))
    except Exception:
        # parsing failed → give every tool a zero with raw response
        judgements = [
            {"tool": t["name"], "score": 0.0, "reason": "ParseError: " + resp}
            for t in server_entry.get("tools", [])
        ]
    # 5) Ensure every tool has an entry
    by_tool = {j["tool"]: j for j in judgements}
    details: List[Dict[str, Any]] = []
    total_score = 0.0

    for tool in server_entry.get("tools", []):
        name = tool["name"]
        rec = by_tool.get(name, {
            "tool":  name,
            "score": 0.0,
            "reason":"No judgement from LLM"
        })
        # coerce numeric
        rec["score"] = float(rec.get("score", 0.0)) / 5.0
        total_score += rec["score"]
        details.append(rec)

    # 6) Compute average score
    avg_score = total_score / len(details) if details else 0.0

    return avg_score, details

# =============================================================================
# Master entry: evaluate_single
# =============================================================================
def evaluate_single(server_name: str,
                    code_path: Path,
                    server_entry: Dict[str, Any],
                    ) -> Dict[str, Any]:
    """
    Evaluate a single environment file.
    Args:
        server_name (str): name of the MCP server
        code_path (Path): path to the env file
        server_entry (Dict[str, Any]): entry from the trainset
    Returns:
        Dict[str, Any]: evaluation results
    """
    # 1) executability
    exec_score, exec_msg = check_executability(code_path)

    # 2) schema fidelity
    schema_score, missing = schema_fidelity_score(code_path, server_entry)

    # 3) functionality
    func_score, func_details = functionality_score_and_details(code_path, server_entry)

    # 4) semantic fidelity
    semantic_score = semantic_fidelity_score(server_entry, code_path)

    # 5) truthfulness / realism
    truth_score, truth_details = truthfulness_score_and_details(code_path, server_entry)

    # weighted final
    final = (
        exec_score      * WEIGHTS["executability"]     +
        schema_score    * WEIGHTS["schema_fidelity"]   +
        func_score      * WEIGHTS["functionality"]     +
        semantic_score  * WEIGHTS["semantic_fidelity"] +
        truth_score     * WEIGHTS["truthfulness"]
    )

    return {
        "server": server_name,
        "executability": exec_score,
        "schema_fidelity": schema_score,
        "missing_tools": missing,
        "functionality": func_score,
        "functionality_details": func_details,
        "semantic_fidelity": semantic_score,
        "truthfulness": truth_score,
        "truthfulness_details": truth_details,
        "final": final,
        "exec_msg": exec_msg,
        "code_path": str(code_path)
    }


def functionality_score(code_path: Path, entry: Dict[str, Any]) -> float:
    """
    Wrapper function for functionality_score_and_details that returns only the score.
    
    Args:
        code_path (Path): path to the environment code file
        entry (Dict[str, Any]): server entry from the trainset
        
    Returns:
        float: functionality score (0-5)
    """
    score, _ = functionality_score_and_details(code_path, entry)
    return score


def truthfulness_score(code_path: Path, entry: Dict[str, Any]) -> float:
    """
    Wrapper function for truthfulness_score_and_details that returns only the score.
    
    Args:
        code_path (Path): path to the environment code file
        entry (Dict[str, Any]): server entry from the trainset
        
    Returns:
        float: truthfulness score (0-5)
    """
    score, _ = truthfulness_score_and_details(code_path, entry)
    return score
