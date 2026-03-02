# utils/simulation_planner.py
"""
Generate a simulation_plan for an MCP Tool-Scheme.

Usage
-----
from src.core.utils.simulation_planner import generate_simulation_plan
plan = generate_simulation_plan(tool_scheme_dict)
"""

import json
import re
from typing import Dict, Any
from pathlib import Path

from .utils import materialise_external_files

from src.utils.llm import call_llm, extract_code

SIMULATION_PLAN_SCHEME = """
{
  "files": [                    
    {
      "name":            <string>,   // filename.extension
      "format":          <string>,   // json, csv, xlsx, parquet, sqlite, pdf, docx, jpg, png, mp3, wav, mp4, mov, zip …
      "description":     <string>,   // one-sentence purpose
      "schema":          <object>?,  // concise JSON-Schema (structured files); omit for unstructured
      "example":         <object>?,  // 1–2 rows or records valid per schema
      "metadata_schema": <object>?   // for un/semistructured (page_count, duration, resolution …)
      "extracted_text":  <string>    // The file example content just for unstructured data. structured files can not have extracted_text.
    }
  ],

  // ---- runtime state (self._state) ----
  "state_schema": {
    "type": "object",
    "properties": {
      "fs_root": {
        "type": "string",
        "description": "Root directory given to MCPFileSystem"
      },
      "file_paths": {
        "type": "object",
        "description": "Mapping from file_id to relative path in registry",
        "additionalProperties": {
          "type": "string",
          "description": "Path such as ./docs/abc.pdf or ./media/xyz.mp3"
        }
      },
      // … add any task-specific state fields here (each with description)
    },
    "required": ["fs_root", "file_paths"]
  },

  // ---- per-tool execution plan (DSL) ----
  "tool_strategies": {
    "<tool_name>": [
      // ordered list of DSL steps, each string exactly one instruction
      "FILE_LOAD(file_id)",
      "STATE_WRITE(last_result, _ret)",
      "RETURN _ret"
    ]
    // … one entry per tool
  },

  // ---- initial sampling rules for _state ----
  "init_strategy": {
    "fs_root": {
      "sampling": "constant: temp/data/<server_name>"
    },
    "file_paths": {
      "sampling": "derived: enumerate all plan.files and map file_id → relative path"
    }
    // … additional field-specific rules
  },

   // ---- human-readable pseudocode for _reset_environment_state() ----
  "init_hint": "Create MCPFileSystem(root=fs_root); STATE = {fs_root, file_paths, …}"
}
"""

STATE_SCHEMA_GUIDANCE = """
{
  "type": "object",
  "properties": {
    "fs_root": {
      "type": "string",
      "description": "Root directory passed to MCPFileSystem (e.g. temp/data/document_edit_mcp)"
    },
    "file_paths": {
      "type": "object",
      "description": "Mapping of file IDs to relative paths recorded in registry.json",
      "additionalProperties": {
        "type": "string",
        "description": "Relative path such as ./docs/report.pdf or ./media/promo.mp4"
      }
    },

    // ---- example task-specific fields ------------------------------------
    "position": {
      "type": "object",
      "description": "Bot coordinates (demo field)",
      "properties": {
        "x": { "type": "number", "description": "X axis" },
        "y": { "type": "number", "description": "Y axis" },
        "z": { "type": "number", "description": "Z axis" }
      },
      "required": ["x", "y", "z"]
    },
    "inventory": {
      "type": "array",
      "description": "Items currently held (demo field)",
      "items": {
        "type": "object",
        "properties": {
          "name":     { "type": "string",  "description": "Item name" },
          "quantity": { "type": "number",  "description": "Count" }
        },
        "required": ["name", "quantity"]
      }
    }
  },

  // At minimum an environment must track fs_root and file_paths
  "required": ["fs_root", "file_paths"]
}
"""

TOOL_STRATEGIES_GUIDANCE="""
tool_strategies:
<tool_name>:
 - <DSL instruction 1>
 - <DSL instruction 2>
 …
### Example

```yaml
tools:
  select-project:
    - STATE_WRITE(current_project, project_id)
    - RETURN "status":"success","current_project":project_id

  list-projects:
    - FILE_LOAD("gcp_projects.json")
    - RETURN _ret

  set-log-count:
    - STATE_WRITE(log_count, count)
    - RETURN true
```
"""

INIT_STRATEGY_GUIDANCE="""
{
  "current_query_id": {
    "source": "file_keys",
    "file": "temp/data/DuneLink/dune_queries.json",
    "key_path": "queries.keys",
    "sampling": "random choice from existing keys"
  },
  "last_result": {
    "source": "file_values",
    "file": "temp/data/DuneLink/dune_queries.json",
    "key_path": "queries[{current_query_id}].result",
    "sampling": "derived from current_query_id"
  },
  "inventory": {
    "pool_example": [
      "diamond", "iron_ingot", "gold_ingot", "wooden_plank", "stone",
      "coal", "torch", "bread", "apple", "bow", "arrow",
      "iron_sword", "wooden_pickaxe", "stone_pickaxe",
      "iron_pickaxe", "diamond_pickaxe", "shield", "helmet",
      "chestplate", "leggings"
    ],
    "sampling": "k ∼ UniformInt(5,10) distinct sample from pool_example"
  },
  "position": {
    "area": "testing_range",
    "bounds": { "x": [-50, 50], "y": [0, 0], "z": [-50, 50] },
    "sampling": "x,z ∼ Uniform(-50,50); y fixed 0"
  }
}

"""

EXTERNAL_FILES_GUIDANCE="""
{
  "files": [
    {
      "name": "bank_transactions.parquet",
      "format": "parquet",
      "description": "Historical transaction records (structured)",
      "schema": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "tx_id":    { "type": "string",  "description": "Transaction ID" },
            "amount":   { "type": "number",  "description": "Signed value in USD" },
            "datetime": { "type": "string",  "format": "date-time" },
            "merchant": { "type": "string" }
          },
          "required": ["tx_id", "amount", "datetime"]
        }
      },
      "example": [
        { "tx_id": "T123", "amount": -25.7, "datetime": "2025-05-01T12:00:00Z", "merchant": "CoffeeShop" }
      ]
    },
    {
      "name": "marketing_brief.pdf",
      "format": "pdf",
      "description": "Campaign design document (unstructured)",
      "metadata_schema": {
        "type": "object",
        "properties": {
          "page_count": { "type": "integer", "description": "Total pages" },
          "title":      { "type": "string",  "description": "Document title" },
          "author":     { "type": "string",  "description": "Primary author" }
        },
        "required": ["page_count"]
      }
    },
    {
      "name": "podcast_intro.mp3",
      "format": "mp3",
      "description": "Intro music clip",
      "metadata_schema": {
        "type": "object",
        "properties": {
          "duration_sec": { "type": "number",  "description": "Length in seconds" },
          "sample_rate":  { "type": "integer", "description": "Hz" },
          "channels":     { "type": "integer", "description": "Mono=1 Stereo=2 ..." }
        },
        "required": ["duration_sec", "sample_rate"]
      }
    }
  ]
}
"""

## SYSTEMPT_PROMPT 
SYSTEM_PROMPT = f"""You are a senior simulation architect for reinforcement‐learning environment generation.
Your task is to analyze the given MCP Tool‐Scheme and produce a fully offline simulation_plan with NO external API calls.

**OUTPUT REQUIREMENTS**  
• Return **only** a single JSON object exactly matching the schema below.  
• Do **not** include any keys beyond those in the schema.

**SIMULATION_PLAN SCHEMA**  
{SIMULATION_PLAN_SCHEME}

**STATE_SCHEMA GUIDANCE**
	•	Under state_schema.properties, every property must include a "description".
	•	All nested properties must also carry a "description".
	•	Avoid huge enumerations; keep the schema concise and readable.
	•	If the top-level files array is not empty, the schema must include:
  • fs_root – the directory that will be passed to MCPFileSystem.
  • file_paths – an object mapping each file_id to the relative path recorded in registry.json (e.g. "report_pdf": "./docs/report.pdf").
Example:

{STATE_SCHEMA_GUIDANCE}

**TOOL_STRATEGIES GUIDANCE**  
Map each tool name to an ordered list of DSL instructions.  
**Every** tool **must** end with `RETURN _ret`.  

- If the tool naturally produces data (e.g., a lookup or query), set `_ret` to that data.  
- If the tool only performs side effects (e.g., setting a parameter, toggling a flag), set `_ret` to a status 
indicator:
  - A boolean success flag: `true` / `false`  
  - A confirmation message string: `"OK"` / `"Parameter set"`  
  - Or a small object summarizing the change: `"status":"success","param":…`

### DSL PRIMITIVES  
```dsl
STATE_READ(key)           # Read a small, ephemeral in‐memory value named key  
STATE_WRITE(key,val)      # Write val (primitive, object, or list) to in‐memory state key  
FILE_LOAD(file_key)       # Load data from disk (any supported file format) at path = state.file_paths[file_key] into _ret
FILE_MODIFY(file_key,obj) # Load data from path = state.file_paths[file_key], merge or patch with obj, then save updated data back to disk
FILE_SAVE(file_key,obj)   # Overwrite or create data file at path = state.file_paths[file_key] with obj
LLM_CALL(prompt)          # Invoke LLM only for truly generative or unpredictable tasks  
IF <cond> THEN            # Conditional branch; <cond> can use ==, !=, >, in, etc.  
  - …nested instructions…  
ELSE                      # Alternate branch  
  - …nested instructions…  
SLICE last N              # Keep only the last N items of the current list in _ret  
RETURN _ret               # Yield the final result stored in _ret

Example:

{TOOL_STRATEGIES_GUIDANCE}


**INIT_STRATEGY GUIDANCE**  
When you describe how an environment should random-initialize its state, follow these rules for every field:

Always specify the sampling rule:
  – State whether the value comes from a range, a choice list, or an external data set.
  – If a field depends on another field, note that dependency explicitly.
  Lists / collections
  • Include a pool_example array with ≥ 20 distinct string entries (e.g., item names, entity types).
  • Provide a sampling rule such as
  k ~ UniformInt(min,max); sample k distinct elements from pool_example.
  Numeric fields
  • Give either Uniform(min,max) (floats) or UniformInt(min,max) (ints).
  • If the value must be positive or non-zero, say so.
  String / enum fields
  • List every possible value in choices.
  • Optional: add a weights array for non-uniform selection.
  Fields backed by external data
  • Add a source key: "file_keys" or "file_values".
  • Provide file (path) and key_path (JSON path or format string).
  • Write sampling: "random choice from existing keys" (or similar) so that only valid IDs are generated.
  Complex objects
  • Show one or two full example objects in example_structure.
  • Then give a sampling paragraph explaining how each sub-field is filled.
  Cross-field consistency
  • If one field must be compatible with another (e.g., last_result must correspond to current_query_id), describe the derivation:
  "derived_from": "current_query_id".

Example:

{INIT_STRATEGY_GUIDANCE}

**EXTERNAL_FILES GUIDANCE**  
• List Content – ​​Add an entry for each external resource in the top-level files array.
The code generation helper will later merge these entries into a registry.json file and create minimal placeholder bytes in the docs/ (text/structured) or media/ (binary) directories.
• Structured formats (json, csv, xlsx, parquet, sqlite, duckdb, etc.)
• Provide a concise schema (JSON-Schema draft-07) with 1- or 2-line examples.
• Semi-structured/unstructured formats (pdf, docx, pptx, jpg, png, mp3, wav, mp4, mov, zip, etc.)
• Skip the schema; instead provide a metadata_schema that captures only the necessary meta information (e.g., number of pages, duration, resolution, codecs).
• Schema Scope – Describe only the top-level structure; avoid using large enumerations or deeply nested details.
• Registry mapping (required in state_schema)
• fs_root – The absolute or relative directory passed to MCPFileSystem.
• file_paths – An object that maps each file_id (stem of the name) to a relative path written to registry.json
(e.g. “marketing_brief”: “./docs/marketing_brief.pdf”).
• Test support – The files array must include all files needed by generated tests, so that create_external_files() can automatically create the placeholders (with extracted_text) and let the entire suite run without manual setup.

{EXTERNAL_FILES_GUIDANCE}

**DECISION RULES**  

1. STATE_  
   • Think “RAM object”  
     – A single, self-contained JSON-serialisable dict fully describes the
       *current* episode.  
     – Keys and value types are fixed and small; counts rarely exceed a few
       dozen.  
     – Examples: board-game position, robot pose + inventory, short chat log.

2. FILE_
  • Think “local data snapshot” – supporting both structured and unstructured files.
  – Structured data (JSON, CSV, XLSX, database dumps…):
    • Store your corpus in one or more files whose structure you describe with a concise JSON-Schema draft-07.
    • At runtime, load only the slices you need (e.g. via JSON‐pointer, SQL queries, pandas filters) and keep lightweight indices, cursors or caches in _state.
  – Unstructured data (PDF, MP3/MP4, images, binaries…):
    • Provide a metadata_schema for each file (e.g. page count, duration, resolution) so the simulator knows its “shape.”
    • Load only the metadata or compute hashes (SHA-256) in _state; avoid deserializing entire blobs unless strictly necessary.
  – In both cases, heavy static data lives on disk; your in-memory _state holds only the minimal working set, indices, and file‐level metadata (paths, hashes, cursors) needed for fast, repeatable simulation.

3. LLM_CALL 
   • Think “ad-hoc web browsing / open world”  
      – The environment must reference information that *cannot* be captured
        by a fixed schema or snapshot: unbounded entities, loosely typed
        attributes, or generative behaviour.  
      – Typical for web-search style tasks, expansive knowledge bases, or
        situations where the format itself is unknown or constantly changing.
      – Can simulate running any form of code, such as SQL, python, etc.
      – Can perform logical reasoning such as prediction and reasoning based on the information provided or extract summaries or translations, etc.

4. always offline  
   • Never send network requests; emulate everything locally in code.

Return **only** the JSON simulation_plan And Only Using English.
"""

def _build_user_prompt(tool_scheme: Dict[str, Any]) -> str:
    """Serialize tool-scheme and embed into the user prompt."""
    pretty_json = json.dumps(tool_scheme, indent=2, ensure_ascii=False)
    user_prompt = f"""
      Here is the Tool-Scheme (in JSON). Decide the simulation strategy and
      output a `simulation_plan` per instructions above.\n\n
      {pretty_json}
      """

    return user_prompt

# ---------- public helper ----------------------------------------------------
def generate_simulation_env_plan(tool_scheme: Dict[str, Any],
                                base_dir: str | Path,
                                model: str = "gpt-4o",
                                temperature: float = 0.2,
                                max_tokens: int = 3000) -> Dict[str, Any]:
    """
    Call an LLM to decide the best simulation strategy for the given Tool-Scheme.

    Parameters
    ----------
    tool_scheme : dict
        The full MCP server JSON spec (metadata + tools).
    base_dir : str | Path
        Base directory for external file creation.
    model : str
        LLM model name passed to utils.llm.call_llm.
    temperature : float
        Sampling temperature.
    max_tokens : int
        Max tokens for the response.

    Returns
    -------
    dict
        A simulation_plan dictionary ready to inject into mcp_data["simulation_plan"].
    """
    user_prompt = _build_user_prompt(tool_scheme)

    raw = call_llm(
        text=user_prompt,
        system_prompt=SYSTEM_PROMPT,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # LLM should return plain JSON; parse & return
    # ---- parse JSON ----
    try:
        plan = json.loads(extract_code(raw))
    except json.JSONDecodeError as err:
        raise ValueError(f"LLM did not return valid JSON:\n{raw}") from err

    # ---- materialise files & patch plan ----
    plan = materialise_external_files(plan, base_dir)

    # ---- persist patched plan ----
    plan_path = Path(base_dir) / "env_plan.schema.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    return plan
