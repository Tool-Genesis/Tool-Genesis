# create_external_files.py

from __future__ import annotations
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
# ---------------------------------------------------------------------------- #
#                              configuration                                   #
# ---------------------------------------------------------------------------- #
STRUCTURED = {"json", "csv", "xlsx", "parquet", "sqlite", "duckdb"}

# ---------------------------------------------------------------------------- #
#                              helper functions                                 #
# ---------------------------------------------------------------------------- #
def _sha256(data: Optional[bytes] = None) -> str:
    """
    Compute SHA-256 hex digest of given bytes.
    Treat None as empty bytes.
    """
    if data is None:
        data = b""
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def _now_iso() -> str:
    """Return current UTC time as ISO-8601 string with 'Z' suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def _binary_placeholder(path: Path, size: int = 1) -> None:
    """Write a placeholder binary file of given size (default 1 byte)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)

# ---------------------------------------------------------------------------- #
#                 single-file materialisation helper                            #
# ---------------------------------------------------------------------------- #
def _create_one(
    spec: Dict[str, Any],
    docs_dir: Path,
    media_dir: Path,
    template_dir: Path,
    registry: Dict[str, Any],
    written: List[Path],
) -> None:
    """
    Materialise one file spec:
      - Always write a 1-byte placeholder under docs/ or media/
      - Inline JSON text into registry.extracted_text for any STRUCTURED format
      - For non-structured, use spec['extracted_text'] if provided
      - Write schema sidecar under templates/
      - Append all written paths to written[]
      - Add registry entry
    """
    file_name = spec["name"]
    fmt = spec["format"].lower()

    # choose docs vs media
    if fmt in STRUCTURED:
        rel = Path("docs") / file_name
    else:
        rel = Path("media") / file_name
    abs_path = docs_dir.parent / rel
    abs_path.parent.mkdir(parents=True, exist_ok=True)

    # write placeholder
    _binary_placeholder(abs_path)
    written.append(abs_path)

    # prepare extracted_text
    extracted_text = ""
    if fmt in STRUCTURED:
        # always inline JSON text
        example = spec.get("example") or {}
        extracted_text = json.dumps(example, ensure_ascii=False, indent=2)
    else:
        # unstructured: use provided text if any
        extracted_text = spec.get("extracted_text", "")

    # write schema sidecar if present
    schema = spec.get("schema") or spec.get("metadata_schema")
    if schema:
        template_dir.mkdir(parents=True, exist_ok=True)
        schema_path = template_dir / f"{file_name}.schema.json"
        schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
        schema_path.write_text(schema_text, encoding="utf-8")
        written.append(schema_path)
    real_path = f"./{rel.as_posix()}"
    # insert into registry
    registry[real_path] = {
        "file_id": file_name.rsplit(".", 1)[0],
        "file_path": real_path,
        "format": fmt,
        "content_hash": _sha256(extracted_text.encode("utf-8") if extracted_text else None),
        "extracted_text": extracted_text,
        "metadata": {},
        "last_modified": _now_iso(),
    }
    # store real path for DSL patching
    spec["real_path"] = f"./{rel.as_posix()}"

# ---------------------------------------------------------------------------- #
#                           DSL patching helper                                 #
# ---------------------------------------------------------------------------- #
def _patch_tool_strategies(
    strategies: Dict[str, List[str]],
    replacement: Dict[str, str]
) -> None:
    """
    Replace FILE_LOAD('orig_name') tokens in each step with the real relative path.
    """
    for steps in strategies.values():
        for idx, line in enumerate(steps):
            token = line.strip()
            if token.startswith("FILE_LOAD("):
                name = token.split("FILE_LOAD(",1)[1].split(")",1)[0].strip("'\"")
                if name in replacement:
                    steps[idx] = f"FILE_LOAD('{replacement[name]}')"

# ---------------------------------------------------------------------------- #
#                          public entry point                                  #
# ---------------------------------------------------------------------------- #
def materialise_external_files(plan: Dict[str, Any], base_dir: str | Path) -> Dict[str, Any]:
    """
    • Materialise placeholders + registry.json under base_dir
    • Inline JSON for all STRUCTURED formats
    • Write schema sidecars under templates/
    • Patch plan in-place:
       - plan['file_paths']: file_id → relative path
       - plan['files_path']: list of absolute paths written
       - patch DSL FILE_LOAD calls
       - ensure state_schema contains file_paths
    """
    base = Path(base_dir)
    docs = base / "docs"
    media = base / "media"
    templates = base / "templates"
    docs.mkdir(parents=True, exist_ok=True)
    media.mkdir(parents=True, exist_ok=True)

    registry: Dict[str, Any] = {}
    written: List[Path] = []

    # materialise each file spec
    for spec in plan.get("files", []):
        _create_one(spec, docs, media, templates, registry, written)

    # write registry.json
    reg_path = base / "registry.json"
    reg_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    written.append(reg_path)

    # patch plan
    plan["file_paths"] = {fid: entry["file_path"] for fid, entry in registry.items()}
    plan["files_path"] = [str(reg_path)]

    # ensure state_schema has file_paths
    state_schema = plan.setdefault("state_schema", {"type":"object","properties":{},"required":[]})
    state_schema["properties"].setdefault(
        "file_paths",
        {
            "type": "object",
            "description": "Map file_id to its on-disk relative path",
            "additionalProperties": {"type":"string"}
        }
    )
    if "file_paths" not in state_schema["required"]:
        state_schema["required"].append("file_paths")

    # patch DSL
    if "tool_strategies" in plan:
        name2path = {spec["name"]: spec["real_path"] for spec in plan["files"]}
        _patch_tool_strategies(plan["tool_strategies"], name2path)

    return plan



STRUCTURED_EXTS = {"json", "csv", "xlsx", "parquet", "sqlite", "duckdb"}

def materialise_task_data_files(
    task_data: Dict[str, Any], 
    base_dir: str | Path = Path("temp/task_data"),
) -> Dict[str, Any]:
    """
    Materialise all files described in a flattened task_data dict.
    - Writes each file under base_dir/docs or base_dir/media.
    - Writes initial_state.json in base_dir.
    - Builds a registry (keyed by real_path) with file metadata.
    - Returns a dict with useful downstream info.
    
    Args:
        task_data: Dict with keys: 'task_id', 'initial_state', 'files' (list of file specs)
        base_dir: Root directory to save task files.

    Returns:
        plan: Dict with:
            - 'task_id', 'files', 'file_paths', 'files_path', 'registry' (see below)
    """
    base = Path(base_dir)
    docs = base / "docs"
    media = base / "media"
    docs.mkdir(parents=True, exist_ok=True)
    media.mkdir(parents=True, exist_ok=True)

    registry: Dict[str, Any] = {}
    written: List[Path] = []
    for spec in task_data.get("files", []):
        fmt = spec.get("format", "txt").lower()
        file_name = spec.get("name")
        if file_name is None:
            continue
        # Determine file path (docs or media)
        if fmt in STRUCTURED_EXTS:
            file_path = docs / file_name
        elif fmt in {"mp3", "mp4", "wav", "jpg", "png", "jpeg"}:
            file_path = media / file_name
        else:
            file_path = docs / file_name

        # Write file content
        if fmt in STRUCTURED_EXTS:
            file_path.write_text(json.dumps(spec.get("data", {}), ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            data = spec.get("data", "")
            if isinstance(data, (dict, list)):
                file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                file_path.write_text(str(data), encoding="utf-8")

        real_path = str(file_path.relative_to(base))
        extracted_text = str(spec.get("data", ""))
        file_id = file_name.rsplit(".", 1)[0] if "." in file_name else file_name
        content_hash = _sha256(extracted_text.encode("utf-8") if extracted_text else None)
        last_modified = _now_iso()
        metadata = spec.get("metadata", {})

        # Fill registry per your required format
        registry[real_path] = {
            "file_id": file_id,
            "file_path": real_path,
            "format": fmt,
            "content_hash": content_hash,
            "extracted_text": extracted_text,
            "metadata": metadata,
            "last_modified": last_modified,
        }
        written.append(file_path)

    # Write registry.json
    reg_path = base / "registry.json"
    reg_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    written.append(reg_path)

    # Write initial_state.json
    init_state_path = base / "init_state.json"
    init_state = task_data.get("initial_state", {})
    init_state_path.write_text(json.dumps(init_state, ensure_ascii=False, indent=2), encoding="utf-8")
    written.append(init_state_path)

    # Assemble output plan/dict
    plan = {
        "task_id": task_data.get("task_id"),
        "files": task_data.get("files", []),
        "file_paths": {k: v["file_path"] for k, v in registry.items()},
        "files_path": [str(reg_path)],
        "registry": registry,
        "init_state_path": str(init_state_path),
    }
    return plan
