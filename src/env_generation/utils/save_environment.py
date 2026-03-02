# src/env_gen/utils.py
# -*- coding: utf-8 -*-
"""
Utilities for saving generated environments and querying the local registry.

Registry format (saved at <output_dir>/registry.json):
{
  "<env_name>": "<absolute path to <env_name>_env.py>",
  ...
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Union

# File name used to store the registry under the output directory
REGISTRY_FILENAME = "registry.json"


def _atomic_write_text(path: Path, text: str) -> None:
    """
    Write text to a temporary file and atomically replace the target path.
    Safer than writing directly (prevents partial/corrupt JSON on crash).
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def get_registry_path(output_dir: Union[str, Path]) -> Path:
    """
    Return the full path to the registry file under output_dir.
    """
    return Path(output_dir) / REGISTRY_FILENAME


def load_registry(output_dir: Union[str, Path]) -> Dict[str, str]:
    """
    Load the registry from <output_dir>/registry.json.

    Returns:
        A dict mapping env_name -> absolute file path (string).
        Returns {} if the file does not exist or is invalid JSON.
    """
    registry_path = get_registry_path(output_dir)
    if not registry_path.is_file():
        return {}
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
        # Ensure it's a dict[str, str]; tolerate other shapes by coercion if possible.
        if isinstance(data, dict):
            coerced: Dict[str, str] = {}
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, str):
                    coerced[k] = v
            return coerced
        return {}
    except Exception:
        # Be tolerant: treat missing/corrupt registry as empty.
        return {}


def load_processed_keys(output_dir: Union[str, Path]) -> Set[str]:
    """
    Convenience: return the set of env_name keys already present in the registry.
    """
    reg = load_registry(output_dir)
    return set(reg.keys())


def is_target_processed(
    target: str,
    output_dir: Union[str, Path],
    key_mapper: Optional[Callable[[str], str]] = None,
) -> bool:
    """
    Check whether a target (e.g., a server_name) has already been processed,
    using an optional key_mapper to translate target -> env_name (registry key).
    """
    if key_mapper is None:
        key_mapper = lambda s: s  # identity
    processed = load_processed_keys(output_dir)
    return key_mapper(target) in processed


def filter_unprocessed_targets(
    targets: Iterable[str],
    output_dir: Union[str, Path],
    key_mapper: Optional[Callable[[str], str]] = None,
) -> List[str]:
    """
    Given an iterable of current targets (e.g., server names), return only those
    not yet processed according to the registry in output_dir.

    Args:
        targets:    Current list of work items (e.g., server names).
        output_dir: Folder where registry.json is stored.
        key_mapper: Optional function that maps a target (server_name) to the
                    registry key (env_name). If omitted, identity is used.

    Returns:
        A list of targets that are NOT present in the registry.
    """
    registry = load_registry(output_dir)
    processed_keys = set(registry.keys())
    if key_mapper is None:
        key_mapper = lambda s: s  # identity

    pending: List[str] = []
    extra_base = Path("temp/agentic/envs")
    for t in targets:
        key = key_mapper(t)
        if key in processed_keys:
            continue
        env_state_path = extra_base / key / "env_state.json"
        if env_state_path.is_file():
            try:
                content = env_state_path.read_text(encoding="utf-8")
                if content.strip() not in ("", "{}", "[]"):
                    continue
            except Exception:
                pass
        pending.append(t)
    return pending

def canonical(s: str) -> str:
    """Normalize a string for use as an env name / filename fragment."""
    return s.strip().replace(" ", "_").replace("-", "_")

def save_environment(code: str, output_dir: Union[str, Path], env_name: str) -> str:
    """
    Save the generated environment code to a file *and* register its path.

    Args:
        code:       Python code string for the environment.
        output_dir: Directory where the environment file should be saved.
        env_name:   Logical server / environment name (used as the registry key).

    Returns:
        Absolute path (str) to the saved *.py file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Keep the original env_name unchanged; if you want normalization, adapt here.
    clean_name = canonical(env_name)

    # Construct target file path
    py_file = output_path / f"{clean_name}_env.py"

    # Write environment code to disk
    py_file.write_text(code, encoding="utf-8")

    # Update registry
    registry_path = get_registry_path(output_path)
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        if not isinstance(registry, dict):
            registry = {}
    except Exception:
        registry = {}

    # Store absolute file path for consistency
    registry[clean_name] = str(py_file.resolve())

    # Write back atomically to avoid partial writes
    _atomic_write_text(registry_path, json.dumps(registry, ensure_ascii=False, indent=2))

    return str(py_file.resolve())
