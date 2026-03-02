import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

MAX_LINE_LENGTH = 120

def run_mcp_hard_filter(registry_path: Path) -> Dict[str, Any]:
    """
    Runs a series of hard checks on MCP server code files.
    """
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry file not found at: {registry_path}")

    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)

    accepted_for_agent_filter: List[Dict[str, str]] = []
    rejected_files: List[Dict[str, Any]] = []

    for server_name, file_path_str in registry.items():
        reasons = []
        file_path = Path(file_path_str)

        if not file_path.exists():
            reasons.append("File not found")
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    f.seek(0)
                    lines = f.readlines()

                # 1. Syntax check
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    reasons.append(f"Syntax error: {e}")

                # 2. Line length check
                for i, line in enumerate(lines, 1):
                    if len(line) > MAX_LINE_LENGTH:
                        reasons.append(f"Line {i} exceeds {MAX_LINE_LENGTH} characters")

                # 3. TODO/FIXME check
                if re.search(r'TODO|FIXME', content, re.IGNORECASE):
                    reasons.append("Contains TODO or FIXME markers")

                # 4. Debug print check
                if 'print(' in content:
                    reasons.append("Contains debug print statements")

            except Exception as e:
                reasons.append(f"An unexpected error occurred during file processing: {e}")

        if reasons:
            rejected_files.append({
                "server_name": server_name,
                "path": file_path_str,
                "reasons": reasons
            })
        else:
            accepted_for_agent_filter.append({
                "server_name": server_name,
                "path": file_path_str
            })

    return {
        "accepted_for_agent_filter": accepted_for_agent_filter,
        "rejected": rejected_files,
        "summary": {
            "total": len(registry),
            "accepted_count": len(accepted_for_agent_filter),
            "rejected_count": len(rejected_files),
        }
    }