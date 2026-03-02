import re
from typing import List,Optional
import ast

def _indent(line: str) -> int:
    """Return the count of leading whitespace characters (tabs counted as 1)."""
    return len(re.match(r'^(\s*)', line).group(1))




def extract_method(src: str, method_name: str) -> Optional[str]:
    """
    Extract the full source code of the specified method from the given Python source.

    Args:
        src (str): The complete Python source code.
        method_name (str): The name of the method to extract (e.g., '_reset_environment_state' or 'init').

    Returns:
        Optional[str]: The source code of the method, including signature, docstring, and body,
                       or None if the method is not found.
    """
    # Normalize method name to handle dunder methods
    name = method_name.strip().strip('_')
    candidates = {method_name, f"{name}", f"_{name}", f"__{name}__"}

    # Parse the source into an AST
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)

    # Walk the AST to find the target FunctionDef node
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in candidates:
            start = node.lineno - 1
            end = getattr(node, 'end_lineno', None)
            if end is not None:
                # Slice the original lines to get the exact method source
                return ''.join(lines[start:end])
            # Fallback: return from start until end of file
            return ''.join(lines[start:])

    # Return None if the method was not found
    return None



def extract_tool_defs(src: str) -> str:
    """
    Extract every function decorated with @mcp.tool() *inside*
    `_initialize_mcp_server`, omitting:

      • Triple-quoted docstrings (single-line or multi-line)
      • Pure comment lines (starting with # or ##, ignoring leading spaces)

    Returns
    -------
    str
        All extracted functions concatenated, separated by two blank lines.
        If none are found, returns an empty string.
    """
    lines = src.splitlines(keepends=True)

    # 1) Locate the `_initialize_mcp_server` signature and its base indentation
    sig_pat = re.compile(r'^(\s*)def\s+_initialize_mcp_server\s*\(')
    for i, ln in enumerate(lines):
        m = sig_pat.match(ln)
        if m:
            base_idx = i
            base_indent = _indent(ln)
            break
    else:
        return ""  # Method not found

    # 2) Capture the method body until indentation decreases
    block: List[str] = [lines[base_idx]]
    for ln in lines[base_idx + 1:]:
        if ln.strip() == "":
            block.append(ln)
        elif _indent(ln) > base_indent:
            block.append(ln)
        else:
            break  # End of method body

    # 3) Find indices where @mcp.tool() decorators occur
    deco_idx = [i for i, ln in enumerate(block) if ln.lstrip().startswith("@mcp.tool()")]
    if not deco_idx:
        return ""

    tool_snippets: List[str] = []
    for k, start in enumerate(deco_idx):
        end = deco_idx[k + 1] if k + 1 < len(deco_idx) else len(block)
        snippet: List[str] = []
        j = start

        while j < end:
            ln = block[j]

            # Stop if we hit registration code or `return mcp`
            if re.match(r'^\s*mcp\._env', ln) or re.match(r'^\s*return\s+mcp\b', ln):
                break

            stripped = ln.lstrip()

            # --- Skip pure comment lines ---
            if stripped.startswith("#"):
                j += 1
                continue

            # --- Skip docstrings ---
            if stripped.startswith(('"""', "'''")):
                delim = stripped[:3]
                # Single-line docstring (opening and closing on same line)
                if stripped.count(delim) >= 2:
                    j += 1
                    continue
                # Multi-line docstring: advance until closing delimiter
                j += 1
                while j < end and delim not in block[j]:
                    j += 1
                j += 1  # Skip the closing delimiter line
                continue

            # Keep actual code lines
            snippet.append(ln)
            j += 1

        tool_snippets.append(''.join(snippet).rstrip())

    return "\n\n".join(tool_snippets)