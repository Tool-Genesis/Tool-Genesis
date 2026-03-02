import argparse
import ast
from pathlib import Path
from typing import List, Tuple


def _iter_env_code_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("env_code.py") if p.is_file())


def _leading_whitespace(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _find_insertions(lines: List[str]) -> List[Tuple[int, str]]:
    tree = ast.parse("\n".join(lines))
    insertions: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != "_initialize_mcp_server":
            continue
        if node.end_lineno is None:
            continue
        start = node.lineno - 1
        end = node.end_lineno - 1
        if start < 0 or end >= len(lines) or end < start:
            continue
        block = lines[start : end + 1]
        body_lines = block[1:]
        body_indent = None
        for line in body_lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                body_indent = _leading_whitespace(line)
                break
        if body_indent is None:
            body_indent = _leading_whitespace(block[0]) + " " * 4
        last_non_empty_index = None
        for idx in range(len(block) - 1, -1, -1):
            stripped = block[idx].strip()
            if stripped and not stripped.startswith("#"):
                last_non_empty_index = idx
                break
        if last_non_empty_index is None:
            continue
        last_line = block[last_non_empty_index].strip()
        if last_line == "return mcp":
            continue
        insert_at = start + last_non_empty_index + 1
        insertions.append((insert_at, f"{body_indent}return mcp"))
    return insertions


def _apply_insertions(lines: List[str], insertions: List[Tuple[int, str]]) -> List[str]:
    if not insertions:
        return lines
    new_lines = list(lines)
    for index, text in sorted(insertions, key=lambda item: item[0], reverse=True):
        new_lines.insert(index, text)
    return new_lines


def _process_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines()
    insertions = _find_insertions(lines)
    if not insertions:
        return False
    updated_lines = _apply_insertions(lines, insertions)
    updated = "\n".join(updated_lines) + ("\n" if original.endswith("\n") else "")
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    args = parser.parse_args()
    root = Path(args.root)
    files = _iter_env_code_files(root)
    updated_files = []
    for path in files:
        if _process_file(path):
            updated_files.append(path)
    print(f"checked={len(files)} updated={len(updated_files)}")


if __name__ == "__main__":
    main()
