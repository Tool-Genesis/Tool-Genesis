import os
import json
import argparse
import re
import random
from typing import Any, Dict, List, Tuple

from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


def _sanitize_path_component(value: Any) -> str:
    s = str(value or "").strip()
    if not s:
        return "unknown"
    s = s.replace(os.sep, "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w.\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _resolve_out_dir(out_dir: str, out_root: str, strategy: str, model: str) -> Path:
    if out_root:
        safe_model = str(model).replace(".", "-")
        name = f"{_sanitize_path_component(strategy)}_{_sanitize_path_component(safe_model)}"
        return Path(out_root) / name
    return Path(out_dir)


def _read_items(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        items = data.get("items")
        return items if isinstance(items, list) else []
    return []

def build_task_description(item: Dict[str, Any]) -> str:
    tasks = item.get("task_example") or []
    example_tasks = tasks
    if isinstance(tasks, list):
        if len(tasks) > 3:
            idxs = list(range(len(tasks)))
            random.shuffle(idxs)
            example_tasks = [tasks[i] for i in idxs[:3]]
        else:
            example_tasks = tasks
    requirement_document = item["agent_input_prompt"]
    task_description = f"""
    {requirement_document}

    Example tasks:
    {example_tasks}
    """
    return task_description

def load_task(data_path: str, out_root: Path) -> List[Dict[str, Any]]:
    items = _read_items(data_path)
    os.makedirs(out_root, exist_ok=True)
    return items

def filter_task(items: List[Dict[str, Any]], out_root: Path, strategy: str) -> List[Dict[str, Any]]:
    processed_slugs = set()
    registry_path = out_root / "registry.json"
    if registry_path.exists():
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, dict):
                processed_slugs = set(existing.keys())
        except Exception:
            processed_slugs = set()
    return [it for it in items if str(it.get("server_slug", "")) not in processed_slugs]

def append_result(out_root: Path, server_slug: str, json_schema_path: str, env_code_path: str, server_id: Any = None, server_name: str = None, strategy: str = "multi_agent") -> None:
    payload = {
        "server_id": server_id,
        "server_name": server_name,
        "server_slug": server_slug,
        "json_schema_path": json_schema_path,
        "env_code_path": env_code_path,
        "strategy": strategy,
    }
    path = out_root / "registry.json"
    reg: Dict[str, Any] = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, dict):
                reg = existing
        except Exception:
            reg = {}
    reg[str(server_slug)] = payload
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)

def generate_for_item(
    item: Dict[str, Any],
    out_root: Path,
    model: str,
    strategy: str,
    platform: str | None = None,
) -> Tuple[str, str]:
    from src.env_generation import generate_environment_json_and_code

    server_name = item["server_slug"]
    task_description = build_task_description(item)
    server_dir = out_root / server_name
    os.makedirs(server_dir, exist_ok=True)
    
    json_schema, tool_code = generate_environment_json_and_code(
        task_description=task_description,
        model=model,
        output_dir=str(server_dir),
        strategy=strategy,
        platform=platform,
    )
    schema_path = server_dir / "tool_schema.json"
    code_path = server_dir / "env_code.py"
    with open(schema_path, "w", encoding="utf-8") as f:
        f.write(json_schema)
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(tool_code)
    return str(schema_path.resolve()), str(code_path.resolve())

def _run_one(
    item: Dict[str, Any],
    out_root_str: str,
    model: str,
    strategy: str,
    platform: str | None = None,
) -> Tuple[str, str, str, Any, str]:
    out_root = Path(out_root_str)
    slug = item.get("server_slug")
    schema_path, code_path = generate_for_item(
        item,
        out_root,
        model,
        strategy,
        platform=platform,
    )
    sid = item.get("server_id")
    sname = item.get("server_name") or slug
    return slug, schema_path, code_path, sid, sname


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/tool_genesis_v1.json")
    parser.add_argument("--out-dir", type=str, default="temp/run_benchmark")
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="multi_agent")
    parser.add_argument("--platform", type=str, default=None)
    args = parser.parse_args()

    out_root = _resolve_out_dir(args.out_dir, args.out_root, args.strategy, args.model)
    print(f"Output dir: {out_root.resolve()}")
    items = load_task(args.data_path, out_root)
    targets: List[Dict[str, Any]] = filter_task(items, out_root, args.strategy)
    print(f"Need to process: {len(targets)}")
    if args.limit is not None and args.limit > 0:
        targets = targets[: args.limit]
    print(f"Will process: {len(targets)}")

    if args.workers and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [
                ex.submit(
                    _run_one,
                    it,
                    str(out_root),
                    args.model,
                    args.strategy,
                    args.platform,
                )
                for it in targets
            ]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Generate MCP"):
                try:
                    slug, schema_path, code_path, sid, sname = fut.result()
                    append_result(out_root, slug, schema_path, code_path, sid, sname, args.strategy)
                except Exception:
                    pass
    else:
        for it in tqdm(targets, desc="Generate MCP"):
            slug = it.get("server_slug")
            schema_path, code_path = generate_for_item(
                it,
                out_root,
                args.model,
                args.strategy,
                platform=args.platform,
            )
            sid = it.get("server_id")
            sname = it.get("server_name") or slug
            append_result(out_root, slug, schema_path, code_path, sid, sname, args.strategy)

if __name__ == "__main__":
    main()
