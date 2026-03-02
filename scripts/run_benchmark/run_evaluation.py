import os
import json
import argparse
import re
from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path
import sys
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from concurrent.futures import ProcessPoolExecutor, as_completed

GT_DATA_PATH = "data/tool_genesis_v3.json"
_EVAL_COMPONENTS = None


def _get_eval_components():
    global _EVAL_COMPONENTS
    if _EVAL_COMPONENTS is None:
        from src.env_evalution import (
            GTResolver,
            l1_protocol_compliance_metrics,
            l2_semantic_correctness_metrics,
            l3_production_robustness_metrics,
        )

        _EVAL_COMPONENTS = (
            GTResolver,
            l1_protocol_compliance_metrics,
            l2_semantic_correctness_metrics,
            l3_production_robustness_metrics,
        )
    return _EVAL_COMPONENTS

def _sanitize_path_component(value: Any) -> str:
    s = str(value or "").strip()
    if not s:
        return "unknown"
    s = s.replace(os.sep, "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w.\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"

def _append_json(path: Path, payload: Dict[str, Any]) -> None:
    arr: List[Dict[str, Any]] = []
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                arr = data
        except Exception:
            arr = []
    arr.append(payload)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)

def _write_debug(out_root: Path, result: Dict[str, Any], debug: Dict[str, Any]) -> None:
    slug = result.get("server_slug") or str(result.get("server_id") or "unknown")
    droot = out_root / "debug" / str(slug)
    os.makedirs(droot, exist_ok=True)
    l1_debug = debug.get("l1") or {}
    with open(droot / "l1_debug.json", "w", encoding="utf-8") as f:
        json.dump(l1_debug, f, ensure_ascii=False, indent=2)
    l2_dbg = debug.get("l2")
    if l2_dbg is not None:
        with open(droot / "l2_debug.json", "w", encoding="utf-8") as f:
            json.dump(l2_dbg, f, ensure_ascii=False, indent=2)
    l3_dbg = debug.get("l3")
    if l3_dbg is not None:
        with open(droot / "l3_debug.json", "w", encoding="utf-8") as f:
            json.dump(l3_dbg, f, ensure_ascii=False, indent=2)

def _load_registry_items(pred_path: str) -> List[Dict[str, Any]]:
    p = Path(pred_path)
    reg_path = p / "registry.json" if p.is_dir() else p
    if not reg_path.exists():
        return []
    try:
        with open(reg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            items: List[Dict[str, Any]] = []
            for slug, payload in data.items():
                if not isinstance(payload, dict):
                    continue
                schema_path = payload.get("json_schema_path")
                code_path = payload.get("env_code_path")
                server_id = payload.get("server_id")
                server_name = payload.get("server_name")
                if isinstance(schema_path, str) or isinstance(code_path, str):
                    items.append({
                        "server_id": server_id,
                        "server_slug": str(slug),
                        "server_name": server_name,
                        "schema_path": schema_path if isinstance(schema_path, str) else None,
                        "env_code_path": code_path if isinstance(code_path, str) else None,
                    })
            return items
    except Exception:
        return []
    return []

def _filter_items(items: List[Dict[str, Any]], out_root: Path) -> List[Dict[str, Any]]:
    processed: set = set()
    results_path = out_root / "results.json"
    if results_path.exists():
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, list):
                for r in existing:
                    v = r.get("server_id") or r.get("server_slug")
                    if v is not None:
                        processed.add(str(v))
        except Exception:
            processed = set()
    def _key(it: Dict[str, Any]) -> str:
        v = it.get("server_id") or it.get("server_slug")
        return str(v) if v is not None else ""
    return [it for it in items if _key(it) not in processed]

def evaluate_one(
    item: Dict[str, Any],
    gt_path: str,
    attempts: int = 3,
    skip_l1: bool = False,
    skip_l2: bool = False,
    skip_l3: bool = False,
    registry_path: Optional[str] = None,
    out_root: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    (
        GTResolver,
        l1_protocol_compliance_metrics,
        l2_semantic_correctness_metrics,
        l3_production_robustness_metrics,
    ) = _get_eval_components()

    server_name = item.get("server_slug")

    gtresolver = GTResolver(gt_path)
    schema, code = gtresolver.resolve_item(item)
    # Use server_name for GT lookup, falling back to server_slug for consistency
    gt_lookup_key = item.get("server_name") or item.get("server_slug")
    gt_schema_obj = gtresolver.get_gt_schema_obj(gt_lookup_key)

    # Guard: if schema or gt_schema_obj resolved to None, L2/L3 cannot run
    if schema is None:
        schema = {}
    if gt_schema_obj is None:
        gt_schema_obj = {}

    if skip_l1:
        m1 = {}
        l1_success = True
    else:
        l1_success, m1, = l1_protocol_compliance_metrics(schema, attempts, server_name, registry_path)
    if skip_l2:
        m2 = {}
        tool_map = {}
        l2_debug = {}
    else:
        trajectory_root: Optional[Path] = None
        if out_root is not None:
            trajectory_root = Path(out_root) / "trajectory" / str(server_name)
        tool_map, m2, l2_debug = l2_semantic_correctness_metrics(
            l1_success,
            schema,
            server_name,
            registry_path,
            gt_schema_obj,
            gt_path,
            item.get("schema_path"),
            trajectory_root,
        )
    if skip_l3:
        m3 = {}
        l3_debug = {}
    else:
        m3, l3_debug = l3_production_robustness_metrics(l1_success, schema, code, tool_map, gt_schema_obj)
    metrics: Dict[str, Any] = {}
    metrics.update(m1)
    metrics.update(m2)
    metrics.update(m3)
    result = {
        "server_id": item.get("server_id"),
        "server_slug": item.get("server_slug"),
        "metrics": metrics,
    }
    
    debug = {
        "l1": m1,
        "l2": l2_debug,
        "l3": l3_debug,
    }
    return result, debug

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-path", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--project-name", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--strategy", type=str, default="multi_agent")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--skip-l1", action="store_true")
    parser.add_argument("--skip-l2", action="store_true")
    parser.add_argument("--skip-l3", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    project_name = args.project_name
    if not project_name:
        safe_model = str(args.model).replace(".", "-")
        project_name = f"{_sanitize_path_component(args.strategy)}_{_sanitize_path_component(safe_model)}"

    pred_root = Path(args.pred_path)
    if not (pred_root / "registry.json").exists():
        pred_root = pred_root / project_name

    out_root = Path(args.out_root) / project_name
    os.makedirs(out_root, exist_ok=True)

    results_path = out_root / "results.json"
    registry_path = str(pred_root / "registry.json")
    if args.reset and results_path.exists():
        try:
            os.remove(results_path)
            print("[Eval] Reset enabled: existing results.json removed")
        except Exception:
            print("[Eval] Reset enabled: failed to remove existing results.json, continuing")

    reg_items = _load_registry_items(str(pred_root))
    filtered_items = reg_items if args.reset else _filter_items(reg_items, out_root)
    
    total_loaded = len(reg_items)
    to_eval = len(filtered_items)
    skipped = total_loaded - to_eval
    print(f"[Eval] Loaded: {total_loaded} | Skipped: {skipped} | Pending: {to_eval} | Attempts: {args.attempts}")
    items = filtered_items
    if args.limit is not None and args.limit > 0:
        items = items[: args.limit]
        print(f"[Eval] Limit: {args.limit} | Will evaluate: {len(items)}")
    else:
        print(f"[Eval] Will evaluate: {len(items)}")

    if args.workers and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [
                ex.submit(
                    evaluate_one,
                    it,
                    GT_DATA_PATH,
                    args.attempts,
                    args.skip_l1,
                    args.skip_l2,
                    args.skip_l3,
                    registry_path,
                    str(out_root),
                )
                for it in items
            ]
            pbar = tqdm(total=len(items), desc="[Eval] Running")
            for fut in as_completed(futures):
                try:
                    res, dbg = fut.result()
                    _write_debug(out_root, res, dbg)
                    _append_json(out_root / "results.json", res)
                except Exception as exc:
                    print(f"[Eval] WARNING: evaluate_one failed: {exc}")
                finally:
                    pbar.update(1)
            pbar.close()
    else:
        pbar = tqdm(items, desc="[Eval] Running")
        for it in pbar:
            name = it.get("server_name") or it.get("server_slug") or str(it.get("server_id") or "")
            pbar.set_description(f"[Eval] {name}")
            res, dbg = evaluate_one(
                it,
                GT_DATA_PATH,
                attempts=args.attempts,
                skip_l1=args.skip_l1,
                skip_l2=args.skip_l2,
                skip_l3=args.skip_l3,
                registry_path=registry_path,
                out_root=str(out_root),
            )
            _write_debug(out_root, res, dbg)
            _append_json(out_root / "results.json", res)

if __name__ == "__main__":
    main()
