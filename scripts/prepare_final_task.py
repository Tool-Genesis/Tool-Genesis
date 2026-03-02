import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def _safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _normalize_item_in_place(item: Dict[str, Any]) -> bool:
    changed = False

    msg = item.get("messages")
    if isinstance(msg, str):
        parsed = _safe_json_loads(msg)
        if isinstance(parsed, (list, dict)):
            item["messages"] = parsed
            changed = True
            msg = item.get("messages")

    meta = item.get("metadata")
    if isinstance(meta, str):
        parsed = _safe_json_loads(meta)
        if isinstance(parsed, dict):
            item["metadata"] = parsed
            changed = True

    messages = item.get("messages")
    if isinstance(messages, list) and len(messages) > 2:
        item["messages"] = messages[:2]
        changed = True

    return changed


def _normalize_data_in_place(data: Any) -> bool:
    changed = False
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                changed |= _normalize_item_in_place(item)
    elif isinstance(data, dict):
        changed |= _normalize_item_in_place(data)
    return changed


def _slugify(text: str) -> str:
    s = text.strip().lower()
    s = s.replace("_", "-")
    s = re.sub(r"[^\w]+", "-", s, flags=re.UNICODE)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def _iter_schema_slugs(schema_root: Path) -> Iterable[str]:
    if not schema_root.exists():
        return []
    return [p.name for p in schema_root.iterdir() if p.is_dir()]


def _find_first_key(obj: Any, key: str) -> Optional[Any]:
    if isinstance(obj, dict):
        if key in obj:
            return obj.get(key)
        for v in obj.values():
            found = _find_first_key(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_first_key(v, key)
            if found is not None:
                return found
    return None


def _ensure_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _extract_overall_score(item: Dict[str, Any]) -> float:
    qqa = item.get("question_quality_assessment")
    if isinstance(qqa, str):
        parsed = _safe_json_loads(qqa)
        if isinstance(parsed, dict):
            qqa = parsed
    if isinstance(qqa, dict):
        score = qqa.get("overall_score")
        if isinstance(score, (int, float)):
            return float(score)
        if isinstance(score, str):
            try:
                return float(score.strip())
            except Exception:
                return 0.0
    return 0.0


def _augment_server_info(
    metadata: Dict[str, Any],
    schema_root: Path,
    current_server_slug: str,
    available_schema_slugs: set,
) -> bool:
    changed = False
    mcp_servers = metadata.get("mcp_servers")
    if not isinstance(mcp_servers, list):
        return changed

    for idx, server in enumerate(mcp_servers):
        if not isinstance(server, dict):
            continue
        server_info = server.get("server_info")
        if not isinstance(server_info, dict):
            continue

        server_name = server.get("server_name")
        server_info_name = server_info.get("name")
        candidates: List[str] = []
        if isinstance(server_name, str) and server_name.strip():
            candidates.append(_slugify(server_name))
        if isinstance(server_info_name, str) and server_info_name.strip():
            candidates.append(_slugify(server_info_name))
        if idx == 0:
            candidates.append(current_server_slug)

        schema_slug = None
        for cand in candidates:
            if cand in available_schema_slugs:
                schema_slug = cand
                break

        if not schema_slug:
            continue

        schema_path = schema_root / schema_slug / "json_schema.json"
        if not schema_path.exists():
            continue

        try:
            schema_obj = _load_json(schema_path)
        except Exception:
            continue

        if "json_schema" in server_info:
            server_info.pop("json_schema", None)
            changed = True

        for k in [
            "python_sdk",
            "configuration_schema",
            "smithery_configuration_requirements",
            "python_sdk_config",
            "python_sdk_url",
        ]:
            v = _find_first_key(schema_obj, k)
            if v is not None and server_info.get(k) != v:
                server_info[k] = v
                changed = True

    return changed


def _process_sample_prepared_json(path: Path, schema_root: Path, current_server_slug: str, available_schema_slugs: set) -> Tuple[bool, bool]:
    data = _load_json(path)
    normalized = _normalize_data_in_place(data)

    augmented = False
    if isinstance(data, list):
        items = [it for it in data if isinstance(it, dict)]
        if len(items) > 20:
            items_sorted = sorted(items, key=_extract_overall_score, reverse=True)
            data = items_sorted[:20]
            normalized = True
        for it in data:
            if not isinstance(it, dict):
                continue
            md = it.get("metadata")
            if isinstance(md, dict):
                augmented |= _augment_server_info(md, schema_root, current_server_slug, available_schema_slugs)
    elif isinstance(data, dict):
        md = data.get("metadata")
        if isinstance(md, dict):
            augmented |= _augment_server_info(md, schema_root, current_server_slug, available_schema_slugs)

    if normalized or augmented:
        _dump_json(path, data)

    return normalized, augmented


def _copy_server_dir(src_dir: Path, dst_dir: Path) -> List[Path]:
    copied_sample_files: List[Path] = []
    for root, dirs, files in os.walk(src_dir):
        rel_root = Path(root).relative_to(src_dir)
        (dst_dir / rel_root).mkdir(parents=True, exist_ok=True)

        for d in dirs:
            (dst_dir / rel_root / d).mkdir(parents=True, exist_ok=True)

        for fn in files:
            src_file = Path(root) / fn
            dst_name = "sample_prepared.json" if fn == "examples.json" else fn
            dst_file = dst_dir / rel_root / dst_name
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            if dst_name == "sample_prepared.json":
                copied_sample_files.append(dst_file)
    return copied_sample_files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    p.add_argument(
        "--kept-json",
        type=str,
        default=str(repo_root / "data" / "Input_selected" / "filtered_servers.json"),
    )
    p.add_argument(
        "--src-root",
        type=str,
        default=str(repo_root / "data" / "filter_task"),
    )
    p.add_argument(
        "--dst-root",
        type=str,
        default=str(repo_root / "data" / "final_task"),
    )
    p.add_argument(
        "--schema-root",
        type=str,
        default=str(repo_root / "data" / "Input_selected_kept"),
    )
    p.add_argument(
        "--all-tasks-out",
        type=str,
        default=None,
        help="Write a consolidated JSON containing up to 20 tasks per server.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    kept_json = Path(args.kept_json)
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    schema_root = Path(args.schema_root)
    all_tasks_out = Path(args.all_tasks_out) if args.all_tasks_out else (dst_root / "all_tasks.json")

    kept_payload = _load_json(kept_json)
    kept = kept_payload.get("kept") if isinstance(kept_payload, dict) else None
    if not isinstance(kept, list):
        raise SystemExit(f"Invalid kept list in: {kept_json}")

    kept_slugs = [s for s in kept if isinstance(s, str) and s.strip()]
    kept_set = set(kept_slugs)
    available_schema_slugs = set(_iter_schema_slugs(schema_root))

    src_existing_slugs = [p.name for p in src_root.iterdir() if p.is_dir()] if src_root.exists() else []
    to_copy = [s for s in src_existing_slugs if s in kept_set]
    missing_in_filter_task = sorted(list(kept_set - set(src_existing_slugs)))

    if args.dry_run:
        print(json.dumps({"to_copy": sorted(to_copy), "missing_in_filter_task": missing_in_filter_task}, ensure_ascii=False, indent=2))
        return

    copied_servers = 0
    copied_samples = 0
    normalized_files = 0
    augmented_files = 0
    servers_without_tasks: List[str] = []
    tasks_by_server: Dict[str, List[Dict[str, Any]]] = {}

    dst_root.mkdir(parents=True, exist_ok=True)
    for slug in sorted(to_copy):
        src_dir = src_root / slug
        dst_dir = dst_root / slug
        sample_files = _copy_server_dir(src_dir, dst_dir)
        copied_servers += 1
        copied_samples += len(sample_files)
        tasks_total = 0
        server_tasks: List[Dict[str, Any]] = []
        for sp in sample_files:
            normalized, augmented = _process_sample_prepared_json(
                sp, schema_root=schema_root, current_server_slug=slug, available_schema_slugs=available_schema_slugs
            )
            if normalized:
                normalized_files += 1
            if augmented:
                augmented_files += 1
            try:
                payload = _load_json(sp)
                if isinstance(payload, list):
                    dict_items = [it for it in payload if isinstance(it, dict)]
                    tasks_total += len(dict_items)
                    server_tasks.extend(dict_items)
                elif isinstance(payload, dict):
                    tasks_total += 1
                    server_tasks.append(payload)
            except Exception:
                pass
        if len(sample_files) == 0 or tasks_total == 0:
            servers_without_tasks.append(slug)
        if server_tasks:
            if len(server_tasks) > 20:
                server_tasks = sorted(server_tasks, key=_extract_overall_score, reverse=True)[:20]
            tasks_by_server[slug] = server_tasks

    all_tasks_payload = [{"server_slug": slug, "tasks": tasks_by_server.get(slug, [])} for slug in sorted(to_copy)]
    _dump_json(all_tasks_out, all_tasks_payload)

    print(
        json.dumps(
            {
                "copied_servers": copied_servers,
                "copied_sample_prepared_files": copied_samples,
                "normalized_sample_prepared_files": normalized_files,
                "augmented_sample_prepared_files": augmented_files,
                "missing_in_filter_task": missing_in_filter_task,
                "servers_without_tasks": sorted(servers_without_tasks),
                "dst_root": str(dst_root),
                "all_tasks_out": str(all_tasks_out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
