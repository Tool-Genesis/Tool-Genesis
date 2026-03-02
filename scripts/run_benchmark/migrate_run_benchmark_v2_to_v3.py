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
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _sanitize_path_component(value: Any) -> str:
    s = str(value or "").strip()
    if not s:
        return "unknown"
    s = s.replace(os.sep, "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w.\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _resolve_dest_model_dir(dst_root: Path, strategy: str, model: str) -> Path:
    return dst_root / f"{_sanitize_path_component(strategy)}_{_sanitize_path_component(model)}"


def _iter_model_dirs(src_root: Path) -> List[Path]:
    if not src_root.exists():
        return []
    out: List[Path] = []
    for p in sorted(src_root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "registry.json").is_file():
            out.append(p)
    return out


def _infer_strategy_and_model(model_dir: Path, registry: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    name = model_dir.name
    strategy: Optional[str] = None
    model: Optional[str] = None
    if isinstance(registry, dict):
        for _, payload in registry.items():
            if isinstance(payload, dict):
                s = payload.get("strategy")
                if isinstance(s, str) and s.strip():
                    strategy = s.strip()
                    break
    if name.startswith("direct_") and strategy is None:
        strategy = "direct"
    if name.startswith("direct_"):
        model = name[len("direct_") :]
    if model is None:
        model = name
    if strategy is None:
        strategy = "unknown"
    return strategy, model


def _copy_dir(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            return
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    shutil.copytree(src, dst)


def _maybe_file(p: Path) -> Optional[str]:
    if p.is_file():
        return str(p.resolve())
    return None


def _build_registry_for_dest(
    *,
    kept: set[str],
    src_model_dir: Path,
    dst_model_dir: Path,
    src_registry: Optional[Dict[str, Any]],
    overwrite: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    dst_model_dir.mkdir(parents=True, exist_ok=True)
    dst_registry_path = dst_model_dir / "registry.json"

    existing: Dict[str, Any] = {}
    if dst_registry_path.exists():
        try:
            data = _load_json(dst_registry_path)
            if isinstance(data, dict):
                existing = data
        except Exception:
            existing = {}

    src_slugs = [p.name for p in src_model_dir.iterdir() if p.is_dir()]
    to_copy = [s for s in src_slugs if s in kept]

    for slug in to_copy:
        src_server_dir = src_model_dir / slug
        dst_server_dir = dst_model_dir / slug
        if dry_run:
            pass
        else:
            _copy_dir(src_server_dir, dst_server_dir, overwrite=overwrite)

    out: Dict[str, Any] = {}
    for slug in sorted(kept):
        dst_server_dir = dst_model_dir / slug
        if not dst_server_dir.is_dir():
            continue
        schema_path = _maybe_file(dst_server_dir / "tool_schema.json")
        code_path = _maybe_file(dst_server_dir / "env_code.py")
        if not schema_path and not code_path:
            continue

        src_payload: Dict[str, Any] = {}
        if isinstance(src_registry, dict):
            maybe = src_registry.get(slug)
            if isinstance(maybe, dict):
                src_payload = maybe
        if not src_payload and isinstance(existing, dict):
            maybe = existing.get(slug)
            if isinstance(maybe, dict):
                src_payload = maybe

        payload = {
            "server_id": src_payload.get("server_id"),
            "server_name": src_payload.get("server_name") or slug,
            "server_slug": slug,
            "json_schema_path": schema_path,
            "env_code_path": code_path,
            "strategy": src_payload.get("strategy"),
        }
        out[str(slug)] = payload

    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src-root", type=str, default="temp/run_benchmark_v2")
    p.add_argument("--dst-root", type=str, default="temp/run_benchmark_v3")
    p.add_argument("--kept-json", type=str, default="data/Input_selected/filtered_servers.json")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit-model-dirs", type=int, default=None)
    args = p.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    kept_json = Path(args.kept_json)

    kept_payload = _load_json(kept_json)
    kept_list = kept_payload.get("kept") if isinstance(kept_payload, dict) else None
    if not isinstance(kept_list, list):
        raise SystemExit(f"Invalid kept list in: {kept_json}")
    kept = {s.strip() for s in kept_list if isinstance(s, str) and s.strip()}

    model_dirs = _iter_model_dirs(src_root)
    if args.limit_model_dirs is not None and args.limit_model_dirs > 0:
        model_dirs = model_dirs[: args.limit_model_dirs]

    summary: List[Dict[str, Any]] = []
    for md in model_dirs:
        reg_path = md / "registry.json"
        src_registry: Optional[Dict[str, Any]] = None
        try:
            data = _load_json(reg_path)
            if isinstance(data, dict):
                src_registry = data
        except Exception:
            src_registry = None

        strategy, model = _infer_strategy_and_model(md, src_registry)
        dst_model_dir = dst_root / md.name

        src_dirs = {p.name for p in md.iterdir() if p.is_dir()}
        to_keep = sorted([s for s in src_dirs if s in kept])

        dst_registry = _build_registry_for_dest(
            kept=kept,
            src_model_dir=md,
            dst_model_dir=dst_model_dir,
            src_registry=src_registry,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

        if not args.dry_run:
            _dump_json(dst_model_dir / "registry.json", dst_registry)

        summary.append(
            {
                "src": str(md),
                "dst": str(dst_model_dir),
                "strategy": strategy,
                "model": model,
                "kept_servers_found": len(to_keep),
                "src_servers_total": len(src_dirs),
                "dst_registry_items": len(dst_registry),
            }
        )

    print(json.dumps({"models": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
