import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"registry.json is not an object: {path}")
    return data


def _dump_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _iter_model_dirs(root: Path):
    if not root.exists():
        return
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "registry.json").is_file():
            yield p


def _is_empty_file(path: Path) -> bool:
    if not path.is_file():
        return True
    try:
        return path.stat().st_size == 0
    except OSError:
        return True


def _cleanup_registry_and_servers(model_dir: Path) -> None:
    registry_path = model_dir / "registry.json"
    try:
        registry = _load_json(registry_path)
    except Exception:
        return

    new_registry: Dict[str, Any] = {}
    removed_slugs = []

    for slug, payload in registry.items():
        if not isinstance(payload, dict):
            new_registry[slug] = payload
            continue
        env_path_str = payload.get("env_code_path")
        if not isinstance(env_path_str, str) or not env_path_str.strip():
            removed_slugs.append(slug)
            continue
        env_path = Path(env_path_str)
        if _is_empty_file(env_path):
            removed_slugs.append(slug)
            continue
        new_registry[slug] = payload

    for slug in removed_slugs:
        server_dir = model_dir / slug
        if server_dir.is_dir():
            shutil.rmtree(server_dir)

    if len(new_registry) != len(registry):
        _dump_json(registry_path, new_registry)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    args = parser.parse_args()

    root = Path(args.root)
    for model_dir in _iter_model_dirs(root):
        _cleanup_registry_and_servers(model_dir)


if __name__ == "__main__":
    main()

