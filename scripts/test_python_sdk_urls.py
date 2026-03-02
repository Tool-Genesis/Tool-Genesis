import argparse
import asyncio
import base64
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def _iter_server_slugs(schema_root: Path) -> Iterable[str]:
    if not schema_root.exists():
        return []
    return [p.name for p in schema_root.iterdir() if p.is_dir()]


def _mask_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        qs = parse_qsl(parts.query, keep_blank_values=True)
        masked = []
        for k, v in qs:
            if k.lower() in {"api_key", "apikey", "key", "token", "access_token", "profile"}:
                masked.append((k, "***" if v else ""))
            else:
                masked.append((k, v))
        new_query = urlencode(masked)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))
    except Exception:
        return url


def _fill_placeholders(url: str, smithery_api_key: str, smithery_profile: str, config_obj: Any) -> str:
    out = url
    if "{smithery_api_key}" in out:
        out = out.replace("{smithery_api_key}", smithery_api_key or "")
    if "{smithery_profile}" in out:
        out = out.replace("{smithery_profile}", smithery_profile or "")
    if "{config_b64}" in out:
        cfg = config_obj
        if cfg is None or cfg == "":
            cfg = {"debug": False}
        if isinstance(cfg, str):
            try:
                cfg = json.loads(cfg)
            except Exception:
                cfg = {"debug": False}
        if not isinstance(cfg, dict):
            cfg = {"debug": False}
        config_b64 = base64.b64encode(json.dumps(cfg).encode("utf-8")).decode("utf-8")
        out = out.replace("{config_b64}", config_b64)

    if smithery_profile:
        try:
            parts = urlsplit(out)
            qs = dict(parse_qsl(parts.query, keep_blank_values=True))
            if "profile" not in qs:
                qs["profile"] = smithery_profile
                out = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(list(qs.items())), parts.fragment))
        except Exception:
            pass

    return out


def _parse_status_code(curl_stdout: str) -> Optional[int]:
    for line in (curl_stdout or "").splitlines():
        if line.startswith("HTTP/"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except Exception:
                    return None
    return None


@dataclass
class TestResult:
    server_slug: str
    schema_path: str
    python_sdk_url: Optional[str]
    masked_url: Optional[str]
    mode: str
    status_code: Optional[int]
    curl_exit_code: Optional[int]
    tool_count: Optional[int]
    error: Optional[str]


def _curl_test(url: str, headers: Dict[str, str], max_time_s: int, connect_timeout_s: int) -> Tuple[Optional[int], int, str]:
    curl_path = shutil.which("curl")
    if not curl_path:
        return None, 127, "curl not found"

    cmd = [
        curl_path,
        "-i",
        "--http1.1",
        "-N",
        "--max-time",
        str(max_time_s),
        "--connect-timeout",
        str(connect_timeout_s),
    ]
    for k, v in (headers or {}).items():
        cmd.extend(["-H", f"{k}: {v}"])
    cmd.append(url)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
        out = (p.stdout or "")
        code = _parse_status_code(out)
        err = (p.stderr or "").strip() or None
        return code, int(p.returncode), err or ""
    except Exception as e:
        return None, 1, repr(e)

 
def _parse_http_status_from_error(err: str) -> Optional[int]:
    if not err:
        return None
    m = re.search(r"\bHTTP\s+(\d{3})\b", err)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    m = re.search(r"\bstatus\s*code\s*[:=]?\s*(\d{3})\b", err, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    m = re.search(r"\b(\d{3})\b", err)
    if m:
        try:
            code = int(m.group(1))
            if 100 <= code <= 599:
                return code
        except Exception:
            return None
    return None


def _parse_headers(headers_json: str, header_lines: List[str]) -> Dict[str, str]:
    headers: Dict[str, str] = {"Accept": "text/event-stream"}
    if headers_json:
        try:
            obj = json.loads(headers_json)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(k, str) and isinstance(v, str):
                        headers[k] = v
        except Exception:
            pass
    for line in header_lines or []:
        if not isinstance(line, str):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        if k:
            headers[k] = v
    return headers


async def _list_tools_async(url: str, headers: Dict[str, str], timeout_s: int) -> Tuple[int, List[str]]:
    import mcp
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.streamable_http import streamablehttp_client

    async def _try_streamable() -> Tuple[int, List[str]]:
        async with streamablehttp_client(url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = [t.name for t in (tools.tools or [])]
                return len(names), names

    async def _try_sse() -> Tuple[int, List[str]]:
        streams = await sse_client(url, headers, sse_read_timeout=timeout_s)
        async with streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = [t.name for t in (tools.tools or [])]
                return len(names), names

    try:
        return await asyncio.wait_for(_try_streamable(), timeout=max(1, int(timeout_s)))
    except Exception:
        return await asyncio.wait_for(_try_sse(), timeout=max(1, int(timeout_s)))


def _run_one(payload: Tuple[str, str, str, str, str, str, int, int]) -> Dict[str, Any]:
    slug, schema_path_s, python_sdk_url, mode, smithery_api_key, smithery_profile, max_time, connect_timeout = payload
    schema_path = Path(schema_path_s)
    try:
        schema_obj = _load_json(schema_path)
    except Exception as e:
        return TestResult(
            server_slug=slug,
            schema_path=schema_path_s,
            python_sdk_url=None,
            masked_url=None,
            mode=mode,
            status_code=None,
            curl_exit_code=None,
            tool_count=None,
            error=f"json_load_error: {e}",
        ).__dict__

    url = _find_first_key(schema_obj, python_sdk_url)
    if not isinstance(url, str) or not url.strip():
        return TestResult(
            server_slug=slug,
            schema_path=schema_path_s,
            python_sdk_url=None,
            masked_url=None,
            mode=mode,
            status_code=None,
            curl_exit_code=None,
            tool_count=None,
            error="missing_python_sdk_url",
        ).__dict__

    config_obj = _find_first_key(schema_obj, "python_sdk_config")
    filled_url = _fill_placeholders(url.strip(), smithery_api_key, smithery_profile, config_obj)

    status_code: Optional[int] = None
    exit_code: Optional[int] = None
    tool_count: Optional[int] = None
    err: str = ""

    headers = _parse_headers(_find_first_key(schema_obj, "python_sdk_headers_json") or "", [])

    if mode == "list-tools":
        try:
            tool_count, _tool_names = asyncio.run(
                _list_tools_async(filled_url, headers=headers, timeout_s=max(1, int(max_time)))
            )
            status_code = 200
            exit_code = 0
        except Exception as e:
            err = str(e)
            status_code = _parse_http_status_from_error(err)
            exit_code = 1
    else:
        status_code, exit_code, err = _curl_test(
            filled_url,
            headers=headers,
            max_time_s=max(1, int(max_time)),
            connect_timeout_s=max(1, int(connect_timeout)),
        )

    if status_code is not None and status_code not in (401, 404):
        time.sleep(5)

    return TestResult(
        server_slug=slug,
        schema_path=schema_path_s,
        python_sdk_url=url.strip(),
        masked_url=_mask_url(filled_url),
        mode=mode,
        status_code=status_code,
        curl_exit_code=exit_code,
        tool_count=tool_count,
        error=err or None,
    ).__dict__


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    p.add_argument(
        "--schema-root",
        type=str,
        default=str(repo_root / "data" / "Input_preparation"),
    )
    p.add_argument("--smithery-api-key", type=str, default="")
    p.add_argument("--smithery-profile", type=str, default="")
    p.add_argument("--mode", type=str, default="curl", choices=["curl", "list-tools"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-time", type=int, default=8)
    p.add_argument("--connect-timeout", type=int, default=5)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    schema_root = Path(args.schema_root)

    results: List[Dict[str, Any]] = []
    total_found = 0
    total_tested = 0
    total_missing_url = 0
    total_ok = 0
    total_unauthorized = 0
    total_not_found = 0
    total_other = 0

    server_slugs = sorted(_iter_server_slugs(schema_root))
    work_items: List[Tuple[str, str, str, str, str, str, int, int]] = []
    for slug in server_slugs:
        schema_path = schema_root / slug / "json_schema.json"
        if schema_path.exists():
            total_found += 1
            work_items.append(
                (
                    slug,
                    str(schema_path),
                    "python_sdk_url",
                    str(args.mode),
                    str(args.smithery_api_key or ""),
                    str(args.smithery_profile or ""),
                    int(args.max_time),
                    int(args.connect_timeout),
                )
            )
        if isinstance(args.limit, int) and args.limit > 0 and len(work_items) >= args.limit:
            break

    workers = int(args.workers) if isinstance(args.workers, int) else 0
    if workers <= 0:
        cpu = os.cpu_count() or 4
        workers = min(16, cpu)

    iterator = None
    if tqdm is not None:
        iterator = tqdm(total=len(work_items), desc="Testing python_sdk_url", unit="srv")

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_run_one, it) for it in work_items]
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)

            err = r.get("error")
            if err == "missing_python_sdk_url":
                total_missing_url += 1

            status_code = r.get("status_code")
            if status_code is not None:
                total_tested += 1
                if isinstance(status_code, int) and 200 <= status_code < 300:
                    total_ok += 1
                elif status_code == 401:
                    total_unauthorized += 1
                elif status_code == 404:
                    total_not_found += 1
                else:
                    total_other += 1
            else:
                if err and err != "missing_python_sdk_url":
                    total_other += 1

            if iterator is not None:
                iterator.update(1)

    if iterator is not None:
        iterator.close()

    payload = {
        "schema_root": str(schema_root),
        "found_schemas": total_found,
        "tested_urls": total_tested,
        "missing_python_sdk_url": total_missing_url,
        "status": {
            "ok_2xx": total_ok,
            "unauthorized_401": total_unauthorized,
            "not_found_404": total_not_found,
            "other": total_other,
        },
        "results": results,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
