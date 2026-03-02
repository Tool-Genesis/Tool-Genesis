import json
import os
import sys
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x
from openai import OpenAI
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_API_BASE_URL", "https://openrouter.ai/api/v1")
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)
from src.utils.llm import call_llm

def _extract_json_text(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None
    s = text.strip()
    jstart = s.find("```json")
    if jstart != -1:
        body = s[jstart + len("```json"):]
        jend = body.find("```")
        if jend != -1:
            body = body[:jend]
        try:
            return json.loads(body.strip())
        except Exception:
            pass
    if s.startswith("```"):
        try:
            nl = s.find("\n")
            if nl != -1:
                s = s[nl + 1 :]
            if s.endswith("```"):
                s = s[:-3]
        except Exception:
            pass
        s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except Exception:
            return None
    return None

def generate_server_classification(server_name: Optional[str], minimal: Dict[str, Any], tool_names: List[str], call_samples: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    instructions = ("""
You are an auditor for an MCP server.

Task 1: Classify the server as 'stateless' or 'stateful'.
Task 2: Determine whether running this server requires external API credentials or access keys ('requires_api': true/false).
Task 3: Assign a sandbox requirement level for running this server in isolation:

L0: Pure — pure functions / pure Python compute (no IO).
L1: File-only — only reads/writes local working directory files (covered by venv+fs).
L2: Network — needs HTTP/download/external services (requires network/proxy/rate limits).
L3: OS-packages — needs apt/yum/brew, gcc/make, system deps, browser drivers, etc.
L4: Services/Daemon — needs postgres/redis/systemd/background services/port binding.
L5: Privileged/Host — needs sudo, mounts, kernel modules, Docker-in-Docker, GPU/CUDA, device access.

Definitions for Task 1:
- Stateless: does NOT depend on cross-call persistent state or external mutable state; unit tests can be treated as input->output. 
  NOTE: reading/writing files under the local working directory as temporary artifacts ALONE does NOT necessarily make it stateful.
- Stateful: depends on cross-call persistent state or external mutable state (filesystem outside working dir, DB/cache, remote stateful service, requires prior initialization/login/seed), 
  so unit tests are (input,state)->(output,next_state).

Guidelines:
- Be conservative: if evidence is insufficient, choose:
  - 'class': 'stateless'
  - 'requires_api': false
  - 'sandbox_level': 'L1'
- Use the schema summary, tool names, and sampled calls to infer all judgments.
- For Task 3:
  - Installing Python dependencies via pip/venv does NOT imply L3.
  - L3 is specifically for OS-level package managers, build toolchains (gcc/make), system libraries, browser drivers, etc.
- Choose the MINIMUM sufficient sandbox level that can run the server in isolation.
  If evidence suggests multiple requirements, choose the HIGHEST required level among them (L5 > L4 > L3 > L2 > L1 > L0).
- Keep reasoning concise and tied to evidence from schema/tool names/call samples.

Output strictly a JSON object with keys:
{server_name, class, class_reasoning, requires_api, requires_api_reasoning, sandbox_level, sandbox_reasoning}

Return your answer wrapped ONLY as a fenced block:
```json
{"server_name":"...","class":"stateless|stateful","class_reasoning":"...","requires_api":false,"requires_api_reasoning":"...","sandbox_level":"L1","sandbox_reasoning":"..."}
```

Do not include any extra text outside the fenced JSON.
""")
    payload = {
        "server_name": server_name,
        "schema_summary": minimal,
        "tool_names_sample": tool_names[:10],
        "call_samples": call_samples[:10],
    }
    user_prompt = instructions + "\n\nINPUT:\n```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"
    try:
        txt = call_llm(user_prompt,max_tokens=1024, temperature=0.0)
        obj = _extract_json_text(txt)
        print(obj)
        return obj
    except Exception as e:
        print(f"API Call Error: {e}")
        return None

def _load_schema_json(schema_path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _select_schema_parts(data: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in [
        "analysis",
        "reasoning",
        "primary_label",
        "secondary_labels",
        "server_id",
        "server_name",
        "rank_by_usage",
        "usage_count",
        "original_file",
        "mode",
        "timestamp",
    ]:
        v = data.get(k)
        if v is not None:
            out[k] = v
    meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else None
    container = meta if isinstance(meta, dict) else data
    rs = container.get("remote_server_response")
    if isinstance(rs, dict):
        rs_out: Dict[str, Any] = {}
        for k in ["url", "is_success", "error", "tool_count", "tool_names"]:
            v = rs.get(k)
            if v is not None:
                rs_out[k] = v
        tools = rs.get("tools")
        if isinstance(tools, list):
            cleaned_tools: List[Dict[str, Any]] = []
            for t in tools:
                if isinstance(t, dict):
                    ct: Dict[str, Any] = {}
                    for kk in ["name", "description", "input_schema", "inputSchema", "annotations"]:
                        vv = t.get(kk)
                        if vv is not None:
                            ct[kk] = vv
                    cleaned_tools.append(ct)
            rs_out["tools"] = cleaned_tools
        out["remote_server_response"] = rs_out
    return out

def _server_calls_path(base_dir: str, slug: str) -> str:
    return os.path.join(base_dir, slug, "clean_tool_call_logs.dedup.json")

def _norm_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(v)

def _load_call_samples(base_dir: str, slug: str, sample_k: int = 10) -> List[Dict[str, Any]]:
    fp = _server_calls_path(base_dir, slug)
    if not os.path.exists(fp):
        return []
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    calls: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                cs = item.get("calls")
                if isinstance(cs, list) and cs:
                    for c in cs:
                        if isinstance(c, dict):
                            one = {
                                "function_name": c.get("function_name"),
                                "arguments": _norm_text(c.get("arguments")),
                                "function_output_content": _norm_text(c.get("function_output_content")),
                            }
                            calls.append(one)
                else:
                    fn = item.get("function_name")
                    if isinstance(fn, str):
                        one = {
                            "function_name": fn,
                            "arguments": _norm_text(item.get("arguments")),
                            "function_output_content": _norm_text(item.get("function_output_content")),
                        }
                        calls.append(one)
    elif isinstance(data, dict):
        cs = data.get("calls")
        if isinstance(cs, list):
            for c in cs:
                if isinstance(c, dict):
                    one = {
                        "function_name": c.get("function_name"),
                        "arguments": _norm_text(c.get("arguments")),
                        "function_output_content": _norm_text(c.get("function_output_content")),
                    }
                    calls.append(one)
    random.shuffle(calls)
    return calls[:sample_k]

def _process_server(args: Tuple[str, str, bool]) -> Tuple[str, bool, Optional[str]]:
    inprep, slug, overwrite = args
    dpath = os.path.join(inprep, slug)
    tool_state_json = os.path.join(dpath, "tool_state_classification.json")
    try:
        if os.path.exists(tool_state_json):
            os.remove(tool_state_json)
    except Exception:
        pass
    schema_path = os.path.join(dpath, "json_schema.json")
    if not os.path.exists(schema_path):
        return slug, False, "missing_schema"
    out_json = os.path.join(dpath, "server_state_classification.json")
    if not overwrite and os.path.exists(out_json):
        return slug, False, "exists"
    data = _load_schema_json(schema_path)
    if not isinstance(data, dict):
        return slug, False, "invalid_schema"
    minimal = _select_schema_parts(data)
    server_name = minimal.get("server_name")
    rs = minimal.get("remote_server_response") or {}
    tools = rs.get("tools") or []
    tool_names = []
    for t in tools:
        if isinstance(t, dict):
            nm = t.get("name")
            if isinstance(nm, str):
                tool_names.append(nm)
    call_samples = _load_call_samples(inprep, slug, sample_k=10)
    one = generate_server_classification(server_name, minimal, tool_names, call_samples)
    if isinstance(one, dict):
        cl = str(one.get("class") or "stateless").strip().lower()
        class_reason = one.get("class_reasoning")
        req_api = bool(one.get("requires_api")) if one.get("requires_api") is not None else False
        req_api_reason = one.get("requires_api_reasoning")
        sandbox_level = one.get("sandbox_level")
        sandbox_reason = one.get("sandbox_reasoning")
        payload = {
            "server_name": server_name,
            "server_class": cl,
            "server_class_reasoning": class_reason,
            "requires_api": req_api,
            "requires_api_reasoning": req_api_reason,
            "sandbox_level": sandbox_level,
            "sandbox_reasoning": sandbox_reason,
        }
    else:
        payload = {
            "server_name": server_name,
            "server_class": "stateless",
            "server_class_reasoning": "no evidence",
            "requires_api": False,
            "requires_api_reasoning": "no evidence",
            "sandbox_level": "L1",
            "sandbox_reasoning": "no evidence",
        }
    try:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return slug, True, None
    except Exception:
        return slug, False, "write_error"

def build_classification_for_servers(base_dir: str, overwrite: bool = False, workers: int = 1, chunksize: int = 16) -> int:
    inprep = base_dir
    if not os.path.isdir(inprep):
        return 0
    targets: List[str] = []
    for d in sorted(os.listdir(inprep)):
        if os.path.isdir(os.path.join(inprep, d)):
            targets.append(d)
    written = 0
    if workers and workers > 1:
        bar = tqdm(total=len(targets), desc="Categorize servers")
        with Pool(processes=workers) as pool:
            it = ((inprep, d, overwrite) for d in targets)
            for slug, ok, reason in pool.imap_unordered(_process_server, it, chunksize=max(1, chunksize)):
                bar.update(1)
                if ok:
                    written += 1
        try:
            bar.close()
        except Exception:
            pass
    else:
        for d in tqdm(targets, desc="Categorize servers"):
            slug, ok, reason = _process_server((inprep, d, overwrite))
            if ok:
                written += 1
    return written

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=str, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    p.add_argument("--chunksize", type=int, default=max(16, (os.cpu_count() or 1) * 2))
    return p.parse_args()

def main():
    args = parse_args()
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    base_dir = args.base_dir or os.path.join(root, "data", "Input_selected")
    total = build_classification_for_servers(base_dir, args.overwrite, args.workers, args.chunksize)
    print(f"Generated {total} server state classification documents")

if __name__ == "__main__":
    main()
