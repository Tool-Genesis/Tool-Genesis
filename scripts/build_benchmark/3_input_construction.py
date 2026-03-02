
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# ================= 配置区域 =================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENROUTER_API_KEY")
BASE_URL = (
    os.getenv("DEEPSEEK_BASE_URL")
    or os.getenv("OPENROUTER_BASE_URL")
    or "https://openrouter.ai/api/v1"
)

# 初始化 Client（按需检查密钥）
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL) if DEEPSEEK_API_KEY else None

def generate_requirements_doc(tools_info):
    """
    使用 DeepSeek 将具体的工具列表转化为抽象的需求文档。
    """
    if client is None:
        raise ValueError(
            "Missing API key. Set DEEPSEEK_API_KEY or OPENROUTER_API_KEY in environment."
        )

    system_prompt = """
You are an expert Technical Product Manager and API Architect.
Your task is to reverse-engineer a "Functional Requirement Document" (PRD) based on a list of existing API tools.

**Goal:** Create a set of requirements that describes WHAT the system should do (Capabilities), and WHICH technology stack to use, without strictly dictating the exact function names.

**CRITICAL RULES:**
1. **Identify Technical Provider:** Look at the input tools. If they belong to a specific known service (e.g., Exa, Stripe, Slack, GitHub, Linear), you **MUST** explicitly state this in a "Technical Context" section. This is crucial for the implementation phase.
2. **One-to-One Mapping:** You must extract exactly ONE functional capability bullet point for EACH tool in the input list.
3. **Abstract Naming:** Do NOT use the exact `tool_name` from the input in the capability title. (e.g., if input is `exa-search-web`, refer to it as "Feature: Web Search").
4. **No Signature Leakage:** Do not describe the exact JSON parameter structure, but DO describe the *information* needed.

**Output Format:**
- **System Scope:** [1-sentence summary of the domain]
- **Technical Context:** [Explicitly name the API/Service provider found in the tools, e.g., "The system must be implemented using the **Exa AI API**."]
- **Required Capabilities:**
  1. **Feature: [Descriptive Name]**
     - Description: ...
     - Key Inputs: ...
"""

    user_prompt = f"""
Here is the list of existing tools (Ground Truth):
{json.dumps(tools_info, indent=2)}

Please generate the Functional Requirement Document based on these tools.
"""

    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-v3.2",  # 或者 deepseek-coder
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3, # 低温度以保持稳重，不需要太发散
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API Call Error: {e}")
        return None

# ================= 主执行流程 =================

def _load_schema_json(schema_path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _select_schema_parts(data: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in ["analysis", "reasoning", "primary_label", "secondary_labels", "server_id", "server_name", "rank_by_usage", "usage_count", "original_file", "mode", "timestamp"]:
        v = data.get(k)
        if v is not None:
            out[k] = v
    rs = data.get("remote_server_response")
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
                    for kk in ["name", "description", "input_schema", "annotations"]:
                        vv = t.get(kk)
                        if vv is not None:
                            ct[kk] = vv
                    cleaned_tools.append(ct)
            rs_out["tools"] = cleaned_tools
        out["remote_server_response"] = rs_out
    return out

def _update_server_status_requirements(inprep: str, slug: str, ok: bool) -> None:
    status_path = os.path.join(inprep, "server_status.json")
    try:
        if not os.path.exists(status_path):
            return
        with open(status_path, "r", encoding="utf-8") as sf:
            status = json.load(sf)
        if not isinstance(status, list):
            return
        for entry in status:
            if isinstance(entry, dict) and entry.get("server_slug") == slug:
                entry["requirement_document"] = bool(ok)
                break
        with open(status_path, "w", encoding="utf-8") as sf:
            json.dump(status, sf, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _process_server(args: Tuple[str, str, bool]) -> Tuple[str, bool, Optional[str]]:
    inprep, slug, overwrite = args
    dpath = os.path.join(inprep, slug)
    schema_path = os.path.join(dpath, "json_schema.json")
    if not os.path.exists(schema_path):
        return slug, False, "missing_schema"
    out_txt = os.path.join(dpath, "requirements_document.txt")
    if not overwrite and os.path.exists(out_txt):
        return slug, False, "exists"
    data = _load_schema_json(schema_path)
    if not isinstance(data, dict):
        return slug, False, "invalid_schema"
    minimal = _select_schema_parts(data)
    content = generate_requirements_doc(minimal)
    if not isinstance(content, str) or not content.strip():
        return slug, False, "empty_content"
    try:
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(content)
        return slug, True, None
    except Exception:
        return slug, False, "write_error"

def build_requirements_for_servers(base_dir: str, overwrite: bool = False, workers: int = 1, chunksize: int = 16) -> int:
    inprep = base_dir
    if not os.path.isdir(inprep):
        return 0
    status_path = os.path.join(inprep, "server_status.json")
    targets: List[str] = []
    if not overwrite and os.path.exists(status_path):
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                status = json.load(f)
            if isinstance(status, list):
                for entry in status:
                    if isinstance(entry, dict) and not bool(entry.get("requirement_document")):
                        slug = entry.get("server_slug")
                        if isinstance(slug, str) and os.path.isdir(os.path.join(inprep, slug)):
                            targets.append(slug)
        except Exception:
            targets = []
    if not targets:
        targets = [d for d in sorted(os.listdir(inprep)) if os.path.isdir(os.path.join(inprep, d))]
    print(targets)
    written = 0
    if workers and workers > 1:
        bar = tqdm(total=len(targets), desc="Generate requirements")
        with Pool(processes=workers) as pool:
            it = ((inprep, d, overwrite) for d in targets)
            for slug, ok, reason in pool.imap_unordered(_process_server, it, chunksize=max(1, chunksize)):
                bar.update(1)
                if ok:
                    written += 1
                    _update_server_status_requirements(inprep, slug, True)
        try:
            bar.close()
        except Exception:
            pass
    else:
        for d in tqdm(targets, desc="Generate requirements"):
            slug, ok, reason = _process_server((inprep, d, overwrite))
            if ok:
                written += 1
                _update_server_status_requirements(inprep, slug, True)
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
    base_dir = args.base_dir or os.path.join(root, "data", "Input_preparation")
    total = build_requirements_for_servers(base_dir, args.overwrite, args.workers, args.chunksize)
    print(f"Generated {total} requirement documents")

if __name__ == "__main__":
    main()
