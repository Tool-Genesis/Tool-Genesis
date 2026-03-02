import json
import os
from typing import Optional
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

def _slug(s: Optional[str]) -> str:
    if not s or not isinstance(s, str):
        return "unknown-server"
    return "-".join(s.strip().lower().split())

def _json_load_maybe(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return None
    return v

def _find_mcp_json(mcp_root: str, server_name: Optional[str], server_slug: str) -> Optional[str]:
    names = [n for n in os.listdir(mcp_root) if n.endswith(".json")]
    for n in sorted(names):
        fp = os.path.join(mcp_root, n)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        meta = data.get("metadata")
        meta = _json_load_maybe(meta) or {}
        nm = meta.get("server_name")
        if isinstance(nm, str) and server_name and nm.strip() == server_name.strip():
            return fp
    for n in sorted(names):
        if server_slug in n:
            return os.path.join(mcp_root, n)
    return None

def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    inprep = os.path.join(root, "data", "Input_preparation")
    mcp_root = os.path.join(root, "data", "mcp_servers")
    dirs = [d for d in sorted(os.listdir(inprep)) if os.path.isdir(os.path.join(inprep, d))]
    for d in tqdm(dirs, desc="Match MCP schemas"):
        dpath = os.path.join(inprep, d)
        tpath = os.path.join(dpath, "task_example.json")
        if not os.path.exists(tpath):
            continue
        try:
            with open(tpath, "r", encoding="utf-8") as f:
                items = json.load(f)
        except Exception:
            items = []
        server_name = None
        if isinstance(items, list) and items:
            sn = items[0].get("server_name")
            if isinstance(sn, str):
                server_name = sn
        srv_slug = d
        src = _find_mcp_json(mcp_root, server_name, srv_slug)
        if not src:
            print(f"Skip {d}: schema not found")
            continue
        outp = os.path.join(dpath, "json_schema.json")
        with open(src, "r", encoding="utf-8") as f:
            content = f.read()
        with open(outp, "w", encoding="utf-8") as f:
            f.write(content)
        # update server_status.json -> MCP-server-json-schema
        status_path = os.path.join(inprep, "server_status.json")
        try:
            if os.path.exists(status_path):
                with open(status_path, "r", encoding="utf-8") as sf:
                    status = json.load(sf)
                if isinstance(status, list):
                    for entry in status:
                        if isinstance(entry, dict) and entry.get("server_slug") == d:
                            entry["MCP-server-json-schema"] = True
                            break
                    with open(status_path, "w", encoding="utf-8") as sf:
                        json.dump(status, sf, ensure_ascii=False, indent=2)
        except Exception:
            pass

if __name__ == "__main__":
    main()
