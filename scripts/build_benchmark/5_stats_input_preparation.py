import argparse
import json
import os

def _json_load(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def collect_stats(input_dir: str):
    servers = 0
    stats = {}
    for name in sorted(os.listdir(input_dir)):
        sdir = os.path.join(input_dir, name)
        if not os.path.isdir(sdir):
            continue
        schema_path = os.path.join(sdir, "json_schema.json")
        if not os.path.exists(schema_path):
            continue
        servers += 1
        tdir = os.path.join(sdir, "tool_call")
        if not os.path.isdir(tdir):
            continue
        for fn in os.listdir(tdir):
            if not fn.endswith(".json"):
                continue
            fp = os.path.join(tdir, fn)
            data = _json_load(fp)
            cnt = len(data) if isinstance(data, list) else 0
            tool = os.path.splitext(fn)[0]
            if tool not in stats:
                stats[tool] = {"count": 0, "servers": 0}
            stats[tool]["count"] += cnt
            stats[tool]["servers"] += 1
    return servers, stats

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=str, default=None)
    args = p.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    input_dir = args.input_dir or os.path.join(root, "data", "Input_preparation")
    servers, stats = collect_stats(input_dir)
    total_counts = sum(v["count"] for v in stats.values())
    total_occurrences = sum(v["servers"] for v in stats.values())
    global_avg = (total_counts / total_occurrences) if total_occurrences > 0 else 0.0
    print(f"Servers: {servers}")
    print(f"GlobalAvgUnitTestsPerTool: {global_avg:.2f}")

if __name__ == "__main__":
    main()
