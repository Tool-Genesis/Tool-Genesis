import os
import json
import glob
import re
from typing import Optional, Dict, Any, List, Union, Tuple

# Matches: run_prompt_tokens=123 / run_completion_tokens=45 / run_total_tokens=168
_PROMPT_RE = re.compile(r"run_prompt_tokens\s*=\s*(\d+)")
_COMPLETION_RE = re.compile(r"run_completion_tokens\s*=\s*(\d+)")
_TOTAL_RE = re.compile(r"run_total_tokens\s*=\s*(\d+)")


def _extract_usage_triplet_from_text(txt: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    从一段纯文本中提取最后一次出现的:
      - run_prompt_tokens
      - run_completion_tokens
      - run_total_tokens
    """
    p = None
    c = None
    t = None

    m = _PROMPT_RE.findall(txt)
    if m:
        p = int(m[-1])

    m = _COMPLETE_RE_FINDALL = _COMPLETION_RE.findall(txt)
    if m:
        c = int(m[-1])

    m = _TOTAL_RE.findall(txt)
    if m:
        t = int(m[-1])

    return p, c, t


def _extract_usage_triplet_from_content(content: Union[str, dict, list]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    从 content 字段（字符串 / dict含text / 列表）中提取三元组 (prompt, completion, total)。
    如果多处出现，返回“最后一次出现”的值。
    """
    last_p: Optional[int] = None
    last_c: Optional[int] = None
    last_t: Optional[int] = None

    def _merge(p: Optional[int], c: Optional[int], t: Optional[int]):
        nonlocal last_p, last_c, last_t
        if p is not None:
            last_p = p
        if c is not None:
            last_c = c
        if t is not None:
            last_t = t

    if isinstance(content, str):
        _merge(*_extract_usage_triplet_from_text(content))

    elif isinstance(content, dict):
        # 常见形态：{"type":"text","text":"..."} 或更复杂对象
        txt = content.get("text")
        if isinstance(txt, str):
            _merge(*_extract_usage_triplet_from_text(txt))

    elif isinstance(content, list):
        # 列表里可能混有 str 或 {"type":"text","text":...}
        for block in content:
            if isinstance(block, str):
                _merge(*_extract_usage_triplet_from_text(block))
            elif isinstance(block, dict):
                txt = block.get("text")
                if isinstance(txt, str):
                    _merge(*_extract_usage_triplet_from_text(txt))

    return last_p, last_c, last_t


def _extract_last_usage_triplet_from_items(items: List[dict]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    在一个会话(item列表)里，找 role == 'usage' 的所有消息，解析其 content，
    并返回最后一条 usage 的 (prompt, completion, total) 三元组（逐次覆盖）。
    """
    last_p: Optional[int] = None
    last_c: Optional[int] = None
    last_t: Optional[int] = None

    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("role") != "usage":
            continue
        content = item.get("content", "")
        p, c, t = _extract_usage_triplet_from_content(content)
        if p is not None:
            last_p = p
        if c is not None:
            last_c = c
        if t is not None:
            last_t = t

    # 如果 total 缺失但有 p/c，可补全；否则保持 None
    if last_t is None and (last_p is not None or last_c is not None):
        last_t = (last_p or 0) + (last_c or 0)

    return last_p, last_c, last_t


def _load_as_array_or_jsonl(filepath: str) -> List[dict]:
    """
    以 JSON array 或 JSONL 方式加载 conversation 文件，统一返回 list[dict]。
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # 先尝试 JSON
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("messages"), list):
            return data["messages"]
    except json.JSONDecodeError:
        pass

    # 回退到 JSONL
    items: List[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                items.append(obj)
        except json.JSONDecodeError:
            continue
    return items


def compute_average_run_tokens(base_dir: str, server_name: Optional[str] = None) -> Dict[str, Any]:
    """
    扫描 base_dir 下所有会话日志，取每个会话最后一条 usage 的
    run_prompt_tokens / run_completion_tokens / run_total_tokens，
    分别计算总和与平均值（各自对各自“有值的会话”进行平均）。
    """
    if server_name:
        pattern = os.path.join(base_dir, server_name, "conversation", "*", "logs", "conversation.json")
    else:
        pattern = os.path.join(base_dir, "*", "conversation", "*", "logs", "conversation.json")

    files = glob.glob(pattern, recursive=False)

    files_scanned = 0
    prompt_list: List[int] = []
    completion_list: List[int] = []
    total_list: List[int] = []

    for fp in files:
        files_scanned += 1
        try:
            items = _load_as_array_or_jsonl(fp)
            p, c, t = _extract_last_usage_triplet_from_items(items)
            if p is not None:
                prompt_list.append(p)
            if c is not None:
                completion_list.append(c)
            if t is not None:
                total_list.append(t)
        except (OSError, IOError):
            continue

    sum_prompt = sum(prompt_list)
    sum_completion = sum(completion_list)
    sum_total = sum(total_list)

    cnt_prompt = len(prompt_list)
    cnt_completion = len(completion_list)
    cnt_total = len(total_list)

    avg_prompt = float(sum_prompt) / cnt_prompt if cnt_prompt else 0.0
    avg_completion = float(sum_completion) / cnt_completion if cnt_completion else 0.0
    avg_total = float(sum_total) / cnt_total if cnt_total else 0.0

    return {
        "files_scanned": files_scanned,
        "conversations_counted_prompt": cnt_prompt,
        "conversations_counted_completion": cnt_completion,
        "conversations_counted_total": cnt_total,
        "sum_prompt_tokens": sum_prompt,
        "sum_completion_tokens": sum_completion,
        "sum_total_tokens": sum_total,
        "average_prompt_tokens": avg_prompt,
        "average_completion_tokens": avg_completion,
        "average_total_tokens": avg_total,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compute average run tokens (prompt/completion/total) from conversation logs.")
    parser.add_argument("base_dir", type=str, help="Base directory to scan, e.g. temp/agentic/envs")
    parser.add_argument("--server-name", type=str, default=None, help="Optional server name filter, e.g. bazi_mcp")
    args = parser.parse_args()

    result = compute_average_run_tokens(args.base_dir, args.server_name)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()