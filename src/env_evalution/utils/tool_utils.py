from typing import Any, Dict, List, Optional, Tuple
import re
import json


def extract_tools_from_schema(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    tools = schema.get("tools") or []
    for t in tools:
        if not isinstance(t, dict):
            continue
        name = t.get("name") or ""
        desc = t.get("description") or ""
        raw = t
        params = t.get("input_schema") or t.get("inputSchema") or t.get("parameters") or {}
        props: List[Dict[str, Any]] = []
        req: List[str] = []
        if isinstance(params, dict):
            pr = params.get("properties") or {}
            if isinstance(pr, dict):
                for k, v in pr.items():
                    if isinstance(k, str) and isinstance(v, dict):
                        tp = v.get("type") or "string"
                        dd = v.get("description") or ""
                        props.append({"name": k, "type": tp, "description": dd})
            rq = params.get("required") or []
            if isinstance(rq, list):
                req = [str(x) for x in rq if isinstance(x, str)]
        elif isinstance(params, list):
            for p in params:
                if not isinstance(p, dict):
                    continue
                pn = p.get("name") or ""
                tp = p.get("type") or "string"
                dd = p.get("description") or ""
                if pn:
                    props.append({"name": pn, "type": tp, "description": dd})
        out.append({
            "name": str(name),
            "description": str(desc),
            "parameters": props,
            "required": req,
            "raw_schema": raw,
        })
    return out


def extract_impl_records_from_code(code: Optional[str]) -> List[Dict[str, Any]]:
    if not code:
        return []
    s = code
    out: List[Dict[str, Any]] = []
    dec_re = re.compile(r"@\s*mcp\.tool\s*\(\s*\)\s*\n\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*:\s*\n", re.S)
    for m in dec_re.finditer(s):
        fn = m.group(1)
        sig = m.group(2)
        start = m.end()
        end = len(s)
        nxt = s.find("@mcp.tool()", start)
        if nxt != -1:
            end = nxt
        body = s[start:end]
        doc = ""
        dm = re.search(r"^[\t ]*\"\"\"([\s\S]*?)\"\"\"", body, re.M)
        if dm:
            doc = dm.group(1).strip()
        out.append({
            "tool_name": fn,
            "docstring": doc,
            "signature": f"def {fn}({sig})",
            "source": (body or "").strip(),
        })
    return out


def group_tools(
    gt_schema_obj: Dict[str, Any],
    gen_schema: Dict[str, Any],
    code: Optional[str],
    tool_map: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    gold_tools = extract_tools_from_schema(gt_schema_obj)
    gen_tools = extract_tools_from_schema(gen_schema)
    gold_by_name = {t.get("name"): t for t in gold_tools}
    gen_by_name = {t.get("name"): t for t in gen_tools}

    impls = extract_impl_records_from_code(code)
    impl_by_name = {r.get("tool_name"): r for r in impls}

    inv_map = {v: k for v, k in tool_map.items()} if tool_map else {}

    matched: List[Dict[str, Any]] = []
    for gname, gold in gold_by_name.items():
        gen_name = inv_map.get(gname)
        if gen_name and gen_name in gen_by_name:
            matched.append({
                "tool_name": gname,
                "gold": gold,
                "gen": gen_by_name.get(gen_name),
                "impl": impl_by_name.get(gen_name),
            })

    used_gen = {m["gen"].get("name") for m in matched if isinstance(m.get("gen"), dict)}

    extras: List[Dict[str, Any]] = []
    for gen in gen_tools:
        gn = gen.get("name")
        if gn not in used_gen:
            extras.append({
                "tool_name": gn,
                "gold": None,
                "gen": gen,
                "impl": impl_by_name.get(gn),
            })

    return matched, extras


def build_prompt_payload(kind: str, **kwargs) -> str:
    if kind == "matched":
        gold = kwargs.get("gold") or {}
        gen = kwargs.get("gen") or {}
        impl = kwargs.get("impl") or {}
        payload = {
            "gold": {
                "name": gold.get("name"),
                "description": gold.get("description"),
                "parameters": gold.get("parameters"),
                "required": gold.get("required"),
                "raw_schema": gold.get("raw_schema"),
            },
            "gen": {
                "name": gen.get("name"),
                "description": gen.get("description"),
                "parameters": gen.get("parameters"),
                "required": gen.get("required"),
                "raw_schema": gen.get("raw_schema"),
            },
            "impl": impl,
        }
        return json.dumps(payload, ensure_ascii=False)
    elif kind == "extra":
        gold_list = kwargs.get("gold_list") or []
        extra = kwargs.get("extra") or {}
        impl = kwargs.get("impl") or {}
        payload = {
            "gold_list": [{"name": g.get("name"), "description": g.get("description")} for g in gold_list],
            "extra": {
                "name": extra.get("name"),
                "description": extra.get("description"),
                "parameters": extra.get("parameters"),
                "required": extra.get("required"),
                "raw_schema": extra.get("raw_schema"),
            },
            "impl": impl,
        }
        return json.dumps(payload, ensure_ascii=False)
    return json.dumps({}, ensure_ascii=False)
