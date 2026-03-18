import re
import math
import json
import difflib
import logging
from typing import Any, Dict, List, Optional, Tuple, Set
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
logger = logging.getLogger(__name__)

TOOL_MATCH_REQUIRE_THRESHOLD: bool = True
TOOL_MATCH_THRESHOLD: float = 0.3
ARG_MATCH_REQUIRE_THRESHOLD: bool = True
ARG_MATCH_THRESHOLD: float = 0.3
ENABLE_DEBUG_PRINTS: bool = False
HARD_MATCH_THRESHOLD: float = 0.7

from src.utils import call_embedding
from src.core.toolkits.mcp_sandbox_bridge_toolkit import MCPServerToolsToolkit
from .execute_task import ExecuteTask

def _f1(pred: List[str], gt: List[str]) -> float:
    ps = set(pred)
    gs = set(gt)
    tp = len(ps & gs)
    if tp == 0:
        return 0.0
    prec = tp / max(1, len(ps))
    rec = tp / max(1, len(gs))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _schema_fidelity(schema: Dict[str, Any], gt_schema: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str], float, Dict[str, Any]]:
    def _extract_tools(s: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        tools = s.get("tools") or []
        for t in tools:
            if not isinstance(t, dict):
                continue
            name = t.get("name") or ""
            desc = t.get("description") or ""
            params = t.get("input_schema") or t.get("inputSchema") or t.get("parameters")
            args: List[Dict[str, str]] = []

            if isinstance(params, dict):
                props = params.get("properties") or {}
                if isinstance(props, dict):
                    for k, v in props.items():
                        if isinstance(k, str):
                            adesc = v.get("description") if isinstance(v, dict) else ""
                            args.append({"name": k, "description": adesc or ""})
            elif isinstance(params, list):
                for p in params:
                    if not isinstance(p, dict):
                        continue
                    an = p.get("name") or ""
                    ad = p.get("description") or ""
                    if an:
                        args.append({"name": an, "description": ad})

            out.append({"name": str(name), "description": str(desc), "args": args})
        return out

    pred_tools = _extract_tools(schema)
    gt_tools = _extract_tools(gt_schema)
    if not pred_tools or not gt_tools:
        return {}, {}, 0.0, {"tool_matches": [], "arg_matches": [], "tool_f1": 0.0, "args_f1": 0.0, "schema_f1": 0.0}

    def _norm(v: List[float]) -> List[float]:
        s = math.sqrt(sum(x * x for x in v))
        if s == 0:
            return v
        return [x / s for x in v]

    def _sim(a: List[float], b: List[float]) -> float:
        na = _norm(a)
        nb = _norm(b)
        # 假设 embedding 维度一致
        return sum(x * y for x, y in zip(na, nb))
    BIG_NEG = -1e9

    def _norm_tool_name(name: str) -> str:
        return (name or "").replace("_", "").lower()

    def _hungarian_maximize(sim, threshold: float = None) -> List[int]:
        try:
            import numpy as np
            from scipy.optimize import linear_sum_assignment
        except Exception:
            n_pred = len(sim)
            n_gt = len(sim[0]) if n_pred > 0 else 0
            assign = [-1] * n_pred
            used: Set[int] = set()
            for i in range(n_pred):
                bj = -1
                bs = float("-inf")
                for j in range(n_gt):
                    s = float(sim[i][j])
                    if threshold is not None and s < threshold:
                        continue
                    if j in used:
                        continue
                    if s > bs:
                        bs = s
                        bj = j
                if bj != -1:
                    assign[i] = bj
                    used.add(bj)
            return assign

        S = np.asarray(sim, dtype=np.float32)
        if S.size == 0:
            return []
        n_pred, n_gt = S.shape

        if threshold is not None:
            S = S.copy()
            S[S < threshold] = BIG_NEG

        dummy = np.zeros((n_pred, n_pred), dtype=S.dtype)
        S_aug = np.concatenate([S, dummy], axis=1)

        try:
            row_ind, col_ind = linear_sum_assignment(S_aug, maximize=True)
        except TypeError:
            valid = S_aug[S_aug > BIG_NEG / 2]
            maxv = float(valid.max()) if valid.size else 0.0
            cost = maxv - S_aug
            cost[S_aug <= BIG_NEG / 2] = 1e9
            row_ind, col_ind = linear_sum_assignment(cost)

        assign = [-1] * n_pred
        for r, c in zip(row_ind, col_ind):
            if r < n_pred and c < n_gt and S_aug[r, c] > BIG_NEG / 2:
                assign[r] = int(c)
        return assign

    # -------------------------
    # 1) Tool-level matching
    # -------------------------
    def _canon_tool(t: Dict[str, Any]) -> str:
        """Canonical serialization: name ⊕ canon(args schema), matching paper §A.2."""
        name = t.get("name", "")
        args = t.get("args", [])
        if args:
            arg_strs = sorted(f"{a['name']}:{a.get('description','')}" for a in args if a.get("name"))
            return f"{name} {' '.join(arg_strs)}"
        return name

    pred_texts = [_canon_tool(t) for t in pred_tools]
    gt_texts = [_canon_tool(t) for t in gt_tools]
    if ENABLE_DEBUG_PRINTS:
        print(pred_texts)
        print(gt_texts)
    pred_emb = call_embedding(pred_texts)
    gt_emb = call_embedding(gt_texts)
    if pred_emb is None or pred_emb == [] or gt_emb is None or gt_emb == []:
        logger.error("Embedding is None or empty")
        return {}, {}, 0.0, {"tool_matches": [], "arg_matches": [], "tool_f1": 0.0, "args_f1": 0.0, "schema_f1": 0.0}
    used_gt: Set[int] = set()
    used_pred: Set[int] = set()
    tool_map: Dict[str, str] = {}
    tool_match_details: List[Dict[str, Any]] = []

    gt_norm2idx: Dict[str, List[int]] = {}
    for j, gt in enumerate(gt_tools):
        gt_norm2idx.setdefault(_norm_tool_name(gt.get("name") or ""), []).append(j)

    for i, pt in enumerate(pred_tools):
        key = _norm_tool_name(pt.get("name") or "")
        cand = gt_norm2idx.get(key) or []
        pick = next((jj for jj in cand if jj not in used_gt), None)
        if pick is None:
            continue
        tool_map[pt["name"]] = gt_tools[pick]["name"]
        tool_match_details.append({"pred": pt["name"], "gt": gt_tools[pick]["name"], "score": 1.0})
        used_gt.add(pick)
        used_pred.add(i)

    remaining_pred_idx = [i for i in range(len(pred_tools)) if i not in used_pred]
    remaining_gt_idx = [j for j in range(len(gt_tools)) if j not in used_gt]
    if remaining_pred_idx and remaining_gt_idx:
        sim_mat = [[_sim(pred_emb[i], gt_emb[j]) for j in remaining_gt_idx] for i in remaining_pred_idx]
        thr = TOOL_MATCH_THRESHOLD if TOOL_MATCH_REQUIRE_THRESHOLD else None
        assign = _hungarian_maximize(sim_mat, threshold=thr)
        for a, b in enumerate(assign):
            if b == -1:
                continue
            pi = remaining_pred_idx[a]
            gj = remaining_gt_idx[b]
            s = sim_mat[a][b]
            tool_map[pred_tools[pi]["name"]] = gt_tools[gj]["name"]
            tool_match_details.append({"pred": pred_tools[pi]["name"], "gt": gt_tools[gj]["name"], "score": s})
            used_gt.add(gj)

    # Tool F1（用你原来的“P::”策略）
    unmatched_pred_tools = [t["name"] for t in pred_tools if t["name"] not in tool_map]
    matched_gt_tool_names = list(tool_map.values())

    pred_for_f1 = matched_gt_tool_names + [f"P::{n}" for n in unmatched_pred_tools]
    gt_for_f1 = [t["name"] for t in gt_tools]
    tool_f1 = _f1(pred_for_f1, gt_for_f1)

    # -------------------------
    # 2) Arg-level matching (per matched tool)
    #    但 args_f1 最终改为“整体一次算”
    # -------------------------
    arg_map: Dict[str, str] = {}
    arg_match_details: List[Dict[str, Any]] = []

    # 为“整体 args_f1”准备全局集合
    # GT 全局参数（仅统计已匹配工具对应的 GT 工具）
    gt_args_global: List[str] = []
    # Pred 全局参数将由两部分组成：
    # - arg_map.values()（已匹配到的 GT 参数 fq 名）
    # - P:: 前缀的未匹配预测参数 fq 名
    unmatched_pred_args_global: List[str] = []

    for p_name, g_name in tool_map.items():
        p_tool = next((t for t in pred_tools if t["name"] == p_name), None)
        g_tool = next((t for t in gt_tools if t["name"] == g_name), None)
        if not p_tool or not g_tool:
            continue

        p_args = p_tool.get("args") or []
        g_args = g_tool.get("args") or []

        # GT 参数（全局）
        for a in g_args:
            an = a.get("name")
            if an:
                gt_args_global.append(f"{g_name}.{an}")

        # 如果任意一侧无参数，就跳过 embedding matching
        # 但要把预测侧的参数作为“未匹配”计入全局
        if not p_args or not g_args:
            for a in p_args:
                an = a.get("name")
                if an:
                    unmatched_pred_args_global.append(f"P::{p_name}.{an}")
            continue

        p_texts = [f"{a['name']} {a.get('description','')}" for a in p_args]
        g_texts = [f"{a['name']} {a.get('description','')}" for a in g_args]

        p_emb = call_embedding(p_texts)
        g_emb = call_embedding(g_texts)

        if not p_emb or not g_emb:
            # Embedding call failed; treat all pred args as unmatched
            for a in p_args:
                an = a.get("name")
                if an:
                    unmatched_pred_args_global.append(f"P::{p_name}.{an}")
            continue

        sim_mat = [[_sim(pe2, ge2) for ge2 in g_emb] for pe2 in p_emb]
        thr2 = ARG_MATCH_THRESHOLD if ARG_MATCH_REQUIRE_THRESHOLD else None
        assign2 = _hungarian_maximize(sim_mat, threshold=thr2)

        for i2, j2 in enumerate(assign2):
            if j2 == -1:
                continue
            pn = p_args[i2]["name"]
            gn = g_args[j2]["name"]
            arg_map[f"{p_name}.{pn}"] = f"{g_name}.{gn}"
            arg_match_details.append({"pred": f"{p_name}.{pn}", "gt": f"{g_name}.{gn}", "score": sim_mat[i2][j2]})

        # 将未匹配的预测参数纳入全局 FP
        for a in p_args:
            pn = a.get("name")
            if not pn:
                continue
            key = f"{p_name}.{pn}"
            if key not in arg_map:
                unmatched_pred_args_global.append(f"P::{key}")

    # 构造整体 arg-level 的 pred/gt 用于 F1
    matched_gt_args_global = list(arg_map.values())
    pred_args_global = matched_gt_args_global + unmatched_pred_args_global

    args_f1 = _f1(pred_args_global, gt_args_global)

    # -------------------------
    # 3) Combine
    # -------------------------
    schema_f1 = (tool_f1 + args_f1) / 2 if (tool_f1 > 0 or args_f1 > 0) else 0.0
    schema_debug = {
        "tool_matches": tool_match_details,
        "arg_matches": arg_match_details,
        "tool_f1": tool_f1,
        "args_f1": args_f1,
        "schema_f1": schema_f1,
    }
    if ENABLE_DEBUG_PRINTS:
        print(f"Tool F1: {tool_f1:.2f}, Args F1: {args_f1:.2f}, Schema F1: {schema_f1:.2f}")
    return tool_map, arg_map, schema_f1, schema_debug


def _extract_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        s = obj.strip()
        if s.startswith("{") or s.startswith("["):
            try:
                return _extract_text(json.loads(s))
            except Exception:
                return s
        return s
    if isinstance(obj, dict):
        for k in ("text", "content", "message", "output", "result"):
            if k in obj:
                t = _extract_text(obj.get(k))
                if t:
                    return t
        try:
            return json.dumps(obj, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(obj)
    if isinstance(obj, list):
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return " ".join(_extract_text(x) for x in obj if x is not None)
    return str(obj)

def _safe_call_mcp_tool(
    tk: MCPServerToolsToolkit,
    server_name: str,
    tool_name: str,
    arguments: Dict[str, Any],
    timeout: float,
) -> Dict[str, Any]:
    ok, active, details = _ensure_server_active(tk, server_name)
    if not ok:
        return {
            "success": False,
            "error": f"Server '{server_name}' not found in active servers: {active}",
            "server_name": server_name,
            "tool_name": tool_name,
            "recovery": details,
        }
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        tk.bridge.call_mcp_tool,
        server_name,
        tool_name,
        arguments,
        timeout,
    )
    try:
        return future.result(timeout=timeout + 2.0)
    except FuturesTimeoutError:
        executor.shutdown(wait=False, cancel_futures=True)
        return {
            "success": False,
            "error": f"Timeout calling tool after {timeout}s",
            "server_name": server_name,
            "tool_name": tool_name,
        }
    except Exception as e:
        executor.shutdown(wait=False, cancel_futures=True)
        return {
            "success": False,
            "error": f"Exception calling tool: {str(e)}",
            "server_name": server_name,
            "tool_name": tool_name,
        }
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _ensure_server_active(tk: MCPServerToolsToolkit, server_name: str) -> Tuple[bool, List[str], Dict[str, Any]]:
    active = tk.bridge.get_active_servers()
    if server_name in active:
        return True, active, {}
    restart_res = tk.bridge.restart_mcp_server(server_name)
    active = tk.bridge.get_active_servers()
    if server_name in active:
        return True, active, {"restart": restart_res}
    create_res = tk.bridge.create_mcp_server(
        server_names=server_name,
        init_states=tk.init_states,
        registry_path=tk.registry_path,
        fail_fast=True,
    )
    active = tk.bridge.get_active_servers()
    if server_name in active:
        return True, active, {"restart": restart_res, "create": create_res}
    return False, active, {"restart": restart_res, "create": create_res}

def _tool_call_with_unit_tests(tk: MCPServerToolsToolkit, tool_map: Dict[str, str], arg_map: Dict[str, str], server_name: str, gt_schema: Dict[str, Any], stateful: bool) -> Tuple[float, float, List[Dict[str, Any]]]:
    ut = gt_schema.get("unit_test") or {}

    LAMBDA_STRUCT = 0.5  # Paper §A.3: equal weight struct + embed

    def _extract_json_paths(obj: Any, prefix: str = "") -> set:
        """Recursively extract key-path strings from a JSON object."""
        paths = set()
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{prefix}.{k}" if prefix else k
                paths.add(p)
                paths.update(_extract_json_paths(v, p))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                p = f"{prefix}[{i}]"
                paths.add(p)
                paths.update(_extract_json_paths(v, p))
        return paths

    def _structural_judge(actual: str, expected: str) -> Tuple[bool, float]:
        """JSON key-path overlap F1 (paper §A.3)."""
        a_str = (actual or "").strip()
        e_str = (expected or "").strip()

        # Hard gate: unexpected error tokens
        a_low = a_str.lower()
        e_low = e_str.lower()
        fail_tokens = ("error", "failed", "exception", "traceback")
        if any(t in a_low for t in fail_tokens) and not any(t in e_low for t in fail_tokens):
            return False, 0.0

        # Try to parse both as JSON for key-path F1
        a_obj, e_obj = None, None
        try:
            a_obj = json.loads(a_str)
        except Exception:
            pass
        try:
            e_obj = json.loads(e_str)
        except Exception:
            pass

        if a_obj is not None and e_obj is not None:
            a_paths = _extract_json_paths(a_obj)
            e_paths = _extract_json_paths(e_obj)
            if not a_paths and not e_paths:
                return True, 1.0
            if not a_paths or not e_paths:
                return True, 0.0
            tp = len(a_paths & e_paths)
            prec = tp / len(a_paths) if a_paths else 0.0
            rec = tp / len(e_paths) if e_paths else 0.0
            f1 = 2 * prec * rec / (prec + rec + 1e-12)
            return True, f1

        # Fallback for non-JSON: simple text equality check
        if a_low == e_low:
            return True, 1.0
        return True, 0.0

    inv_tool_map: Dict[str, str] = {g: p for p, g in tool_map.items()}
    inv_arg_map: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for pk, gv in arg_map.items():
        try:
            p_tool, p_arg = pk.split(".", 1)
            g_tool, g_arg = gv.split(".", 1)
            inv_arg_map[(g_tool, g_arg)] = (p_tool, p_arg)
        except Exception:
            continue

    total = 0
    hard_ok = 0
    soft_sum = 0.0
    details: List[Dict[str, Any]] = []

    server_dead = False
    for g_tool, cases in ut.items():
        if not isinstance(cases, list):
            continue
        p_tool = inv_tool_map.get(g_tool)
        if not p_tool:
            continue
        for case in cases:
            if not isinstance(case, dict):
                continue
            g_args = case.get("arguments") or {}
            expected_raw = case.get("function_output_content")
            expected = _extract_text(expected_raw)

            if stateful:
                tk.bridge.restart_mcp_server(server_name.strip())

            pred_args: Dict[str, Any] = {}
            if isinstance(g_args, dict):
                for ga_k, ga_v in g_args.items():
                    pair = inv_arg_map.get((g_tool, ga_k))
                    if pair and pair[0] == p_tool:
                        pred_args[pair[1]] = ga_v

            total += 1
            res = _safe_call_mcp_tool(tk, server_name.strip(), p_tool, pred_args, timeout=30.0)
            if not isinstance(res, dict) or res.get("success") is False:
                err = res.get("error") if isinstance(res, dict) else "Unknown error"
                details.append({
                    "gt_tool": g_tool, "pred_tool": p_tool, "args": pred_args,
                    "expected": expected,
                    "status": "err", "error": str(err),
                    "hard_pass": False, "struct_score": 0.0, "embed_score": 0.0,
                    "matched": False, "final_score": 0.0,
                })
                if "not found in active servers" in str(err) or "not running" in str(err):
                    server_dead = True
                if server_dead:
                    break
                continue
            text = _extract_text(res)
            hard_pass, struct_score = _structural_judge(text, expected)
            hard_ok += int(hard_pass)
            matched, sim = (False, 0.0)
            if hard_pass:
                matched, sim = _similar_score(text, expected)
            final_score = (LAMBDA_STRUCT * float(struct_score) + (1.0 - LAMBDA_STRUCT) * float(sim)) if hard_pass else 0.0
            soft_sum += float(final_score)
            details.append({
                "gt_tool": g_tool, "pred_tool": p_tool, "args": pred_args,
                "expected": expected, "actual": text,
                "hard_pass": hard_pass,
                "struct_score": float(struct_score),
                "embed_score": float(sim),
                "matched": matched,
                "final_score": float(final_score),
            })
        if server_dead:
            break

    soft_avg = soft_sum / total if total > 0 else 0.0
    hard_rate = hard_ok / total if total > 0 else 0.0
    return soft_avg, hard_rate, details


# def _trajectory_call_with_logs(tk: MCPServerToolsToolkit, tool_map: Dict[str, str], arg_map: Dict[str, str], server_name: str, gt_schema: Dict[str, Any], stateful: bool) -> Tuple[float, float, List[Dict[str, Any]]]:
#     logs = gt_schema.get("execution_logs") or []

#     inv_tool_map: Dict[str, str] = {g: p for p, g in tool_map.items()}
#     inv_arg_map: Dict[Tuple[str, str], Tuple[str, str]] = {}
#     for pk, gv in arg_map.items():
#         try:
#             p_tool, p_arg = pk.split(".", 1)
#             g_tool, g_arg = gv.split(".", 1)
#             inv_arg_map[(g_tool, g_arg)] = (p_tool, p_arg)
#         except Exception:
#             continue

#     total = 0
#     hard_ok = 0
#     soft_sum = 0.0
#     details: List[Dict[str, Any]] = []

#     def _as_list(x: Any) -> List[Dict[str, Any]]:
#         out: List[Dict[str, Any]] = []
#         if isinstance(x, list):
#             for y in x:
#                 if isinstance(y, dict) and y.get("function_name"):
#                     out.append(y)
#         return out

#     for traj in logs:
#         if stateful:
#             tk.bridge.restart_mcp_server(server_name.strip())

#         steps = _as_list(traj)
#         for rec in steps:
#             g_tool = rec.get("function_name")
#             g_args = rec.get("arguments") or {}
#             expected = _extract_text(rec.get("function_output_content"))
#             if not isinstance(g_tool, str):
#                 continue
#             p_tool = inv_tool_map.get(g_tool)
#             if not p_tool:
#                 continue
#             pred_args: Dict[str, Any] = {}
#             if isinstance(g_args, dict):
#                 for ga_k, ga_v in g_args.items():
#                     pair = inv_arg_map.get((g_tool, ga_k))
#                     if pair and pair[0] == p_tool:
#                         pred_args[pair[1]] = ga_v
#             total += 1
#             try:
#                 res = tk.bridge.call_mcp_tool(server_name.strip(), p_tool, pred_args)
#                 text = _extract_text(res)
#                 matched, sim = _similar_score(text, expected)
#                 soft_sum += float(sim)
#                 hard_ok += int(matched)
#                 details.append({"gt_tool": g_tool, "pred_tool": p_tool, "args": pred_args, "expected": expected, "actual": text, "matched": matched, "sim_score": sim})
#             except Exception as e:
#                 details.append({"gt_tool": g_tool, "pred_tool": p_tool, "args": pred_args, "expected": expected, "status": "err", "error": str(e), "sim_score": 0.0})

#     soft_avg = soft_sum / total if total > 0 else 0.0
#     hard_rate = hard_ok / total if total > 0 else 0.0
#     return soft_avg, hard_rate, details

def _load_server_stateful(gt_root: Optional[str], server_name: str) -> bool:
    try:
        if not gt_root:
            return False
        p = Path(gt_root) / server_name / "server_state_classification.json"
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        v = data.get("stateful")
        if isinstance(v, bool):
            return v
        cls = data.get("server_class") or data.get("class")
        if isinstance(cls, str):
            return cls.strip().lower() == "stateful"
    except Exception:
        return False
    return False

def l2_semantic_correctness_metrics(
    l1_success: bool,
    schema: Dict[str, Any],
    server_name: str,
    registry_path: str,
    gt_schema: Dict[str, Any],
    gt_path: Optional[str] = None,
    schema_path: Optional[str] = None,
    trajectory_root: Optional[Path] = None,
) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, Any]]:
    
    logger.info("Computing L2 Semantic Correctness Metrics")
    
    if not l1_success:
        return {},{
            "schema_f1": 0.0,
            "tool_call_success_rate_soft": 0.0,
            "tool_call_success_rate_hard": 0.0,
            "trajectory_level_validation_rate_soft": 0.0,
            "trajectory_level_validation_rate_hard": 0.0,
        },{
            "schema": {},
            "unit_tests": {"soft_avg": 0.0, "hard_rate": 0.0, "details": []},
            "trajectory": {"soft_avg": 0.0, "hard_rate": 0.0, "details": []},
        }

    tool_map, arg_map, schema_f1, schema_debug = _schema_fidelity(schema, gt_schema)
    if ENABLE_DEBUG_PRINTS:
        print(tool_map)
        print(arg_map)

    tk = MCPServerToolsToolkit(server_names=server_name.strip(), registry_path=registry_path)
    tk.initialize_servers()
    ok, active_servers, recovery = _ensure_server_active(tk, server_name.strip())
    if not ok:
        tk.cleanup()
        return tool_map, {
            "schema_f1": schema_f1,
            "tool_call_success_rate_soft": 0.0,
            "tool_call_success_rate_hard": 0.0,
            "trajectory_level_validation_rate_soft": 0.0,
            "trajectory_level_validation_rate_hard": 0.0,
        },{
            "schema": schema_debug,
            "unit_tests": {"soft_avg": 0.0, "hard_rate": 0.0, "details": []},
            "trajectory": {"soft_avg": 0.0, "hard_rate": 0.0, "details": []},
            "error": f"Server '{server_name.strip()}' not found in active servers: {active_servers}",
            "recovery": recovery,
        }

    example_tasks = gt_schema.get("task_example") or []
    tasks: List[str] = []
    for t in example_tasks:
        if isinstance(t, str):
            tasks.append(t)

    execute_task = ExecuteTask(
        tasks=tasks,
        server_name=server_name.strip(),
        toolkit=tk,
        schema_path=schema_path,
        agent_result_path=trajectory_root,
    )
    
    stateful = _load_server_stateful(gt_path, server_name.strip())
    unit_soft, unit_hard, _u = _tool_call_with_unit_tests(tk, tool_map, arg_map, server_name, gt_schema, stateful)
    if ENABLE_DEBUG_PRINTS:
        print(f"unit_soft: {unit_soft}, unit_hard: {unit_hard}")
    traj_score, _t = execute_task.get_execute_task_result()
    traj_soft = float(traj_score)  # avg(score/5), continuous [0,1]
    # traj_hard: fraction of tasks with score >= 3 (solved)
    traj_details = _t.get("details", [])
    traj_hard_count = sum(1 for d in traj_details if d.get("solved"))
    traj_hard = traj_hard_count / len(traj_details) if traj_details else 0.0
    
    tk.cleanup()
    return tool_map, {
        "schema_f1": schema_f1,
        "tool_call_success_rate_soft": unit_soft,
        "tool_call_success_rate_hard": unit_hard,
        "trajectory_level_validation_rate_soft": traj_soft,
        "trajectory_level_validation_rate_hard": traj_hard,
    },{
        "schema": schema_debug,
        "unit_tests": {"soft_avg": unit_soft, "hard_rate": unit_hard, "details": _u},
        "trajectory": {"soft_avg": traj_soft, "hard_rate": traj_hard, "details": _t},
    }

def _similar_score(a: str, b: str) -> Tuple[bool, float]:
    sa = (a or "").strip().lower()
    sb = (b or "").strip().lower()
    if not sa and not sb:
        return True, 1.0
    if sa == sb:
        return True, 1.0
    emb = call_embedding([sa, sb])
    if not isinstance(emb, list) or len(emb) < 2:
        return False, 0.0
    ea, eb = emb[0], emb[1]
    na = math.sqrt(sum(x * x for x in ea))
    nb = math.sqrt(sum(y * y for y in eb))
    if na == 0 or nb == 0:
        return False, 0.0
    sim = sum((x / na) * (y / nb) for x, y in zip(ea, eb))
    return sim >= HARD_MATCH_THRESHOLD, sim
