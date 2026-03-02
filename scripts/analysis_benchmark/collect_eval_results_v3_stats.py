import argparse
import json
import math
import os
import importlib.util
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def _load_json(path: Path) -> Optional[Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _as_list(x: Any) -> List[Dict[str, Any]]:
    if isinstance(x, list):
        return [it for it in x if isinstance(it, dict)]
    return []


def _num(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _mean(vals: List[float]) -> float:
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    mx = sum(xs[:n]) / n
    my = sum(ys[:n]) / n
    num = 0.0
    dx = 0.0
    dy = 0.0
    for i in range(n):
        vx = xs[i] - mx
        vy = ys[i] - my
        num += vx * vy
        dx += vx * vx
        dy += vy * vy
    den = math.sqrt(dx * dy)
    if den == 0:
        return 0.0
    return num / den


def _parse_strategy_model(dirname: str) -> Tuple[str, str]:
    if dirname.startswith("direct_"):
        return "direct", dirname[len("direct_") :]
    if dirname.startswith("coder_agent_"):
        return "coder_agent", dirname[len("coder_agent_") :]
    parts = dirname.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "unknown", dirname


def _collect_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    schema_vals: List[float] = []
    tool_soft_vals: List[float] = []
    tool_hard_vals: List[float] = []
    traj_soft_vals: List[float] = []
    for r in rows:
        m = r.get("metrics") or {}
        schema_f1 = _num(m.get("schema_f1"))
        schema_vals.append(schema_f1)
        tool_soft_vals.append(_num(m.get("tool_call_success_rate_soft")))
        tool_hard_vals.append(_num(m.get("tool_call_success_rate_hard")))
        traj_soft_vals.append(_num(m.get("trajectory_level_validation_rate_soft")))
    precision_schema = _soft_precision(schema_vals, traj_soft_vals)
    precision_soft = _soft_precision(tool_soft_vals, traj_soft_vals)
    precision_hard = _soft_precision(tool_hard_vals, traj_soft_vals)
    return {
        "schema_f1": _mean(schema_vals),
        "tool_call_success_rate_soft": _mean(tool_soft_vals),
        "tool_call_success_rate_hard": _mean(tool_hard_vals),
        "precision_schema": precision_schema,
        "precision_tool_soft": precision_soft,
        "precision_tool_hard": precision_hard,
        "count": float(len(rows)),
    }


def _soft_precision(signal_vals: List[float], sr_vals: List[float]) -> float:
    n = min(len(signal_vals), len(sr_vals))
    if n == 0:
        return 0.0
    signal_sum = 0.0
    true_sum = 0.0
    zero_count = 0
    for i in range(n):
        signal = signal_vals[i]
        if signal <= 0:
            zero_count += 1
            continue
        signal_sum += signal
        true_sum += signal * sr_vals[i]
    denom = signal_sum + zero_count
    if denom == 0:
        return 0.0
    return true_sum / denom

def _cumulative_pass(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(rows)
    if n == 0:
        return {"S0": 1.0, "S1": 0.0, "S2": 0.0, "S3": 0.0, "S4": 0.0, "count": 0.0}
    s0 = 1.0
    s1_acc = 0.0
    s2_acc = 0.0
    s3_acc = 0.0
    s4_acc = 0.0
    for r in rows:
        m = r.get("metrics") or {}
        p1 = _num(m.get("schema_f1"))
        p2 = _num(m.get("tool_call_success_rate_soft"))
        p3 = _num(m.get("trajectory_level_validation_rate_soft"))
        p4 = _num(m.get("capability_boundary_score"))
        s1_acc += p1
        s2_acc += p1 * p2
        s3_acc += p1 * p2 * p3
        s4_acc += p1 * p2 * p3 * p4
    inv = 1.0 / float(n)
    return {
        "S0": s0,
        "S1": s1_acc * inv,
        "S2": s2_acc * inv,
        "S3": s3_acc * inv,
        "S4": s4_acc * inv,
        "count": float(n),
    }


def _safe_var_name(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out)
    if not s:
        return "unknown"
    if s[0].isdigit():
        return f"_{s}"
    return s


def _format_list(vals: List[Optional[float]]) -> str:
    parts = []
    for v in vals:
        if v is None:
            parts.append("None")
        else:
            parts.append(f"{v:.4f}")
    return "[" + ", ".join(parts) + "]"

def _iter_projects(run_root: Path) -> List[Path]:
    if not run_root.exists() or not run_root.is_dir():
        return []
    return [p for p in sorted(run_root.iterdir()) if p.is_dir()]


def _iter_servers(project_dir: Path) -> List[Path]:
    return [p for p in sorted(project_dir.iterdir()) if p.is_dir()]


def _as_messages(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [it for it in data if isinstance(it, dict)]
    if isinstance(data, dict):
        msgs = data.get("messages")
        if isinstance(msgs, list):
            return [it for it in msgs if isinstance(it, dict)]
        conv = data.get("conversation")
        if isinstance(conv, list):
            return [it for it in conv if isinstance(it, dict)]
    return []


def _extract_first_user_content(messages: List[Dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            return json.dumps(content, ensure_ascii=False)
        return str(content or "")
    return ""


def _extract_last_assistant_content(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            return json.dumps(content, ensure_ascii=False)
        return str(content or "")
    return ""


def _collect_sr_by_project(eval_root: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for entry in sorted(eval_root.iterdir()):
        if not entry.is_dir():
            continue
        res_path = entry / "results.json"
        if not res_path.exists():
            continue
        raw = _load_json(res_path)
        rows = _as_list(raw)
        if not rows:
            continue
        project_sr: Dict[str, float] = {}
        for r in rows:
            slug = str(r.get("server_slug") or "")
            if not slug:
                continue
            m = r.get("metrics") or {}
            sr = _num(m.get("trajectory_level_validation_rate_soft"))
            project_sr[slug] = sr
        out[entry.name] = project_sr
    return out


def _find_conversation_json(server_dir: Path) -> Optional[Path]:
    matches: List[Path] = []
    for root, _, files in os.walk(server_dir):
        for fn in files:
            if fn == "conversation.json":
                matches.append(Path(root) / fn)
    if not matches:
        return None
    return sorted(matches)[-1]


def _find_phase_conversation_json(server_dir: Path, phase: str) -> Optional[Path]:
    matches = sorted(server_dir.glob(f"{phase}/**/logs/conversation.json"))
    if not matches:
        return None
    return matches[-1]


def _load_conversation_messages(path: Path) -> List[Dict[str, Any]]:
    data = _load_json(path)
    return _as_messages(data)


def _extract_code_agent_prompts(run_root: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    for project_dir in _iter_projects(run_root):
        strategy, model = _parse_strategy_model(project_dir.name)
        if strategy != "coder_agent":
            continue
        for server_dir in _iter_servers(project_dir):
            conv_path = _find_conversation_json(server_dir)
            if conv_path is None:
                continue
            messages = _load_conversation_messages(conv_path)
            user_text = _extract_first_user_content(messages)
            assistant_text = _extract_last_assistant_content(messages)
            out[(model, server_dir.name)] = {
                "user": user_text,
                "assistant": assistant_text,
            }
    return out


def _extract_prompt_completion(path: Optional[Path]) -> Tuple[int, int]:
    if path is None or not path.exists():
        return 0, 0
    data = _load_json(path)
    prompt_tokens = 0
    completion_tokens = 0
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            if "prompt_tokens" in item or "completion_tokens" in item:
                prompt_tokens = int(item.get("prompt_tokens") or 0)
                completion_tokens = int(item.get("completion_tokens") or 0)
            usage = item.get("usage")
            if isinstance(usage, dict) and ("prompt_tokens" in usage or "completion_tokens" in usage):
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                completion_tokens = int(usage.get("completion_tokens") or 0)
    elif isinstance(data, dict):
        if "prompt_tokens" in data or "completion_tokens" in data:
            prompt_tokens = int(data.get("prompt_tokens") or 0)
            completion_tokens = int(data.get("completion_tokens") or 0)
        usage = data.get("usage")
        if isinstance(usage, dict) and ("prompt_tokens" in usage or "completion_tokens" in usage):
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
    return prompt_tokens, completion_tokens


def _collect_code_agent_usage(run_root: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for project_dir in _iter_projects(run_root):
        strategy, model = _parse_strategy_model(project_dir.name)
        if strategy != "coder_agent":
            continue
        for server_dir in _iter_servers(project_dir):
            toolgen_path = _find_phase_conversation_json(server_dir, "toolgen")
            codegen_path = _find_phase_conversation_json(server_dir, "codegen")
            toolgen_prompt, toolgen_completion = _extract_prompt_completion(toolgen_path)
            codegen_prompt, codegen_completion = _extract_prompt_completion(codegen_path)
            user_text = ""
            if toolgen_path is not None:
                messages = _load_conversation_messages(toolgen_path)
                user_text = _extract_first_user_content(messages)
            out[(model, server_dir.name)] = {
                "toolgen_tokens": toolgen_prompt + toolgen_completion,
                "codegen_tokens": codegen_prompt + codegen_completion,
                "user": user_text,
            }
    return out


def _count_tokens(text: str, model: str) -> int:
    if not text:
        return 0
    fn = _load_count_tokens()
    if fn is None:
        return 0
    try:
        return int(fn(text, model))
    except Exception:
        return 0


_COUNT_TOKENS_FUNC: Optional[Callable[[str, str], int]] = None


def _load_count_tokens() -> Optional[Callable[[str, str], int]]:
    global _COUNT_TOKENS_FUNC
    if _COUNT_TOKENS_FUNC is not None:
        return _COUNT_TOKENS_FUNC
    module_path = Path(__file__).resolve().parents[2] / "src" / "utils" / "count_tokens.py"
    if not module_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("count_tokens_module", module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if "tiktoken" not in module.__dict__:
        try:
            import tiktoken
            module.__dict__["tiktoken"] = tiktoken
        except Exception:
            class _FallbackTokenizer:
                def encode(self, text: str) -> List[str]:
                    return (text or "").split()

            class _FallbackTikToken:
                def get_encoding(self, _name: str) -> "_FallbackTokenizer":
                    return _FallbackTokenizer()

                def encoding_for_model(self, _model: str) -> "_FallbackTokenizer":
                    return _FallbackTokenizer()

            module.__dict__["tiktoken"] = _FallbackTikToken()
    fn = getattr(module, "count_tokens", None)
    if not callable(fn):
        return None
    _COUNT_TOKENS_FUNC = fn
    return fn


def _emit_token_sr_csv(
    *,
    run_root: Path,
    eval_root: Path,
    out_path: Path,
) -> None:
    sr_by_project = _collect_sr_by_project(eval_root)
    code_agent_usage = _collect_code_agent_usage(run_root)
    rows: List[Dict[str, Any]] = []
    for project_dir in _iter_projects(run_root):
        strategy, model = _parse_strategy_model(project_dir.name)
        paradigm = "code_agent" if strategy == "coder_agent" else strategy
        project_sr = sr_by_project.get(project_dir.name, {})
        for server_dir in _iter_servers(project_dir):
            server_slug = server_dir.name
            sr_soft = project_sr.get(server_slug, 0.0)
            tokens_total = 0
            if paradigm == "code_agent":
                usage = code_agent_usage.get((model, server_slug), {})
                tokens_total = int(usage.get("toolgen_tokens", 0)) + int(usage.get("codegen_tokens", 0))
            elif paradigm == "direct":
                usage = code_agent_usage.get((model, server_slug), {})
                user_text = str(usage.get("user", "") or "")
                env_code_path = server_dir / "env_code.py"
                output_text = ""
                if env_code_path.exists():
                    try:
                        output_text = env_code_path.read_text(encoding="utf-8")
                    except Exception:
                        output_text = ""
                tokens_total = int(usage.get("toolgen_tokens", 0))
                tokens_total += _count_tokens(user_text, model) + _count_tokens(output_text, model)
            else:
                continue
            payload = {
                "server_id": server_slug,
                "model": model,
                "paradigm": paradigm,
                "tokens_total": tokens_total,
                "tokens_log10": math.log10(tokens_total) if tokens_total > 0 else 0.0,
                "sr_soft": sr_soft,
            }
            rows.append(payload)
    if not rows:
        print("No token/sr data produced")
        return
    try:
        import pandas as pd  # type: ignore
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
    except Exception:
        import csv
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["server_id", "model", "paradigm", "tokens_total", "tokens_log10", "sr_soft"])
            for r in rows:
                w.writerow([r["server_id"], r["model"], r["paradigm"], r["tokens_total"], r["tokens_log10"], r["sr_soft"]])
    print(f"[TokenSR] saved: {out_path}")


def _emit_python(models: List[str], data_by_strategy: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    print("models = [")
    for m in models:
        print(f"    {m!r},")
    print("]")
    print()
    ordered = ["direct", "coder_agent"]
    extra = [k for k in sorted(data_by_strategy.keys()) if k not in ordered]
    for strategy in ordered + extra:
        if strategy not in data_by_strategy:
            continue
        strat_data = data_by_strategy[strategy]
        corr_schema_vals: List[float] = []
        corr_soft_vals: List[float] = []
        corr_hard_vals: List[float] = []
        for m in models:
            rec = strat_data.get(m)
            if rec is None:
                continue
            corr_schema_vals.append(rec.get("precision_schema", 0.0))
            corr_soft_vals.append(rec.get("precision_tool_soft", 0.0))
            corr_hard_vals.append(rec.get("precision_tool_hard", 0.0))
        var_name = f"data_{_safe_var_name(strategy)}"
        print(f"{var_name} = {{")
        print(f"    'Precision(Schema)': {_format_list(corr_schema_vals)},")
        print(f"    'Precision(FVRSoft)': {_format_list(corr_soft_vals)},")
        print(f"    'Precision(FVRHard)': {_format_list(corr_hard_vals)},")
        print("}")
        print()


def _emit_json(models: List[str], data_by_strategy: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    ordered = ["direct", "coder_agent"]
    extra = [k for k in sorted(data_by_strategy.keys()) if k not in ordered]
    out: Dict[str, Any] = {"models": models, "strategies": {}}
    for strategy in ordered + extra:
        if strategy not in data_by_strategy:
            continue
        strat_data = data_by_strategy[strategy]
        corr_schema_vals: List[float] = []
        corr_soft_vals: List[float] = []
        corr_hard_vals: List[float] = []
        for m in models:
            rec = strat_data.get(m)
            if rec is None:
                continue
            corr_schema_vals.append(rec.get("precision_schema", 0.0))
            corr_soft_vals.append(rec.get("precision_tool_soft", 0.0))
            corr_hard_vals.append(rec.get("precision_tool_hard", 0.0))
        out["strategies"][strategy] = {
            "precision_schema": corr_schema_vals,
            "precision_tool_soft": corr_soft_vals,
            "precision_tool_hard": corr_hard_vals,
        }
    print(json.dumps(out, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="temp/eval_results_v3",
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default="temp/run_benchmark_v3",
    )
    parser.add_argument(
        "--output-format",
        choices=["python", "json"],
        default="python",
    )
    parser.add_argument(
        "--emit-token-sr",
        action="store_true",
    )
    parser.add_argument(
        "--token-sr-out",
        type=str,
        default="json_schema.csv",
    )
    parser.add_argument(
        "--save-excel",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        print(f"Not a directory: {root}")
        return
    if args.emit_token_sr:
        run_root = Path(args.run_root)
        if not run_root.exists() or not run_root.is_dir():
            print(f"Not a directory: {run_root}")
            return
        _emit_token_sr_csv(
            run_root=run_root,
            eval_root=root,
            out_path=Path(args.token_sr_out),
        )
    data_by_strategy: Dict[str, Dict[str, Dict[str, float]]] = {}
    models_by_strategy: Dict[str, set] = {}
    funnel_by_strategy: Dict[str, Dict[str, Dict[str, float]]] = {}
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        res_path = entry / "results.json"
        if not res_path.exists():
            continue
        raw = _load_json(res_path)
        rows = _as_list(raw)
        if len(rows) < 10:
            continue
        strategy, model = _parse_strategy_model(entry.name)
        metrics = _collect_metrics(rows)
        data_by_strategy.setdefault(strategy, {})[model] = metrics
        models_by_strategy.setdefault(strategy, set()).add(model)
        funnel = _cumulative_pass(rows)
        funnel_by_strategy.setdefault(strategy, {})[model] = funnel
    if not data_by_strategy:
        print("No results.json found in subdirectories")
        return
    common_models: Optional[set] = None
    for strat_models in models_by_strategy.values():
        common_models = set(strat_models) if common_models is None else common_models & set(strat_models)
    models = sorted(common_models or set(), key=lambda s: str(s).lower())
    if not models:
        print("No common models across strategies")
        return
    if args.output_format == "json":
        _emit_json(models, data_by_strategy)
    else:
        _emit_python(models, data_by_strategy)
    ordered = ["direct", "coder_agent"]
    extra = [k for k in sorted(funnel_by_strategy.keys()) if k not in ordered]
    for strategy in ordered + extra:
        if strategy not in funnel_by_strategy:
            continue
        sd = funnel_by_strategy[strategy]
        print(f"[Funnel] {strategy}")
        for m in models:
            rec = sd.get(m)
            if rec is None:
                continue
            print(f"  {m}: S0={rec['S0']:.4f}, S1={rec['S1']:.4f}, S2={rec['S2']:.4f}, S3={rec['S3']:.4f}, S4={rec['S4']:.4f}, count={int(rec['count'])}")
    excel_path = args.save_excel or str(root / "funnel_cumulative_pass.xlsx")
    rows_out: List[Dict[str, Any]] = []
    for strategy, sd in funnel_by_strategy.items():
        for model, rec in sd.items():
            rows_out.append(
                {
                    "strategy": strategy,
                    "model": model,
                    "S0": rec.get("S0", 0.0),
                    "S1": rec.get("S1", 0.0),
                    "S2": rec.get("S2", 0.0),
                    "S3": rec.get("S3", 0.0),
                    "S4": rec.get("S4", 0.0),
                    "count": int(rec.get("count", 0.0)),
                }
            )
    if rows_out:
        wrote_excel = False
        try:
            import pandas as pd  # type: ignore
            df = pd.DataFrame(rows_out)
            df.to_excel(excel_path, index=False)
            wrote_excel = True
            print(f"[Excel] saved: {excel_path}")
        except Exception:
            csv_path = args.save_excel or str(root / "funnel_cumulative_pass.csv")
            import csv
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["strategy", "model", "S0", "S1", "S2", "S3", "S4", "count"])
                for r in rows_out:
                    w.writerow([r["strategy"], r["model"], r["S0"], r["S1"], r["S2"], r["S3"], r["S4"], r["count"]])
            print(f"[CSV] saved: {csv_path}")


if __name__ == "__main__":
    main()
