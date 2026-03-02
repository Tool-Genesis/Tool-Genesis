"""
Evolution Feedback Collector for Experiment B.

Collects task-level feedback from existing L4 trajectory evaluations,
formatted as a structured feedback report that can be fed back to the model
for iterative improvement.

Only uses train_tasks — test_tasks are never exposed to the model.
"""

import json
from typing import Any, Dict, List, Optional


def collect_feedback(
    l2_debug: dict,
    train_task_indices: List[int],
    task_descriptions: List[str],
    round_num: int = 0,
) -> Dict[str, Any]:
    """
    Collect feedback from trajectory evaluation on train tasks.

    Args:
        l2_debug: The l2_debug.json data containing trajectory results.
        train_task_indices: Indices of train tasks.
        task_descriptions: Full list of task descriptions (task_example).
        round_num: Current evolution round number.

    Returns:
        Structured feedback report dict.
    """
    traj = l2_debug.get("trajectory", {})
    details_obj = traj.get("details", {})
    if isinstance(details_obj, dict):
        detail_list = details_obj.get("details", [])
    elif isinstance(details_obj, list):
        detail_list = details_obj
    else:
        detail_list = []

    passed = 0
    failed = 0
    failure_summary: List[Dict[str, str]] = []

    for idx in train_task_indices:
        if idx >= len(detail_list):
            continue

        det = detail_list[idx]
        solved = det.get("solved", False)
        task_desc = task_descriptions[idx] if idx < len(task_descriptions) else f"Task #{idx}"

        if solved:
            passed += 1
        else:
            failed += 1
            # Classify the error type from the judge output
            judge_output = det.get("judge_output", "")
            response = det.get("response", "")
            error_type, error_detail = _classify_failure(judge_output, response)

            failure_summary.append({
                "task_index": idx,
                "task_description": task_desc[:200],
                "error_type": error_type,
                "error_detail": error_detail[:300],
            })

    total = passed + failed
    return {
        "round": round_num,
        "total_tasks": total,
        "passed": passed,
        "failed": failed,
        "success_rate": round(passed / total, 4) if total > 0 else 0.0,
        "failure_summary": failure_summary,
    }


def collect_ut_feedback(
    l2_debug: dict,
    ut_train_indices: Dict[str, List[int]],
) -> Dict[str, Any]:
    """
    Collect feedback from unit test results on train split.

    Returns summary of UT failures that can help the model fix bugs.
    """
    ut = l2_debug.get("unit_tests", {})
    ut_details = ut.get("details", [])

    if not ut_details:
        return {"total_cases": 0, "passed": 0, "failed": 0, "failures": []}

    # Build tool groups from ut_details
    tool_groups = []
    current_tool = None
    current_start = 0
    for i, det in enumerate(ut_details):
        gt_tool = det.get("gt_tool", "")
        if gt_tool != current_tool:
            if current_tool is not None:
                tool_groups.append((current_tool, current_start, i))
            current_tool = gt_tool
            current_start = i
    if current_tool is not None:
        tool_groups.append((current_tool, current_start, len(ut_details)))

    passed = 0
    failed = 0
    failures = []

    for tool_name, start, end in tool_groups:
        indices = ut_train_indices.get(tool_name, [])
        for case_idx in indices:
            flat_idx = start + case_idx
            if flat_idx >= end:
                continue

            det = ut_details[flat_idx]
            if det.get("hard_pass"):
                passed += 1
            else:
                failed += 1
                failures.append({
                    "tool_name": tool_name,
                    "pred_tool": det.get("pred_tool", ""),
                    "args": str(det.get("args", ""))[:200],
                    "expected_snippet": str(det.get("expected", ""))[:150],
                    "actual_snippet": str(det.get("actual", ""))[:150],
                    "status": det.get("status", "wrong_output"),
                    "error": str(det.get("error", ""))[:200] if det.get("error") else None,
                })

    return {
        "total_cases": passed + failed,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / (passed + failed), 4) if (passed + failed) > 0 else 0.0,
        "failures": failures[:20],  # Cap at 20 to avoid overwhelming the prompt
    }


def format_feedback_text(
    task_feedback: Dict[str, Any],
    ut_feedback: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Format feedback as human-readable text for the evolution prompt.
    """
    lines = []

    # Task feedback
    tf = task_feedback
    lines.append(f"## Task Execution Feedback (Round {tf['round']})")
    lines.append(f"- {tf['passed']}/{tf['total_tasks']} tasks succeeded "
                 f"(success rate: {tf['success_rate']:.1%})")
    lines.append("")

    if tf["failure_summary"]:
        lines.append(f"### Failed Tasks ({tf['failed']} total):")
        for i, f_item in enumerate(tf["failure_summary"][:10], 1):
            lines.append(f"\n**Failure {i}:**")
            lines.append(f"- Task: {f_item['task_description']}")
            lines.append(f"- Error type: {f_item['error_type']}")
            lines.append(f"- Detail: {f_item['error_detail']}")

        if len(tf["failure_summary"]) > 10:
            lines.append(f"\n... and {len(tf['failure_summary']) - 10} more failures")

    # UT feedback
    if ut_feedback and ut_feedback.get("total_cases", 0) > 0:
        uf = ut_feedback
        lines.append(f"\n## Unit Test Feedback")
        lines.append(f"- {uf['passed']}/{uf['total_cases']} test cases passed "
                     f"(pass rate: {uf['pass_rate']:.1%})")

        if uf["failures"]:
            lines.append(f"\n### Failed Test Cases ({uf['failed']} total):")
            for i, f_item in enumerate(uf["failures"][:5], 1):
                lines.append(f"\n**UT Failure {i}:**")
                lines.append(f"- Tool: {f_item['tool_name']} → {f_item['pred_tool']}")
                lines.append(f"- Args: {f_item['args']}")
                if f_item.get("error"):
                    lines.append(f"- Error: {f_item['error']}")
                else:
                    lines.append(f"- Expected (snippet): {f_item['expected_snippet']}")
                    lines.append(f"- Actual (snippet): {f_item['actual_snippet']}")

            if len(uf["failures"]) > 5:
                lines.append(f"\n... and {len(uf['failures']) - 5} more UT failures")

    return "\n".join(lines)


def _classify_failure(judge_output: str, response: str) -> tuple:
    """
    Rule-based classification of task failure from judge output.

    Returns (error_type, error_detail).
    """
    judge_lower = judge_output.lower() if judge_output else ""
    resp_lower = response.lower() if response else ""

    # Check for tool call errors
    if "error" in resp_lower and ("tool" in resp_lower or "function" in resp_lower):
        return "tool_call_error", _extract_error_snippet(response)

    if "not found" in resp_lower or "no tool" in resp_lower:
        return "tool_not_found", _extract_error_snippet(response)

    if "timeout" in resp_lower or "timed out" in resp_lower:
        return "timeout", "Tool execution timed out"

    if "parameter" in resp_lower and "missing" in resp_lower:
        return "missing_parameter", _extract_error_snippet(response)

    if "format" in judge_lower and ("wrong" in judge_lower or "incorrect" in judge_lower):
        return "wrong_output_format", _extract_error_snippet(judge_output)

    if "incomplete" in judge_lower or "partial" in judge_lower:
        return "incomplete_result", _extract_error_snippet(judge_output)

    if judge_output:
        return "task_logic_failure", judge_output[:200]

    return "unknown", response[:200] if response else "No details available"


def _extract_error_snippet(text: str) -> str:
    """Extract a relevant error snippet from text."""
    lines = text.split("\n")
    for line in lines:
        lower = line.lower()
        if "error" in lower or "fail" in lower or "exception" in lower:
            return line.strip()[:300]
    return text[:200].strip()
