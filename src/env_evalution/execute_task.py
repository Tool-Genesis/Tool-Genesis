from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.toolkits.mcp_sandbox_bridge_toolkit import MCPServerToolsToolkit
from .prompt import build_solve_task_prompt, build_judge_task_prompt
from .agentic_framwork import initialize_agent

from loguru import logger

class JudgeResult:
    def __init__(self, score: int, analysis: str) -> None:
        self.score = score
        self.analysis = analysis

    def __str__(self) -> str:
        return f"Score: {self.score}, Analysis: {self.analysis}"

class ExecuteTask:
    def __init__(
        self,
        tasks: List[str],
        server_name: str,
        toolkit: MCPServerToolsToolkit,
        schema_path: Optional[str] = None,
        agent_result_path: Optional[Union[str, Path]] = "temp/agentic",
    ):
        self.tasks = tasks
        self.server_name = server_name
        self.simulation_toolkit = toolkit
        self.schema_path = schema_path
        base_dir = Path(agent_result_path) if agent_result_path is not None else Path("temp/agentic")
        self.agent_result_root = base_dir
        os.makedirs(self.agent_result_root, exist_ok=True)
        self.agents = initialize_agent(self.server_name, self.simulation_toolkit, str(self.agent_result_root))

    def _build_conversation_history(self, agent) -> str:
        try:
            openai_messages, _ = agent.memory.get_context_all()
        except Exception:
            return ""
        try:
            return json.dumps(openai_messages, ensure_ascii=False, indent=2)
        except Exception:
            return "\n".join(str(m) for m in openai_messages)

    def _parse_judge_response(self, xml_text: str) -> "JudgeResult":
        text = xml_text.strip()
        start_idx = text.find("<response")
        end_idx = text.rfind("</response>")
        if start_idx != -1 and end_idx != -1:
            end_idx += len("</response>")
            text = text[start_idx:end_idx]
        if not text:
            return JudgeResult(score=0, analysis="")
        try:
            root = ET.fromstring(text)
        except Exception:
            return JudgeResult(score=0, analysis=text)
        reasoning_elem = root.find(".//reasoning")
        rating_elem = root.find(".//rating")
        if reasoning_elem is not None:
            reasoning = "".join(reasoning_elem.itertext()).strip()
        else:
            reasoning = text
        rating_raw = ""
        if rating_elem is not None and rating_elem.text:
            rating_raw = rating_elem.text.strip()
        rating_lower = rating_raw.lower()
        score_map = {
            "very incomplete": 1,
            "incomplete": 2,
            "partially complete": 3,
            "mostly complete": 4,
            "fully complete": 5,
        }
        score = 0
        if rating_lower.isdigit():
            try:
                score = int(rating_lower)
            except Exception:
                score = 0
        else:
            score = score_map.get(rating_lower, 0)
        if score < 0 or score > 5:
            score = 0
        return JudgeResult(score=score, analysis=reasoning)

    def _judge_task_solved_with_debug(self, agents, task: str, conversation_history: str) -> JudgeResult:
        last_judge: Optional[JudgeResult] = None
        for _ in range(3):
            prompt = build_judge_task_prompt(task, conversation_history)
            try:
                raw_response = agents["judge"].step(prompt)
                content = getattr(raw_response, "content", str(raw_response))
                task_judge = self._parse_judge_response(content)
                last_judge = task_judge
                # score > 0 means the judge returned a meaningful rating (1-5);
                # score == 0 means parse failure, so retry.
                if task_judge.score > 0:
                    return task_judge
            except Exception:
                continue
        if last_judge is not None:
            return last_judge
        return JudgeResult(score=0, analysis="Failed to obtain valid score from judge after 3 attempts.")

    def _simulate_execute_task(self) -> Tuple[List[JudgeResult], List[Dict[str, Any]]]:
        if not self.tasks:
            return [], []

        num_tasks = len(self.tasks)
        tasks_judge: List[Optional[JudgeResult]] = [None] * num_tasks
        details: List[Optional[Dict[str, Any]]] = [None] * num_tasks

        def run_single_task(idx: int, task: str):
            task_dir = self.agent_result_root / f"task_{idx+1}"
            os.makedirs(task_dir, exist_ok=True)

            try:
                agents = initialize_agent(self.server_name, self.simulation_toolkit, str(task_dir))

                prompt = build_solve_task_prompt(task)
                raw_response = agents["simulate_solve"].step(prompt)
                response = getattr(raw_response, "content", str(raw_response))
                conversation_history = self._build_conversation_history(agents["simulate_solve"])
                task_judge = self._judge_task_solved_with_debug(agents, task, conversation_history)
                detail = {
                    "task": task,
                    "response": response,
                    "judge_output": task_judge.analysis,
                    "solved": task_judge.score >= 3,
                }
                return idx, task_judge, detail
            except Exception as exc:
                logger.exception("Task execution failed for task_%s", idx + 1)
                task_judge = JudgeResult(score=0, analysis=f"execution_failed: {exc}")
                detail = {
                    "task": task,
                    "response": "",
                    "judge_output": task_judge.analysis,
                    "solved": False,
                }
                return idx, task_judge, detail

        # max_workers = min(num_tasks, os.cpu_count() or 1)
        max_workers = 8
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            future_to_idx: Dict[Any, int] = {}
            for idx, task in enumerate(self.tasks):
                future = executor.submit(run_single_task, idx, task)
                futures.append(future)
                future_to_idx[future] = idx
            for future in as_completed(futures):
                try:
                    idx, task_judge, detail = future.result()
                    tasks_judge[idx] = task_judge
                    details[idx] = detail
                except Exception as exc:
                    logger.exception("Task future failed")
                    idx = future_to_idx.get(future, 0)
                    task_judge = JudgeResult(score=0, analysis=f"future_failed: {exc}")
                    detail = {
                        "task": self.tasks[idx] if 0 <= idx < len(self.tasks) else "",
                        "response": "",
                        "judge_output": task_judge.analysis,
                        "solved": False,
                    }
                    tasks_judge[idx] = task_judge
                    details[idx] = detail

        return [t for t in tasks_judge if t is not None], [d for d in details if d is not None]

    def get_execute_task_result(self) -> Tuple[float, Dict[str, Any]]:
        if not self.tasks:
            return 0.0, {"details": []}
        simulation_tasks_judge, details = self._simulate_execute_task()
        number_of_tasks = len(self.tasks)
        if number_of_tasks == 0:
            return 0.0, {"details": details}

        # Per-task scores normalised to [0,1]
        task_scores = []
        for t in simulation_tasks_judge:
            s = max(0.0, min(1.0, int(t.score) / 5.0))
            task_scores.append(s)

        # soft_avg: mean normalised score
        soft_avg = sum(task_scores) / number_of_tasks
        # hard_rate: fraction with score >= 3 (solved)
        hard_rate = sum(1 for t in simulation_tasks_judge if t.score >= 3) / number_of_tasks

        # Store per-task normalised score in details for oracle-normalised SR
        for i, d in enumerate(details):
            if i < len(task_scores):
                d["normalised_score"] = task_scores[i]

        return soft_avg, {
            "details": details,
            "soft_avg": soft_avg,
            "hard_rate": hard_rate,
        }

    @staticmethod
    def oracle_normalized_sr(
        gen_details: List[Dict[str, Any]],
        gt_details: List[Dict[str, Any]],
        eps: float = 1e-6,
    ) -> float:
        """Compute oracle-normalized SR per paper §A.4.

        SR_j = (1 - s_gt_j) / (1 - s_gen_j + eps)
        where s_gen_j and s_gt_j are per-task normalised scores in [0,1].

        Returns dataset-level average SR.
        """
        if not gen_details or not gt_details:
            return 0.0
        n = min(len(gen_details), len(gt_details))
        sr_sum = 0.0
        for j in range(n):
            s_gen = gen_details[j].get("normalised_score", 0.0)
            s_gt = gt_details[j].get("normalised_score", 1.0)
            # Clamp to [0,1]
            s_gen = max(0.0, min(1.0, s_gen))
            s_gt = max(0.0, min(1.0, s_gt))
            sr_j = 1.0 - (1.0 - s_gt) / (1.0 - s_gen + eps)
            sr_j = max(0.0, min(1.0, sr_j))
            sr_sum += sr_j
        return sr_sum / n if n > 0 else 0.0
