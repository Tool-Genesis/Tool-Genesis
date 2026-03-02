import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple


def iter_projects(run_root: Path) -> List[Path]:
    if not run_root.exists() or not run_root.is_dir():
        return []
    projects: List[Path] = []
    for entry in sorted(run_root.iterdir()):
        if entry.is_dir():
            projects.append(entry)
    return projects


def iter_servers(project_dir: Path) -> List[Path]:
    servers: List[Path] = []
    for entry in sorted(project_dir.iterdir()):
        if entry.is_dir():
            servers.append(entry)
    return servers


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def has_tool_calls_in_messages(messages: List[Dict[str, Any]]) -> bool:
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list) and len(tool_calls) > 0:
            return True
    return False


def has_tool_calls_in_conversation_file(path: Path) -> bool:
    data = load_json(path)
    if data is None:
        return False
    if isinstance(data, list):
        if has_tool_calls_in_messages(data):
            return True
    if isinstance(data, dict):
        messages = data.get("messages")
        if isinstance(messages, list) and has_tool_calls_in_messages(messages):
            return True
        conv = data.get("conversation")
        if isinstance(conv, list) and has_tool_calls_in_messages(conv):
            return True
    return False


def analyze_server(server_dir: Path, conversation_filename: str) -> Tuple[int, int, bool]:
    total_conversations = 0
    conversations_with_tool_calls = 0
    has_any_tool_calls = False
    for root, _, files in os.walk(server_dir):
        for fn in files:
            if fn != conversation_filename:
                continue
            fpath = Path(root) / fn
            total_conversations += 1
            if has_tool_calls_in_conversation_file(fpath):
                conversations_with_tool_calls += 1
                has_any_tool_calls = True
    return total_conversations, conversations_with_tool_calls, has_any_tool_calls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=str,
        default="temp/run_benchmark_v3",
        help="Root directory of generated projects, e.g. temp/run_benchmark_v3",
    )
    parser.add_argument(
        "--conversation-filename",
        type=str,
        default="conversation.json",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    run_root = Path(args.run_root)
    conversation_filename = args.conversation_filename

    projects = iter_projects(run_root)
    if not projects:
        print(f"No projects found under {run_root}")
        return

    summary: Dict[str, Any] = {}
    global_total_servers = 0
    global_servers_with_tool_calls = 0

    for project_dir in projects:
        project_name = project_dir.name
        servers = iter_servers(project_dir)
        project_total_servers = 0
        project_servers_with_tool_calls = 0
        project_servers: Dict[str, Any] = {}

        for server_dir in servers:
            server_name = server_dir.name
            total_conversations, conversations_with_tool_calls, has_any_tool_calls = analyze_server(
                server_dir, conversation_filename
            )
            if total_conversations == 0:
                continue
            project_total_servers += 1
            if has_any_tool_calls:
                project_servers_with_tool_calls += 1
            project_servers[server_name] = {
                "total_conversations": total_conversations,
                "conversations_with_tool_calls": conversations_with_tool_calls,
                "has_tool_calls": has_any_tool_calls,
            }

        if project_total_servers == 0:
            continue

        summary[project_name] = {
            "total_servers": project_total_servers,
            "servers_with_tool_calls": project_servers_with_tool_calls,
            "servers": project_servers,
        }
        global_total_servers += project_total_servers
        global_servers_with_tool_calls += project_servers_with_tool_calls

    if not summary:
        print("No servers with conversations found")
        return

    for project_name, pdata in summary.items():
        total_servers = pdata["total_servers"]
        servers_with_tool_calls = pdata["servers_with_tool_calls"]
        print(f"Project: {project_name}")
        print(f"  Servers with conversations: {total_servers}")
        print(f"  Servers with tool calls:    {servers_with_tool_calls}")
        for server_name, sdata in sorted(pdata["servers"].items()):
            mark = "Y" if sdata["has_tool_calls"] else "N"
            print(
                f"    [{mark}] {server_name} "
                f"(conversations={sdata['total_conversations']}, "
                f"with_tool_calls={sdata['conversations_with_tool_calls']})"
            )
        print()

    print("Global summary")
    print(f"  Total servers with conversations: {global_total_servers}")
    print(f"  Servers with tool calls:          {global_servers_with_tool_calls}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_root": str(run_root),
                    "total_servers": global_total_servers,
                    "servers_with_tool_calls": global_servers_with_tool_calls,
                    "projects": summary,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Wrote JSON summary to {out_path}")


if __name__ == "__main__":
    main()

