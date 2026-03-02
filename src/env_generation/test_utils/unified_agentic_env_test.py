# debug_env/test_env.py
import importlib
import json
from pathlib import Path

import pytest

# Template placeholders: replace before running generated tests.
SERVER_MODULE = "env_module_test"
SERVER_CLASS_NAME = "ServerClassPlaceholder"
SERVER = "<server_name>"
TOOL_NAME = "<tool_name>"
COMBINED = "data/tools/combined_tools.json"


@pytest.fixture
def env():
    try:
        mod = importlib.import_module(SERVER_MODULE)
        server_cls = getattr(mod, SERVER_CLASS_NAME)
    except Exception as exc:
        pytest.skip(f"Template test requires replacement of server placeholders: {exc}")

    with Path(COMBINED).open("r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        srv = next(s for s in data["servers"] if s["server_name"] == SERVER)
    except StopIteration:
        pytest.skip("Template test requires replacement of <server_name> placeholder.")
    tools = srv["tools"]
    fs_root = f"temp/data/{SERVER.replace(' ', '_').replace('-', '_')}"
    mcp_server = server_cls(tools, fs_root)._initialize_mcp_server()
    return mcp_server


class DummyMCP:
    """Stand-in for FastMCP that attaches @tool() methods as attributes."""

    def __init__(self, name):
        self.name = name

    def tool(self):
        def decorator(fn):
            setattr(self, fn.__name__, fn)
            return fn

        return decorator


@pytest.fixture(autouse=True)
def patch_fastmcp(monkeypatch):
    try:
        mod = importlib.import_module(SERVER_MODULE)
    except Exception:
        return
    monkeypatch.setattr(mod, "FastMCP", DummyMCP, raising=False)


def _get_underlying_env(mcp_server):
    return mcp_server._env


def _exec(m, tool, args):
    env_obj = _get_underlying_env(m)
    return env_obj._execute_action(json.dumps({"tool": tool, "args": args}))


def _state(m):
    env_obj = _get_underlying_env(m)
    return json.loads(env_obj._get_environment_state())


@pytest.fixture(autouse=True)
def reset_state(env):
    _get_underlying_env(env)._reset_environment_state()


def test_tool_template(env):
    s = _state(env)
    args = {
        # Fill in the required and optional parameters of tools
    }
    out = _exec(env, TOOL_NAME, args)

    assert isinstance(out, dict), "Response must be dict"
    assert "members" in out, "Missing key 'members'"
    assert isinstance(out["members"], list), "'members' should be list"

    expected_name = args.get("name") or s["current_user"]
    assert out["members"], "Expected non-empty members list"
    assert all(m["username"] == expected_name for m in out["members"])
