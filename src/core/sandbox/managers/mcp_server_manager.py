# manager.py
from __future__ import annotations
import logging, sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import anyio
from anyio import create_memory_object_stream
from anyio.abc import ObjectReceiveStream, ObjectSendStream
from anyio.from_thread import start_blocking_portal

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[MCP-MGR] %(levelname)s %(message)s"))
    logger.addHandler(h)


@dataclass
class MCPServerConfig:
    """MCP server configuration class, encapsulates all parameters needed to start a server"""
    name: str  # Server name
    python_exec: str = sys.executable
    boot_module: str = "src.apps.factory.mcp_boot"  # Boot module (Python -m format)
    file_path: Optional[str] = None  # Option 1: module file path + optional class name
    class_name: Optional[str] = None
    code_path: Optional[str] = None  # Option 2: code file path
    spec_path: Optional[str] = None  # JSON: passed to App.__init__(spec)
    auto_load_path: Optional[str] = None
    auto_save_path: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    max_concurrency: int = 8
    ready_timeout: Optional[float] = 30.0


@dataclass
class _Cmd:
    kind: str  # 'WAIT_READY' | 'LIST' | 'CALL' | 'STOP' | 'HEALTH'
    payload: Dict[str, Any]
    reply: ObjectSendStream


class _ServerCore:
    def __init__(self, name: str, params: StdioServerParameters, *, max_concurrency: int = 8) -> None:
        self.name = name
        self.params = params
        self.max_concurrency = max_concurrency
        self._recv: Optional[ObjectReceiveStream] = None
        self._send: Optional[ObjectSendStream] = None
        self._ready_evt: Optional[anyio.Event] = None
        self._running: bool = False
        self._tools_cache: List[str] = []

    def attach_cmd_streams(self, recv: ObjectReceiveStream, send: ObjectSendStream) -> None:
        self._recv = recv
        self._send = send

    async def run(self) -> None:
        assert self._recv is not None
        assert self._send is not None
        self._ready_evt = anyio.Event()
        self._running = True

        try:
            async with stdio_client(self.params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    resp = await session.list_tools()
                    self._tools_cache = [t.name for t in resp.tools]
                    self._ready_evt.set()
                    logger.info(f"[{self.name}] ready, tools={self._tools_cache}")

                    sem = anyio.Semaphore(self.max_concurrency)

                    async def handle_call(cmd: _Cmd) -> None:
                        tool = cmd.payload.get("tool")
                        args = cmd.payload.get("args", {})
                        timeout = cmd.payload.get("timeout", None)
                        try:
                            async with sem:
                                if timeout:
                                    with anyio.fail_after(float(timeout)):
                                        result = await session.call_tool(tool, args)
                                else:
                                    result = await session.call_tool(tool, args)

                            if hasattr(result, "model_dump"):
                                serializable_result = result.model_dump()
                            elif hasattr(result, "dict"):
                                serializable_result = result.dict()
                            elif hasattr(result, "__dict__"):
                                serializable_result = result.__dict__
                            else:
                                serializable_result = result

                            await cmd.reply.send(("ok", serializable_result))
                        except Exception as e:
                            await cmd.reply.send(("err", repr(e)))

                    async with anyio.create_task_group() as tg:
                        async for cmd in self._recv:
                            if cmd.kind == "WAIT_READY":
                                await self._ready_evt.wait()
                                await cmd.reply.send(True)
                            elif cmd.kind == "LIST":
                                await cmd.reply.send(list(self._tools_cache))
                            elif cmd.kind == "HEALTH":
                                await cmd.reply.send(self._running)
                            elif cmd.kind == "CALL":
                                tg.start_soon(handle_call, cmd)
                            elif cmd.kind == "STOP":
                                await cmd.reply.send(True)
                                tg.cancel_scope.cancel()
                                break
        except Exception as e:
            logger.error(f"[{self.name}] owner crashed: {e}")
        finally:
            self._running = False
            try:
                if self._recv:
                    await self._recv.aclose()
            except Exception:
                pass
            logger.info(f"[{self.name}] owner stopped.")


class MCPServerManagerSync:
    """
    Unified external API exposing only one "start" API: start_server(...)
    —— Internally starts via factory boot script (mcp_boot.py) as stdio subprocess.
    Other public capabilities: list_tools / call_tool / stop_server / servers are retained.
    """

    def __init__(self) -> None:
        self._portal_cm = start_blocking_portal()
        self._portal = self._portal_cm.__enter__()
        self._servers: Dict[str, Tuple[_ServerCore, ObjectSendStream]] = {}
        self._server_configs: Dict[str, MCPServerConfig] = {}

    def close(self) -> None:
        for name in list(self._servers.keys()):
            try:
                logger.info(f"[{name}] stopping")
                self.stop_server(name)
            except Exception:
                pass
        self._portal_cm.__exit__(None, None, None)

    def __enter__(self) -> "MCPServerManagerSync":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---------- Internal message channel ----------

    @staticmethod
    async def _await_ready_and_list(send: ObjectSendStream, ready_timeout: Optional[float]) -> List[str]:
        rs, rr = create_memory_object_stream(1)
        if ready_timeout:
            with anyio.fail_after(ready_timeout):
                await send.send(_Cmd("WAIT_READY", {}, rs))
                await rr.receive()
        else:
            await send.send(_Cmd("WAIT_READY", {}, rs))
            await rr.receive()

        rs2, rr2 = create_memory_object_stream(1)
        await send.send(_Cmd("LIST", {}, rs2))
        tools = await rr2.receive()
        return tools

    @staticmethod
    async def _send_list(send: ObjectSendStream) -> List[str]:
        rs, rr = create_memory_object_stream(1)
        await send.send(_Cmd("LIST", {}, rs))
        return await rr.receive()

    @staticmethod
    async def _send_call(send: ObjectSendStream, tool: str, args: Dict[str, Any], timeout: Optional[float]) -> Any:
        rs, rr = create_memory_object_stream(1)
        await send.send(_Cmd("CALL", {"tool": tool, "args": args, "timeout": timeout}, rs))
        status, payload = await rr.receive()
        if status == "ok":
            return payload
        raise RuntimeError(payload)

    @staticmethod
    async def _send_stop(send: ObjectSendStream) -> bool:
        rs, rr = create_memory_object_stream(1)
        await send.send(_Cmd("STOP", {}, rs))
        return await rr.receive()

    @staticmethod
    async def _send_health(send: ObjectSendStream) -> bool:
        rs, rr = create_memory_object_stream(1)
        await send.send(_Cmd("HEALTH", {}, rs))
        return await rr.receive()

    # ---------- Unified startup entry point (via factory) ----------

    def start_server(self, config: MCPServerConfig) -> List[str]:
        """
        Unified startup method: use boot script + MCPServerFactory to build FastMCP in subprocess and run_stdio().
        - You only need to provide file_path/class_name or code_path (choose one).
        - Others (spec/auto_load/auto_save) are all optional.
        """
        if config.name in self._servers:
            logger.info(f"[{config.name}] already exists, restarting")
            self.stop_server(config.name)
        if not (config.file_path or config.code_path):
            raise ValueError("One of file_path or code_path must be provided")

        # Assemble command line (python -m mcp_boot ...)
        cmd_args: List[str] = ["-m", config.boot_module]
        if config.file_path:
            cmd_args += ["--file-path", config.file_path]
        if config.class_name:
            cmd_args += ["--class-name", config.class_name]
        if config.code_path:
            cmd_args += ["--code-path", config.code_path]
        if config.spec_path:
            cmd_args += ["--spec-path", config.spec_path]
        if config.auto_load_path:
            cmd_args += ["--auto-load", config.auto_load_path]
        if config.auto_save_path:
            cmd_args += ["--auto-save", config.auto_save_path]

        env = dict(config.env or {})
        env.setdefault("PYTHONUNBUFFERED", "1")
        # Ensure project root directory is in PYTHONPATH
        import os
        from pathlib import Path
        project_root = str(Path(__file__).parent.parent.parent.parent)
        current_pythonpath = env.get("PYTHONPATH", "")
        if current_pythonpath:
            env["PYTHONPATH"] = f"{project_root}{os.pathsep}{current_pythonpath}"
        else:
            env["PYTHONPATH"] = project_root

        params = StdioServerParameters(command=config.python_exec, args=cmd_args, env=env)

        send, recv = create_memory_object_stream(max_buffer_size=100)
        core = _ServerCore(config.name, params, max_concurrency=config.max_concurrency)
        core.attach_cmd_streams(recv, send)
        self._portal.start_task_soon(core.run)

        tools: List[str] = self._portal.call(self._await_ready_and_list, send, config.ready_timeout)
        self._servers[config.name] = (core, send)
        self._server_configs[config.name] = config
        logger.info(f"[{config.name}] started, tools={tools}")
        return tools

    def list_tools(self, name: str, public_only: bool = False) -> List[str]:
        if name not in self._servers:
            raise RuntimeError(f"Server '{name}' not found")
        _, send = self._servers[name]
        tools = self._portal.call(self._send_list, send)
        if public_only:
            tools = [t for t in tools if not t.startswith("_")]
        return tools

    def call_tool(self, name: str, tool: str, arguments: Dict[str, Any], timeout: Optional[float] = 30.0) -> Any:
        if name not in self._servers:
            raise RuntimeError(f"Server '{name}' not found")
        _, send = self._servers[name]
        return self._portal.call(self._send_call, send, tool, arguments, timeout)

    def stop_server(self, name: str) -> bool:
        if name not in self._servers:
            return True
        _, send = self._servers.pop(name)
        ok = self._portal.call(self._send_stop, send)
        try:
            self._portal.call(send.aclose)
        except Exception:
            pass
        logger.info(f"[{name}] stopped={ok}")
        return ok

    def restart_server(self, name: str) -> List[str]:
        if name not in self._server_configs:
            raise RuntimeError(f"Server '{name}' not found")
        if name in self._servers:
            try:
                logger.info(f"[{name}] restarting")
                self.stop_server(name)
            except Exception:
                pass
        return self.start_server(self._server_configs[name])

    def reset_server(self, name: str) -> List[str]:
        """Reset server by restarting"""
        return self.restart_server(name)
    
    def get_status(self, name: str) -> bool:
        """Check server running status"""
        if name not in self._servers:
            return False
        _, send = self._servers[name]
        try:
            return self._portal.call(self._send_health, send)
        except Exception:
            return False

    def servers(self) -> List[str]:
        return list(self._servers.keys())
