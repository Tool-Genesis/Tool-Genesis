from __future__ import annotations
"""
持久化沙箱实现 - 添加兼容性方法
"""
import os
import sys
import subprocess
import json
import signal
import shutil
import ast
import threading
import uuid
from pathlib import Path
import tempfile
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Sequence
from dataclasses import dataclass
from enum import Enum
from queue import Queue, Empty
from .core import HEAVY_PACKAGES, _global_cleaner
from .managers.mcp_server_manager import MCPServerManagerSync, MCPServerConfig
from .managers.environment_manager import EnvironmentManager
from .managers.session_manager import SessionManager, SessionConfig
from .managers.workspace_manager import WorkspaceManager, WorkspaceConfig, WorkspaceHandle, PersistPolicy

logger = logging.getLogger(__name__)

class PersistentEnvironmentSandbox:
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        memory_limit_mb: int = 512,
        cpu_time_limit: int = 30,
        wall_time_limit: int = 30 * 60,
        temp_dir: Optional[str] = None,
        debug: bool = True,
        auto_cleanup: bool = True,
        timeout_minutes: int = 5,
        memory_only_mode: bool = False,
        enable_lazy_cache_write: bool = True,
        cache_update_threshold_minutes: int = 5,
        workspace_persist_policy: PersistPolicy = PersistPolicy.EPHEMERAL,
        cleanup_paths_on_close: Optional[List[str]] = None,
        ws_manager: Optional[WorkspaceManager] = None,
        mount_dir: Optional[str] = None,
        snapshot_retention_days: Optional[int] = None,
        allow_network: bool = True,
        allow_subprocess: bool = True,
        extra_allowed_paths: Optional[List[str]] = None,
    ):
        # 基础配置
        self.base_mem_mb = memory_limit_mb
        self.memory_limit_mb = memory_limit_mb
        self.mem_bytes = memory_limit_mb * 1024 * 1024
        self.cpu_limit = cpu_time_limit
        self.wall_limit = wall_time_limit
        td = temp_dir
        if not td:
            td = os.environ.get("TMPDIR") or tempfile.gettempdir() or "/tmp"
        td = str(td).strip()
        try:
            os.makedirs(td, mode=0o700, exist_ok=True)
        except Exception:
            td = os.path.join(os.getcwd(), ".tmp")
            os.makedirs(td, mode=0o700, exist_ok=True)
        self.temp_dir = td
        self.debug = debug
        self.allow_network = bool(allow_network)
        self.allow_subprocess = bool(allow_subprocess)
        self.extra_allowed_paths = [
            os.path.abspath(p) for p in (extra_allowed_paths or []) if isinstance(p, str) and p.strip()
        ]
        
        # 生成会话ID（如果未提供）
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # 创建会话管理器
        session_config = SessionConfig(
            session_id=session_id,
            timeout_minutes=timeout_minutes,
            auto_cleanup=auto_cleanup,
            debug=debug
        )
        self.session_manager = SessionManager(session_config, self._cleanup_callback)
                
        # 为了保持向后兼容性，保留这些属性
        self.session_id = self.session_manager.session_id
        self.session_timeout = timeout_minutes
        self.timeout_minutes = timeout_minutes
        self.auto_cleanup = auto_cleanup
        self.created_at = self.session_manager.created_at
        self.last_used = self.session_manager.last_used
        self.last_accessed = self.session_manager.last_accessed
        self.cleanup_timer = None
        
        # 持久化状态
        self.venv_path: Optional[str] = None
        self.work_dir: Optional[str] = None
        
        # 持久化进程相关
        self.process: Optional[subprocess.Popen] = None
        self.process_lock = threading.Lock()
        self.command_counter = 0
        
        # 会话状态
        self._session_active = True
        self._installed_packages = set()
        
        # MCP服务器管理器
        self._mcp_server_manager = MCPServerManagerSync()
        
        # 环境管理器
        self._environment_manager = EnvironmentManager(
            memory_only_mode=memory_only_mode,
            enable_lazy_cache_write=enable_lazy_cache_write,
            cache_update_threshold_minutes=cache_update_threshold_minutes
        )
        
        # 工作区管理器
        if ws_manager is None:
            # 创建默认的工作区管理器
            ws_config = WorkspaceConfig(
                workspace_root=os.path.join(self.temp_dir, "sandbox", "ws", "workspaces"),
                snapshot_root=os.path.join(self.temp_dir, "sandbox", "ws", "snapshots"),
                persist_policy=workspace_persist_policy,
                cleanup_paths_on_close=cleanup_paths_on_close,
                mount_dir=mount_dir,
                debug=debug,
                snapshot_retention_days=snapshot_retention_days,
            )
            self._workspace_manager = WorkspaceManager(ws_config)
        else:
            self._workspace_manager = ws_manager
        
        # 创建工作区句柄
        self._workspace_handle = self._workspace_manager.create(
            session_id=self.session_id,
            policy=workspace_persist_policy,
            mount_dir=mount_dir
        )
        
        # 设置工作目录（保持向后兼容性）
        self.work_dir = self._workspace_handle.work_dir    
    
    def _cleanup_callback(self, session_id: str):
        """会话管理器的清理回调"""
        self.cleanup_session()
    
    def is_timeout(self) -> bool:
        """检查会话是否超时 - 委托给会话管理器"""
        return self.session_manager.is_timeout()
    
    def get_remaining_time(self) -> float:
        """获取剩余时间（分钟） - 委托给会话管理器"""
        return self.session_manager.get_remaining_time()
    
    def extend_timeout(self, additional_minutes: int = 5):
        """延长会话超时时间 - 委托给会话管理器"""
        self.session_manager.extend_timeout(additional_minutes)
        # 更新本地属性以保持兼容性
        self.timeout_minutes = self.session_manager.config.timeout_minutes
        self.session_timeout = self.timeout_minutes
    
    def cleanup(self):
        """兼容EnvironmentSandbox的cleanup方法"""
        self.cleanup_session()
    
    def touch(self):
        """更新最后访问时间并重置清理定时器 - 委托给会话管理器"""
        self.session_manager.touch()
        # 更新本地属性以保持兼容性
        self.last_used = self.session_manager.last_used
        self.last_accessed = self.session_manager.last_accessed
    
    
    def _create_persistent_script(self) -> str:
        """创建持久化Python脚本"""
        if self.venv_path and os.path.exists(f"{self.venv_path}/bin/python"):
            python_executable = f"{self.venv_path}/bin/python"
        else:
            python_executable = sys.executable
        allow_network_literal = "True" if self.allow_network else "False"
        allow_subprocess_literal = "True" if self.allow_subprocess else "False"
        extra_allowed_paths_json = json.dumps(self.extra_allowed_paths, ensure_ascii=False)

        script = f'''
import sys
import platform
import json
import traceback
import signal
import os
import resource
import builtins
import subprocess
import io
from contextlib import redirect_stdout, redirect_stderr

# 确保输出立即刷新
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# 全局调试/日志开关（可被命令或环境变量覆盖）
DEBUG_ENABLED = False
STREAM_INSTALL_LOGS = True
ALLOW_NETWORK = {allow_network_literal}
ALLOW_SUBPROCESS = {allow_subprocess_literal}
EXTRA_ALLOWED_PATHS = {extra_allowed_paths_json}
_env_dbg = os.environ.get("PERSISTENT_DEBUG")
if _env_dbg is not None:
    DEBUG_ENABLED = _env_dbg.lower() not in ("0", "false", "no", "off")
_env_stream = os.environ.get("PERSISTENT_STREAM_INSTALL_LOGS")
if _env_stream is not None:
    STREAM_INSTALL_LOGS = _env_stream.lower() not in ("0", "false", "no", "off")

def log_debug(msg):
    """调试日志"""
    if DEBUG_ENABLED:
        print(f"[PERSISTENT_DEBUG] {{msg}}", file=sys.stderr, flush=True)

os.chdir("{self.work_dir}")

if "VIRTUAL_ENV" in os.environ:
    del os.environ["VIRTUAL_ENV"]
if "CONDA_DEFAULT_ENV" in os.environ:
    del os.environ["CONDA_DEFAULT_ENV"]

venv_path = r"{self.venv_path or ''}"
if venv_path and os.path.exists(venv_path):
    os.environ["VIRTUAL_ENV"] = venv_path
    # 确保 PATH 中虚拟环境的 bin 目录在最前面
    venv_bin = os.path.join(venv_path, "bin")
    if os.path.exists(venv_bin):
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = venv_bin + ":" + current_path

# 设置资源限制（macOS 上跳过 AS 和 DATA，因为不支持）
try:
    mem_soft = {self.mem_bytes}
    mem_hard = mem_soft + 512 * 1024 * 1024

    # Only set address‐space and data‐segment limits on non-Darwin
    if platform.system() != "Darwin" and hasattr(resource, "RLIMIT_AS"):
        resource.setrlimit(resource.RLIMIT_AS, (mem_soft, mem_hard))
    if platform.system() != "Darwin" and hasattr(resource, "RLIMIT_DATA"):
        resource.setrlimit(resource.RLIMIT_DATA, (mem_soft, mem_hard))

    # These are OK everywhere
    resource.setrlimit(resource.RLIMIT_CPU, ({self.cpu_limit}, {self.cpu_limit} + 5))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 256))

except Exception as e:
    log_debug(f"Resource limit warning: {{e}}")

_original_subprocess = subprocess

# =====================================================================
# 文件系统访问限制 — 通过闭包隐藏原始引用，防止沙箱代码恢复
# =====================================================================
def _install_fs_restrictions():
    import io as _io
    import pathlib as _pathlib

    _orig_builtins_open = builtins.open
    _orig_io_open = _io.open
    _orig_io_FileIO = _io.FileIO
    _orig_os_open = os.open
    _sandbox_root = os.path.abspath(os.getcwd())
    _allowed_roots = [_sandbox_root] + [os.path.abspath(p) for p in EXTRA_ALLOWED_PATHS if p]

    def _is_in_allowed_roots(abs_path):
        for root in _allowed_roots:
            try:
                if os.path.commonpath([abs_path, root]) == root:
                    return True
            except Exception:
                continue
        return False

    def _check_path(file):
        p = str(file)
        abs_path = os.path.abspath(p)
        if not _is_in_allowed_roots(abs_path):
            raise PermissionError('禁止访问沙箱目录之外的文件: ' + abs_path)

    # builtins.open
    def _safe_open(file, *a, **k):
        _check_path(file)
        return _orig_builtins_open(file, *a, **k)
    builtins.open = _safe_open

    # io.open
    def _safe_io_open(file, *a, **k):
        _check_path(file)
        return _orig_io_open(file, *a, **k)
    _io.open = _safe_io_open

    # io.FileIO
    class _SafeFileIO(_orig_io_FileIO):
        def __init__(self, name, *a, **k):
            _check_path(name)
            super().__init__(name, *a, **k)
    _io.FileIO = _SafeFileIO

    # os.open
    def _safe_os_open(path, flags, mode=0o777, *, dir_fd=None):
        _check_path(path)
        return _orig_os_open(path, flags, mode, dir_fd=dir_fd)
    os.open = _safe_os_open

    # pathlib.Path — 限制 open / read_text / write_text / read_bytes / write_bytes
    _orig_path_open = _pathlib.Path.open
    _orig_path_read_text = _pathlib.Path.read_text
    _orig_path_write_text = _pathlib.Path.write_text
    _orig_path_read_bytes = _pathlib.Path.read_bytes
    _orig_path_write_bytes = _pathlib.Path.write_bytes

    def _safe_path_open(self, *a, **k):
        _check_path(self)
        return _orig_path_open(self, *a, **k)
    def _safe_path_read_text(self, *a, **k):
        _check_path(self)
        return _orig_path_read_text(self, *a, **k)
    def _safe_path_write_text(self, *a, **k):
        _check_path(self)
        return _orig_path_write_text(self, *a, **k)
    def _safe_path_read_bytes(self, *a, **k):
        _check_path(self)
        return _orig_path_read_bytes(self, *a, **k)
    def _safe_path_write_bytes(self, *a, **k):
        _check_path(self)
        return _orig_path_write_bytes(self, *a, **k)
    _pathlib.Path.open = _safe_path_open
    _pathlib.Path.read_text = _safe_path_read_text
    _pathlib.Path.write_text = _safe_path_write_text
    _pathlib.Path.read_bytes = _safe_path_read_bytes
    _pathlib.Path.write_bytes = _safe_path_write_bytes

    return _orig_builtins_open  # needed by pip install internals

_orig_open_ref = _install_fs_restrictions()

# =====================================================================
# 子进程执行限制
# =====================================================================
if not ALLOW_SUBPROCESS:
    def _deny_subprocess(*_args, **_kwargs):
        raise PermissionError("Subprocess execution is disabled in this sandbox session")
    subprocess.run = _deny_subprocess
    subprocess.Popen = _deny_subprocess
    subprocess.call = _deny_subprocess
    subprocess.check_call = _deny_subprocess
    subprocess.check_output = _deny_subprocess
    subprocess.getoutput = _deny_subprocess
    subprocess.getstatusoutput = _deny_subprocess
    # Also restrict os-level process functions
    os.system = _deny_subprocess
    os.popen = _deny_subprocess
    if hasattr(os, 'execv'):
        os.execv = _deny_subprocess
    if hasattr(os, 'execve'):
        os.execve = _deny_subprocess
    if hasattr(os, 'execvp'):
        os.execvp = _deny_subprocess
    if hasattr(os, 'execvpe'):
        os.execvpe = _deny_subprocess

# =====================================================================
# 网络访问限制
# =====================================================================
if not ALLOW_NETWORK:
    import socket as _socket
    _orig_socket_init = _socket.socket.__init__
    _orig_create_connection = _socket.create_connection
    _NET_ERR = "Network access is disabled in this sandbox session"
    def _deny_create_connection(*_args, **_kwargs):
        raise PermissionError(_NET_ERR)
    class _NoNetSocket(_socket.socket):
        def __init__(self, *_args, **_kwargs):
            super().__init__(*_args, **_kwargs)
        def connect(self, *_args, **_kwargs):
            raise PermissionError(_NET_ERR)
        def connect_ex(self, *_args, **_kwargs):
            raise PermissionError(_NET_ERR)
        def sendto(self, *_args, **_kwargs):
            raise PermissionError(_NET_ERR)
    _socket.create_connection = _deny_create_connection
    _socket.socket = _NoNetSocket
    # Also block urllib
    try:
        import urllib.request as _ureq
        _orig_urlopen = _ureq.urlopen
        def _deny_urlopen(*_a, **_k):
            raise PermissionError(_NET_ERR)
        _ureq.urlopen = _deny_urlopen
    except Exception:
        pass

PERSISTENT_GLOBALS = {{
    '__name__': '__main__',
    '__builtins__': __builtins__,
}}

# 发送就绪信号
print(json.dumps({{"type": "READY", "session_id": "{self.session_id}"}}, ensure_ascii=False), flush=True)

def execute_with_timeout(code, timeout_sec):
    """在超时限制下执行代码，捕获所有输出"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"代码执行超时 ({{timeout_sec}}秒)")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)
    
    import io
    import contextlib
    
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    try:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        # 重定向到缓冲区
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer
        
        try:
            exec(code, PERSISTENT_GLOBALS)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        
        stdout_content = stdout_buffer.getvalue()
        stderr_content = stderr_buffer.getvalue()
        
        # 确保输出不为空时有内容
        log_debug(f"Captured stdout: {{repr(stdout_content)}}")
        log_debug(f"Captured stderr: {{repr(stderr_content)}}")
        
        return None, stdout_content, stderr_content
        
    except Exception as e:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
        stderr_content = stderr_buffer.getvalue()
        if not stderr_content:
            stderr_content = traceback.format_exc()
        return e, stdout_buffer.getvalue(), stderr_content
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        stdout_buffer.close()
        stderr_buffer.close()


def evaluate_with_timeout(code, timeout_sec):
    """在超时限制下求值"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"代码求值超时 ({{timeout_sec}}秒)")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)
    
    try:
        result = eval(code, PERSISTENT_GLOBALS)
        return result, None
    except Exception as e:
        return None, e
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def send_response(response):
    """发送响应并确保刷新"""
    try:
        response_json = json.dumps(response, ensure_ascii=False)
        log_debug(f"Sending response: {{response_json[:100]}}...")
        print(response_json, flush=True)
        log_debug("Response sent and flushed")
    except Exception as e:
        log_debug(f"Error sending response: {{e}}")
        error_response = {{
            "type": "ERROR",
            "error": f"响应发送失败: {{str(e)}}"
        }}
        print(json.dumps(error_response, ensure_ascii=False), flush=True)

log_debug("Persistent process started and ready")

while True:
    try:
        line = input()
        if line.strip() == "EXIT":
            break
            
        try:
            command_data = json.loads(line)
        except json.JSONDecodeError as e:
            send_response({{"type": "ERROR", "error": f"JSON解析错误: {{e}}"}})
            continue
            
        # 按命令覆盖调试/日志开关
        cmd_debug = command_data.get("debug")
        if cmd_debug is not None:
            DEBUG_ENABLED = bool(cmd_debug)
        cmd_stream = command_data.get("stream_logs")
        if cmd_stream is not None:
            STREAM_INSTALL_LOGS = bool(cmd_stream)

        command_type = command_data.get("type")
        command_id = command_data.get("id", "unknown")
        timeout_sec = command_data.get("timeout", {self.wall_limit})
        
        log_debug(f"Processing command: {{command_type}}")
        
        if command_type == "EXEC":
            code = command_data.get("code", "")
            try:
                error, stdout_content, stderr_content = execute_with_timeout(code, timeout_sec)
                if error:
                    response = {{
                        "type": "ERROR", 
                        "id": command_id,
                        "error": str(error),
                        "error_type": type(error).__name__,
                        "stdout": stdout_content,
                        "stderr": stderr_content,
                        "traceback": traceback.format_exc()
                    }}
                else:
                    response = {{
                        "type": "SUCCESS", 
                        "id": command_id,
                        "message": "代码执行成功",
                        "stdout": stdout_content,
                        "stderr": stderr_content
                    }}
                send_response(response)
            except Exception as e:
                response = {{
                    "type": "FATAL_ERROR", 
                    "id": command_id,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }}
                send_response(response)
        
        elif command_type == "EVAL":
            code = command_data.get("code", "")
            try:
                result, error = evaluate_with_timeout(code, timeout_sec)
                if error:
                    response = {{
                        "type": "ERROR", 
                        "id": command_id,
                        "error": str(error),
                        "error_type": type(error).__name__,
                        "traceback": traceback.format_exc()
                    }}
                else:
                    response = {{
                        "type": "RESULT", 
                        "id": command_id,
                        "value": str(result)
                    }}
                send_response(response)
            except Exception as e:
                response = {{
                    "type": "FATAL_ERROR", 
                    "id": command_id,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }}
                send_response(response)
        
        elif command_type == "INSTALL":
            packages = command_data.get("packages", [])
            try:
                # 控制详细日志的开关：默认跟随 DEBUG_ENABLED，可被环境变量覆盖
                install_verbose = DEBUG_ENABLED
                env_override = os.environ.get("AGENTGEN_PIP_VERBOSE")
                if env_override is not None:
                    install_verbose = env_override.lower() not in ("0", "false", "no", "off")

                # 打印安装上下文信息
                try:
                    log_debug(f"[INSTALL_CTXT] python={{sys.executable}} | version={{sys.version.split()[0]}} | platform={{platform.platform()}}")
                    pv = _original_subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True, text=True)
                    log_debug(f"[INSTALL_CTXT] {{pv.stdout.strip() or pv.stderr.strip()}}")
                    log_debug(f"[INSTALL_CTXT] CWD={{os.getcwd()}} | PATH[:120]={{os.environ.get('PATH','')[:120]}}")
                    for key in ("PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy"):
                        val = os.environ.get(key)
                        log_debug(f"[INSTALL_ENV] {{key}}={{val if val else 'None'}}")
                except Exception as ctx_e:
                    log_debug(f"[INSTALL_CTXT] context error: {{ctx_e}}")

                log_debug(f"Installing packages: {{packages}}")

                details = []
                overall_stdout = io.StringIO()
                overall_stderr = io.StringIO()
                timeout_sec = command_data.get("timeout", 300)

                for pkg in packages:
                    cmd = [sys.executable, "-m", "pip", "install"]
                    if install_verbose:
                        cmd += ["-vvv", "--progress-bar", "off"]
                    else:
                        cmd += ["--progress-bar", "off"]

                    use_no_cache = os.environ.get("AGENTGEN_PIP_NO_CACHE", "1").lower() not in ("0", "false", "no")
                    if use_no_cache:
                        cmd += ["--no-cache-dir"]

                    # 将 pip 日志写入沙箱内文件，并实时尾随到 stderr，以提供安装进度
                    uuid_hex = __import__('uuid').uuid4().hex
                    log_path = os.path.join(os.getcwd(), ".pip-install-" + pkg.replace("/", "_") + "-" + uuid_hex + ".log")
                    cmd += ["--disable-pip-version-check", "--log", log_path, pkg]
                    stop_tail = False
                    _tail_thread = None
                    def _tail_log():
                        try:
                            import time as _t
                            waited = 0.0
                            while not os.path.exists(log_path) and waited < 30.0 and not stop_tail:
                                _t.sleep(0.2)
                                waited += 0.2
                            if not os.path.exists(log_path):
                                return
                            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                                while not stop_tail:
                                    where = f.tell()
                                    line = f.readline()
                                    if not line:
                                        _t.sleep(0.2)
                                        f.seek(where)
                                    else:
                                        # 仅在允许流式日志时打印
                                        if STREAM_INSTALL_LOGS:
                                            print("[INSTALL_LOG] " + line.rstrip(), file=sys.stderr, flush=True)
                        except Exception as _te:
                            log_debug("[INSTALL_LOG_ERR] " + str(_te))
                    # 仅在允许流式日志时启动尾随线程
                    if STREAM_INSTALL_LOGS:
                        _tail_thread = __import__('threading').Thread(target=_tail_log, daemon=True)
                        _tail_thread.start()

                    start_t = __import__('time').time()
                    log_debug(f"[INSTALL_RUN] cmd={{' '.join(cmd)}}")
                    try:
                        result = _original_subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
                        elapsed = __import__('time').time() - start_t
                        if result.stdout:
                            # 仅在允许流式日志时打印到 stderr，避免打断 stdout 的 JSON 协议
                            if STREAM_INSTALL_LOGS:
                                print(result.stdout, end="", file=sys.stderr)
                            overall_stderr.write(result.stdout)
                        if result.stderr:
                            if STREAM_INSTALL_LOGS:
                                print(result.stderr, end="", file=sys.stderr)
                            overall_stderr.write(result.stderr)
                        details.append({{
                            "package": pkg,
                            "returncode": result.returncode,
                            "elapsed": elapsed
                        }})
                    except _original_subprocess.TimeoutExpired:
                        elapsed = __import__('time').time() - start_t
                        msg = f"Timeout after {{timeout_sec}}s: {{pkg}}"
                        log_debug(f"[INSTALL_ERR] {{msg}}")
                        details.append({{"package": pkg, "returncode": -1, "elapsed": elapsed, "error": msg}})
                    except Exception as run_e:
                        elapsed = __import__('time').time() - start_t
                        log_debug(f"[INSTALL_ERR] exception installing {{pkg}}: {{run_e}}")
                        details.append({{"package": pkg, "returncode": -2, "elapsed": elapsed, "error": str(run_e)}})
                    finally:
                        # 结束日志线程（如果已启动）
                        stop_tail = True
                        try:
                            if _tail_thread is not None:
                                _tail_thread.join(timeout=1.0)
                        except Exception:
                            pass

                if details and all(d.get("returncode") == 0 for d in details):
                    response = {{
                        "type": "INSTALL_SUCCESS",
                        "id": command_id,
                        "message": f"成功安装: {{', '.join(packages)}}",
                        "stdout": "",
                        "stderr": overall_stderr.getvalue(),
                        "details": details
                    }}
                else:
                    response = {{
                        "type": "INSTALL_ERROR",
                        "id": command_id,
                        "error": "Some packages failed",
                        "stdout": "",
                        "stderr": overall_stderr.getvalue(),
                        "details": details
                    }}
                send_response(response)
            except Exception as e:
                response = {{
                    "type": "INSTALL_ERROR",
                    "id": command_id,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }}
                send_response(response)
        
        elif command_type == "STATUS":
            global_vars = [k for k in PERSISTENT_GLOBALS.keys() if not k.startswith('_')]
            imported_modules = [name for name in sys.modules.keys() 
                            if not name.startswith('_') and '.' not in name]
            
            response = {{
                "type": "STATUS", 
                "id": command_id,
                "global_variables": global_vars,
                "imported_modules": imported_modules[:20],
                "total_modules": len(imported_modules)
            }}
            send_response(response)
        
        elif command_type == "PING":
            log_debug("Responding to PING")
            response = {{
                "type": "PONG",
                "id": command_id,
                "timestamp": __import__('time').time()
            }}
            send_response(response)
        
        else:
            response = {{
                "type": "ERROR", 
                "id": command_id,
                "error": f"未知命令类型: {{command_type}}"
            }}
            send_response(response)
            
    except EOFError:
        log_debug("EOF received, exiting")
        break
    except KeyboardInterrupt:
        log_debug("KeyboardInterrupt received")
        break
    except Exception as e:
        log_debug(f"Main loop error: {{e}}")
        response = {{
            "type": "FATAL_ERROR", 
            "id": "system",
            "error": str(e),
            "traceback": traceback.format_exc()
        }}
        send_response(response)
        break

log_debug("Persistent process exiting")
print(json.dumps({{"type": "EXIT", "session_id": "{self.session_id}"}}, ensure_ascii=False), flush=True)
    '''
        return script



    # Patterns for environment variable keys that may contain secrets.
    # Any key matching these (case-insensitive) will be stripped before
    # being passed to a sandbox child process.
    _SECRET_KEY_PATTERNS = (
        "API_KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL",
        "AUTH", "PRIVATE_KEY", "ACCESS_KEY", "CLIENT_ID", "CLIENT_SECRET",
        "SIGNING_KEY", "ENCRYPTION_KEY", "BEARER",
    )

    # Keys that are always safe to pass through to the child process.
    _ENV_WHITELIST = {
        "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL",
        "LC_CTYPE", "TERM", "TMPDIR", "TMP", "TEMP",
        "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED", "PYTHONPATH",
        "PYTHONHOME", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV",
        "MALLOC_ARENA_MAX", "CI", "PIP_NO_INPUT",
        "DEBIAN_FRONTEND", "GIT_TERMINAL_PROMPT",
        "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "no_proxy",
        "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_TRUSTED_HOST",
        "PIP_NO_CACHE_DIR", "AGENTGEN_PIP_NO_CACHE",
        "PERSISTENT_DEBUG", "PERSISTENT_STREAM_INSTALL_LOGS",
    }

    @classmethod
    def _is_secret_key(cls, key: str) -> bool:
        """Check if an environment variable key likely contains a secret."""
        upper = key.upper()
        for pat in cls._SECRET_KEY_PATTERNS:
            if pat in upper:
                return True
        return False

    def _get_process_env(self, additional_env: Dict[str, str] = None) -> Dict[str, str]:
        """获取统一的进程环境变量配置 — 过滤掉敏感信息"""
        # Start with a filtered copy of the system environment:
        # keep whitelisted keys and any key that does NOT look like a secret.
        env: Dict[str, str] = {}
        for k, v in os.environ.items():
            if k in self._ENV_WHITELIST or not self._is_secret_key(k):
                env[k] = v

        # 添加沙箱标准环境变量
        env.update({
            'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONUNBUFFERED': '1',
            'MALLOC_ARENA_MAX': '2',
        })

        # 如果有虚拟环境，确保PATH包含虚拟环境的bin目录
        if self.venv_path and os.path.exists(f"{self.venv_path}/bin"):
            venv_bin = f"{self.venv_path}/bin"
            current_path = env.get('PATH', '')
            if venv_bin not in current_path:
                env['PATH'] = f"{venv_bin}:{current_path}"

        # 应用额外的环境变量（可以覆盖上面的设置）
        if additional_env:
            env.update(additional_env)

        return env

    def _start_persistent_process(self):
        """启动持久化的Python进程"""
        if self.process and self.process.poll() is None:
            return
        
        with self.process_lock:
            # 创建venv
            if not self.venv_path:
                import venv
                self.venv_path = os.path.join(self.work_dir, "venv")
                if not os.path.exists(self.venv_path):
                    if self.debug:
                        print(f"[DEBUG] Creating venv at: {self.venv_path}")
                    try:
                        venv.create(self.venv_path, with_pip=True)
                    except Exception as e:
                        if self.debug:
                            print(f"[DEBUG] Venv creation failed: {e}")
                        self.venv_path = None
            
            if self.venv_path and os.path.exists(f"{self.venv_path}/bin/python"):
                python_bin = f"{self.venv_path}/bin/python"
            else:
                python_bin = sys.executable
                if self.debug:
                    print(f"[DEBUG] Using system Python: {python_bin}")
            
            script_content = self._create_persistent_script()
            
            # 使用统一的环境变量配置
            env = self._get_process_env()
            
            if self.debug:
                print(f"[DEBUG] Starting process with Python: {python_bin}")
                print(f"[DEBUG] Working directory: {self.work_dir}")
            
            try:
                self.process = subprocess.Popen(
                    [python_bin, '-c', script_content],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=0,
                    preexec_fn=os.setsid,
                    cwd=self.work_dir,
                    env=env
                )
                
                # 启动stderr监控线程
                def monitor_stderr():
                    while self.process and self.process.poll() is None:
                        try:
                            import select
                            ready, _, _ = select.select([self.process.stderr], [], [], 1.0)
                            if ready:
                                line = self.process.stderr.readline()
                                if line and self.debug:
                                    print(f"[PROCESS_STDERR] {line.rstrip()}")
                        except:
                            break
                
                stderr_thread = threading.Thread(target=monitor_stderr, daemon=True)
                stderr_thread.start()
                
                # 等待进程就绪 - 增加超时
                import select
                ready, _, _ = select.select([self.process.stdout], [], [], 30)
                
                if not ready:
                    # 检查stderr是否有错误信息
                    stderr_ready, _, _ = select.select([self.process.stderr], [], [], 0.1)
                    if stderr_ready:
                        error_msg = self.process.stderr.read()
                        raise RuntimeError(f"进程启动超时，stderr: {error_msg}")
                    else:
                        raise RuntimeError("进程启动超时，未收到READY信号")
                
                ready_line = self.process.stdout.readline().strip()
                if self.debug:
                    print(f"[DEBUG] Process ready line: {ready_line}")
                
                try:
                    ready_data = json.loads(ready_line)
                    if ready_data.get("type") != "READY":
                        raise RuntimeError(f"进程启动失败，收到: {ready_data}")
                    
                    if self.debug:
                        print(f"[PERSISTENT] Process started successfully for session {self.session_id}")
                        
                except json.JSONDecodeError as e:
                    # 可能是错误信息，读取stderr
                    stderr_output = ""
                    try:
                        import select
                        if select.select([self.process.stderr], [], [], 1)[0]:
                            stderr_output = self.process.stderr.read()
                    except:
                        pass
                    
                    raise RuntimeError(f"进程启动失败，JSON解析错误: {e}，stderr: {stderr_output}")
                    
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Process start failed: {e}")
                
                if self.process:
                    try:
                        self.process.terminate()
                        self.process.wait(timeout=5)
                    except:
                        try:
                            self.process.kill()
                        except:
                            pass
                    self.process = None
                raise RuntimeError(f"持久化进程启动失败: {e}")


    
    def _send_command(self, command: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
        """向持久化进程发送命令"""
        max_retries = 3
        
        for attempt in range(max_retries + 1):
            try:
                if not self.process or self.process.poll() is not None:
                    if self.debug:
                        print(f"[DEBUG] Starting new process for session {self.session_id} (attempt {attempt + 1})")
                    self._start_persistent_process()
                    import time
                    time.sleep(0.5)
                
                with self.process_lock:
                    self.command_counter += 1
                    command["id"] = f"cmd_{self.command_counter}"
                    if "timeout" not in command:
                        command["timeout"] = timeout
                    
                    try:
                        command_json = json.dumps(command, ensure_ascii=False)
                        if self.debug:
                            print(f"[DEBUG] Sending command: {command.get('type', 'UNKNOWN')} (attempt {attempt + 1})")
                            print(f"[DEBUG] Command JSON: {command_json[:200]}...")
                        
                        process_status = self.process.poll()
                        if process_status is not None:
                            raise RuntimeError(f"进程已退出，退出码: {process_status}")
                        
                        self.process.stdin.write(command_json + '\n')
                        self.process.stdin.flush()
                        
                        cmd_timeout = int(command.get("timeout", timeout))
                        max_wait_time = max(60, cmd_timeout + 30)
                        
                        import select
                        import os
                        import signal
                        import time
                        ready, _, _ = select.select([self.process.stdout], [], [], max_wait_time)
                        
                        if ready:
                            try:
                                response_data = b""
                                fd = self.process.stdout.fileno()
                                start_time = time.time()
                                while time.time() - start_time < max_wait_time:
                                    try:
                                        chunk = os.read(fd, 1024)
                                        if not chunk:
                                            if self.debug:
                                                print(f"[DEBUG] EOF received, data so far: {response_data}")
                                            break
                                        
                                        response_data += chunk
                                        if b'\n' in response_data:
                                            break
                                            
                                    except (OSError, IOError) as e:
                                        if self.debug:
                                            print(f"[DEBUG] Read error: {e}")
                                        break

                                    time.sleep(0.01)
                                
                                if not response_data:
                                    process_status = self.process.poll()
                                    
                                    error_details = {
                                        "session_id": self.session_id,
                                        "command_id": command.get("id"),
                                        "command_type": command.get("type"),
                                        "process_status": process_status,
                                        "process_pid": self.process.pid if self.process else None,
                                        "attempt": attempt + 1,
                                        "issue": "no_data_received"
                                    }
                                    
                                    if attempt < max_retries:
                                        if self.debug:
                                            print(f"[DEBUG] No data received, restarting process (attempt {attempt + 1})")
                                        self._force_restart_process()
                                        continue
                                    else:
                                        raise RuntimeError(f"未收到任何响应数据 - 详细信息: {json.dumps(error_details, ensure_ascii=False, indent=2)}")
                                
                                try:
                                    response_text = response_data.decode('utf-8', errors='replace')
                                except UnicodeDecodeError as e:
                                    if self.debug:
                                        print(f"[DEBUG] Unicode decode error: {e}")
                                        print(f"[DEBUG] Raw data: {response_data[:200]}")
                                    
                                    if attempt < max_retries:
                                        self._force_restart_process()
                                        continue
                                    else:
                                        raise RuntimeError(f"响应数据编码错误: {e}")
                                
                                if self.debug:
                                    print(f"[DEBUG] Raw response: {repr(response_text[:200])}")

                                lines = response_text.strip().split('\n')
                                response_line = None
                                
                                for line in lines:
                                    line = line.strip()
                                    if line and (line.startswith('{') or line.startswith('[')):
                                        response_line = line
                                        break
                                
                                if not response_line:
                                    if self.debug:
                                        print(f"[DEBUG] No valid JSON line found in: {lines}")
                                    
                                    if attempt < max_retries:
                                        if self.debug:
                                            print(f"[DEBUG] No valid JSON, restarting process (attempt {attempt + 1})")
                                        self._force_restart_process()
                                        continue
                                    else:
                                        error_details = {
                                            "session_id": self.session_id,
                                            "raw_response": response_text[:500],
                                            "lines_received": len(lines),
                                            "first_few_lines": lines[:5],
                                            "issue": "no_valid_json_line"
                                        }
                                        raise RuntimeError(f"响应中没有有效的JSON行 - 详细信息: {json.dumps(error_details, ensure_ascii=False, indent=2)}")
                                
                            except Exception as read_error:
                                if attempt < max_retries:
                                    if self.debug:
                                        print(f"[DEBUG] Read error, restarting process (attempt {attempt + 1}): {read_error}")
                                    self._force_restart_process()
                                    continue
                                else:
                                    raise read_error
                            
                            if self.debug:
                                print(f"[DEBUG] Parsing JSON: {response_line[:100]}...")
                            
                            try:
                                response = json.loads(response_line)
                                if self.debug:
                                    print(f"[DEBUG] Successfully parsed response: {response.get('type', 'UNKNOWN')}")
                                return response
                            except json.JSONDecodeError as e:
                                if self.debug:
                                    print(f"[DEBUG] JSON decode error: {e}")
                                    print(f"[DEBUG] Problematic line: {repr(response_line[:200])}")
                                
                                if attempt < max_retries:
                                    if self.debug:
                                        print(f"[DEBUG] JSON decode failed, restarting process (attempt {attempt + 1})")
                                    self._force_restart_process()
                                    continue
                                else:
                                    error_details = {
                                        "session_id": self.session_id,
                                        "json_error": str(e),
                                        "problematic_line": response_line[:200],
                                        "line_length": len(response_line),
                                        "issue": "json_decode_failed"
                                    }
                                    raise RuntimeError(f"响应JSON解析失败 - 详细信息: {json.dumps(error_details, ensure_ascii=False, indent=2)}")
                        else:
                            # 超时处理
                            if attempt < max_retries:
                                if self.debug:
                                    print(f"[DEBUG] Timeout, restarting process (attempt {attempt + 1})")
                                self._force_restart_process()
                                continue
                            else:
                                process_status = self.process.poll() if self.process else None
                                error_details = {
                                    "session_id": self.session_id,
                                    "command_id": command.get("id"),
                                    "timeout_seconds": max_wait_time,
                                    "process_status": process_status,
                                    "issue": "select_timeout"
                                }
                                raise RuntimeError(f"进程响应超时 - 详细信息: {json.dumps(error_details, ensure_ascii=False, indent=2)}")
                            
                    except Exception as e:
                        if attempt < max_retries:
                            if self.debug:
                                print(f"[DEBUG] Command execution failed, restarting process (attempt {attempt + 1}): {str(e)}")
                            self._force_restart_process()
                            continue
                        else:
                            raise e
            
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Outer exception caught (attempt {attempt + 1}): {str(e)}")
                
                if attempt < max_retries:
                    continue
                else:
                    return {
                        "type": "COMMUNICATION_ERROR",
                        "error": f"进程通信失败: {str(e)}",
                        "stdout": "",
                        "stderr": "",
                        "returncode": -1,
                        "session_id": self.session_id
                    }
        return {
            "type": "COMMUNICATION_ERROR",
            "error": "所有重试尝试都失败了",
            "stdout": "",
            "stderr": "",
            "returncode": -1,
            "session_id": self.session_id
        }

    def _force_restart_process(self):
        import os
        import signal
        import time
        
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except:
                try:
                    if hasattr(os, 'killpg'):
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    else:
                        self.process.kill()
                    time.sleep(0.5)
                except:
                    pass
        
        self.process = None
        time.sleep(0.2)


    def install_packages(self, packages: List[str], *, stream_logs: Optional[bool] = None, debug: Optional[bool] = None, timeout_sec: Optional[int] = None) -> Dict[str, Any]:
        """安装Python包 - 使用环境管理器优化"""
        if not packages:
            return {
                "success": True,
                "type": "SUCCESS", 
                "message": "无需安装包",
                "packages": []
            }
        
        # 获取Python可执行文件路径
        if self.venv_path and os.path.exists(f"{self.venv_path}/bin/python"):
            python_executable = f"{self.venv_path}/bin/python"
        else:
            python_executable = sys.executable
        
        # 使用环境管理器检查哪些包需要安装
        try:
            missing_packages = self._environment_manager.get_missing_packages(
                packages, self.venv_path or self.work_dir, python_executable
            )
            
            if not missing_packages:
                return {
                    "success": True,
                    "type": "SUCCESS", 
                    "message": f"所有包已安装，跳过: {', '.join(packages)}",
                    "packages": []
                }
            
            if self.debug:
                logger.info(f"[ENV] 需要安装的包: {missing_packages} (跳过已安装: {set(packages) - set(missing_packages)})")
        except Exception as e:
            logger.warning(f"环境检查失败，使用原始安装方式: {e}")
            missing_packages = packages
        
        max_mem_needed = self.base_mem_mb
        for pkg in missing_packages:
            pkg_name = pkg.split("==")[0].split(">=")[0].split("<=")[0]
            if pkg_name in HEAVY_PACKAGES:
                needed_mem = HEAVY_PACKAGES[pkg_name]
                max_mem_needed = max(max_mem_needed, needed_mem)
        
        if max_mem_needed > self.base_mem_mb:
            self.mem_bytes = max_mem_needed * 1024 * 1024
            if self.debug:
                print(f"[PERSISTENT] heavy packages → mem {max_mem_needed}MB")
        
        # 允许调用方控制调试与日志流
        if debug is None:
            debug = self.debug
        if stream_logs is None:
            _env_stream = os.environ.get("PERSISTENT_STREAM_INSTALL_LOGS")
            if _env_stream is not None:
                stream_logs = _env_stream.lower() not in ("0", "false", "no", "off")
            else:
                stream_logs = bool(debug)
        if timeout_sec is None:
            timeout_sec = self.wall_limit

        command = {
            "type": "INSTALL",
            "packages": missing_packages,
            "debug": bool(debug),
            "stream_logs": bool(stream_logs),
            "timeout": int(timeout_sec)
        }
        
        # Respect per-call timeout and leave room for parent/child protocol overhead.
        # Child process uses command["timeout"] for each pip invocation.
        wait_timeout = max(300, int(timeout_sec) + 30, int(timeout_sec) * max(1, len(missing_packages)) + 30)
        response = self._send_command(command, timeout=wait_timeout)
        
        # 如果安装成功，更新环境管理器
        if response.get("type") in ["SUCCESS", "INSTALL_SUCCESS"]:
            try:
                self._environment_manager.update_installed_packages(
                    missing_packages, self.venv_path or self.work_dir, python_executable
                )
            except Exception as e:
                logger.warning(f"更新环境缓存失败: {e}")
            
            # 统一返回格式，确保包含success字段
            return {
                "success": True,
                "type": response.get("type"),
                "message": response.get("message", "包安装成功"),
                "packages": missing_packages
            }
        
        # 安装失败
        return {
            "success": False,
            "type": response.get("type", "ERROR"),
            "message": response.get("message", "包安装失败"),
            "error": response.get("error", "未知错误")
        }
    
    def run_code(self, code: str, requirements: Optional[List[str]] = None, timeout_sec: int = 60, debug: Optional[bool] = None, stream_logs: Optional[bool] = None, **kwargs) -> Dict[str, Any]:
        """运行代码 - 兼容会话管理器接口"""
        result = self.run_code_original(
            code,
            env_requirements=requirements,
            debug=debug,
            stream_logs=stream_logs,
            timeout_sec=timeout_sec,
        )

        if requirements and result.get('success', False):
            self._installed_packages.update(requirements)
        
        return result
    
    def run_code_original(self, code: str, env_requirements: Optional[List[str]] = None, *, debug: Optional[bool] = None, stream_logs: Optional[bool] = None, timeout_sec: Optional[int] = None) -> Dict[str, Any]:
        """在持久化进程中运行代码 - 原始实现"""
        self.touch()
        # 允许调用方控制调试与日志流
        if debug is None:
            debug = self.debug
        if stream_logs is None:
            _env_stream = os.environ.get("PERSISTENT_STREAM_INSTALL_LOGS")
            if _env_stream is not None:
                stream_logs = _env_stream.lower() not in ("0", "false", "no", "off")
            else:
                stream_logs = bool(debug)
        if timeout_sec is None:
            timeout_sec = self.wall_limit
        
        # 1. 静态安全扫描
        scan_result = self.test_environment_safety(code)
        if not scan_result["safe"]:
            dangerous_issues = [issue for issue in scan_result["issues"] 
                            if "eval" in issue or "exec" in issue]
            if dangerous_issues:
                return {
                    "success": False,
                    "error": "安全检查未通过",
                    "issues": dangerous_issues,
                    "stdout": "",
                    "stderr": "",
                    "session_id": self.session_id,
                }
        
        # 2. 安装依赖
        if env_requirements:
            install_result = self.install_packages(env_requirements, stream_logs=stream_logs, debug=debug, timeout_sec=timeout_sec)
            if install_result.get("type") not in ["SUCCESS", "INSTALL_SUCCESS"]:
                return {
                    "success": False,
                    "error": f"依赖安装失败: {install_result.get('error', 'Unknown error')}",
                    "stdout": install_result.get("stdout", ""),
                    "stderr": install_result.get("stderr", ""),
                    "session_id": self.session_id,
                }
        
        # 3. 执行代码
        command = {
            "type": "EXEC",
            "code": code,
            "debug": bool(debug),
            "stream_logs": bool(stream_logs),
            "timeout": int(timeout_sec)
        }

        response = self._send_command(command, timeout=timeout_sec)
        
        # 4. 处理响应 - 添加对所有响应类型的支持
        if response.get("type") == "SUCCESS":
            return {
                "success": True,
                "stdout": response.get("stdout", ""),
                "stderr": response.get("stderr", ""),
                "returncode": 0,
                "session_id": self.session_id,
            }
        elif response.get("type") in ["ERROR", "FATAL_ERROR"]:
            return {
                "success": False,
                "error": response.get("error", "Unknown error"),
                "stdout": response.get("stdout", ""),
                "stderr": response.get("stderr", response.get("traceback", "")),
                "returncode": 1,
                "session_id": self.session_id,
            }
        elif response.get("type") == "COMMUNICATION_ERROR":
            return {
                "success": False,
                "error": response.get("error", "Communication failed"),
                "stdout": "",
                "stderr": "",
                "returncode": -1,
                "session_id": self.session_id,
            }
        elif response.get("type") == "EXEC_RESULT":
            # 处理bash命令的响应
            return {
                "success": response.get("returncode", -1) == 0,
                "stdout": response.get("stdout", ""),
                "stderr": response.get("stderr", ""),
                "returncode": response.get("returncode", -1),
                "session_id": self.session_id,
            }
        else:
            return {
                "success": False,
                "error": f"未知响应类型: {response.get('type')}",
                "stdout": "",
                "stderr": str(response),
                "returncode": -1,
                "session_id": self.session_id,
            }
    
    def save_file(self, relative_path: str, content: str):
        """
        Save content to a file relative to the sandbox's working directory.

        Args:
            relative_path (str): The relative file path.
            content (str): The content to write.

        Raises:
            Exception: If the path escapes the sandbox directory.
        """
        return self._workspace_manager.save_text(self._workspace_handle, relative_path, content)

    def read_file(self, relative_path: str) -> str:
        """
        Read file content from a path relative to the sandbox's working directory.

        Args:
            relative_path (str): The relative file path.

        Returns:
            str: The file content.

        Raises:
            Exception: If the path escapes the sandbox directory.
        """
        return self._workspace_manager.read_text(self._workspace_handle, relative_path)
    

    def _resolve_dest_in_sandbox(self, dest_relative: str) -> Path:
        if not self.work_dir:
            raise RuntimeError("Sandbox work_dir is not initialized.")
        base = Path(self.work_dir).resolve()
        dest = (base / dest_relative).resolve()
        if str(dest) != str(base) and not str(dest).startswith(str(base) + os.sep):
            raise PermissionError(f"Destination escapes sandbox: {dest}")
        return dest

    def _refresh_imports_in_child(self, add_path: Optional[str] = None) -> None:
        # Make current session see newly copied libs (and optionally add to sys.path)
        if not (self.process and self.process.poll() is None):
            return
        if add_path:
            code = (
                "import sys, os, importlib\n"
                f"p = os.path.abspath({repr(add_path)})\n"
                "if os.path.isfile(p): p = os.path.dirname(p)\n"
                "if p not in sys.path: sys.path.insert(0, p)\n"
                "importlib.invalidate_caches()\n"
            )
        else:
            code = "import importlib; importlib.invalidate_caches()\n"
        self._send_command({"type": "EXEC", "code": code}, timeout=5)

    def put_many_into_sandbox(
        self,
        sources: Sequence[os.PathLike | str],
        dest_relative: str,
        *,
        add_to_sys_path: bool = False,
        merge: bool = True,
        overwrite: bool = True,
        ignore_patterns: Optional[Sequence[str]] = ("__pycache__", "*.pyc", "*.pyo", ".git", ".mypy_cache"),
    ) -> List[str]:
        """
        批量复制文件/目录到沙箱，委托给 WorkspaceManager
        """
        copied_paths = self._workspace_manager.put_many(
            self._workspace_handle,
            [str(src) for src in sources],
            dest_relative,
            add_to_sys_path=False,  # 这里不直接处理sys.path
            merge=merge,
            overwrite=overwrite,
            ignore_patterns=ignore_patterns
        )
        
        # 处理sys.path逻辑（如果需要）
        if add_to_sys_path:
            rel_for_child = os.path.relpath(
                self._workspace_manager._resolve(self._workspace_handle, dest_relative),
                start=self.work_dir
            )
            self._refresh_imports_in_child(add_path=rel_for_child)
        else:
            self._refresh_imports_in_child(add_path=None)
        
        return copied_paths

    def put_into_sandbox(self, source: os.PathLike | str, dest_relative: str, **kwargs) -> List[str]:
        return self.put_many_into_sandbox([source], dest_relative, **kwargs)

    def import_file_map(
        self,
        file_map: dict[str, str],
        *,
        add_to_sys_path: bool = False,
        merge: bool = True,
    ) -> dict[str, Any]:
        """
        导入文件映射到沙箱中，委托给 WorkspaceManager
        """
        result = self._workspace_manager.import_file_map(
            self._workspace_handle,
            file_map,
            add_to_sys_path=False,  # 这里不直接处理sys.path
            merge=merge
        )
        
        # 处理sys.path逻辑（如果需要）
        if add_to_sys_path and result.get("success") and result.get("add_sys_paths"):
            for abs_path in result["add_sys_paths"]:
                rel_path = os.path.relpath(abs_path, start=self.work_dir)
                try:
                    self._refresh_imports_in_child(add_path=rel_path)
                except Exception as e:
                    if self.debug:
                        logger.warning(f"[PERSISTENT] 添加sys.path失败 {rel_path}: {e}")
        
        # 移除内部使用的add_sys_paths字段，保持向后兼容性
        if "add_sys_paths" in result:
            del result["add_sys_paths"]
        
        return result
    
    def evaluate_expression(self, expression: str) -> Dict[str, Any]:
        """求值表达式"""
        self.touch()
        
        command = {
            "type": "EVAL",
            "code": expression
        }
        
        response = self._send_command(command)
        
        if response.get("type") == "RESULT":
            return {
                "success": True,
                "result": response.get("value"),
                "session_id": self.session_id,
            }
        else:
            return {
                "success": False,
                "error": response.get("error", "Unknown error"),
                "session_id": self.session_id,
            }
    
    def get_status(self) -> Dict[str, Any]:
        """获取会话状态"""
        self.touch()
        
        command = {"type": "STATUS"}
        response = self._send_command(command)
        
        if response.get("type") == "STATUS":
            return {
                "success": True,
                "session_id": self.session_id,
                "global_variables": response.get("global_variables", []),
                "imported_modules": response.get("imported_modules", []),
                "total_modules": response.get("total_modules", 0),
                "created_at": self.created_at.isoformat(),
                "last_used": self.last_used.isoformat(),
                "time_remaining": self._get_time_remaining(),
            }
        else:
            return {
                "success": False,
                "error": response.get("error", "Failed to get status"),
                "session_id": self.session_id,
            }
    
    # Dangerous bare function names (direct calls like eval(...))
    _DANGEROUS_FUNC_NAMES = {"eval", "exec", "compile", "__import__", "breakpoint"}

    # Dangerous attribute patterns: (module, attr) pairs detected via ast.Attribute
    _DANGEROUS_ATTR_CALLS = {
        ("os", "system"), ("os", "popen"), ("os", "popen2"), ("os", "popen3"),
        ("os", "execv"), ("os", "execve"), ("os", "execvp"), ("os", "execvpe"),
        ("os", "execlp"), ("os", "execl"), ("os", "execle"),
        ("subprocess", "run"), ("subprocess", "Popen"), ("subprocess", "call"),
        ("subprocess", "check_call"), ("subprocess", "check_output"),
        ("subprocess", "getoutput"), ("subprocess", "getstatusoutput"),
        ("shutil", "rmtree"),
        ("importlib", "import_module"),
        ("ctypes", "cdll"), ("ctypes", "CDLL"),
    }

    def test_environment_safety(self, code: str) -> Dict[str, Any]:
        """静态安全扫描 — 基于 AST 检测危险调用模式"""
        issues: List[str] = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                # Direct name calls: eval(...), exec(...), __import__(...)
                if isinstance(func, ast.Name) and func.id in self._DANGEROUS_FUNC_NAMES:
                    issues.append(f"危险调用: {func.id}()")
                # Attribute calls: os.system(...), subprocess.run(...)
                elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                    pair = (func.value.id, func.attr)
                    if pair in self._DANGEROUS_ATTR_CALLS:
                        issues.append(f"危险调用: {func.value.id}.{func.attr}()")
                # Detect ctypes import as an attribute access
                if isinstance(node, ast.Call) and isinstance(func, ast.Attribute):
                    if func.attr in ("LoadLibrary", "dlopen"):
                        issues.append(f"危险调用: *.{func.attr}()")
        except SyntaxError as e:
            issues.append(f"语法错误: {e}")
        return {"safe": not issues, "issues": issues}
    
    def get_session_info(self) -> Dict[str, Any]:
        """获取会话信息 - 委托给会话管理器并添加沙箱特有信息"""
        session_info = self.session_manager.get_session_info()
        
        # 添加沙箱特有的信息
        session_info.update({
            "work_dir": self.work_dir,
            "has_venv": self.venv_path is not None,
            "process_running": self.process is not None and self.process.poll() is None,
            "allow_network": self.allow_network,
            "allow_subprocess": self.allow_subprocess,
            "extra_allowed_paths": list(self.extra_allowed_paths),
        })
        
        return session_info
    
    def _get_time_remaining(self) -> str:
        """获取剩余时间的字符串表示 - 委托给会话管理器"""
        remaining_minutes = self.session_manager.get_remaining_time()
        
        if remaining_minutes == float('inf'):
            return "unlimited"
        elif remaining_minutes <= 0:
            return "expired"
        
        minutes = int(remaining_minutes)
        seconds = int((remaining_minutes - minutes) * 60)
        return f"{minutes}:{seconds:02d}"
    
    def cleanup_session(self):
        """清理会话 - 停用会话管理器并清理资源"""
        if self.debug:
            logger.info(f"[PERSISTENT] Cleaning up session {self.session_id}")
        
        # 停用会话管理器
        self.session_manager.deactivate()
        
        # 更新本地状态以保持兼容性
        self._session_active = False

        # 清理MCP服务器
        if self._mcp_server_manager:
            try:
                self._mcp_server_manager.close()
            except Exception as e:
                if self.debug:
                    logger.error(f"[PERSISTENT] 清理MCP服务器失败: {e}")

        # 清理环境缓存（可选，通常不需要清理全局缓存）
        if hasattr(self, '_environment_manager') and self._environment_manager:
            try:
                # 只清理当前会话的虚拟环境相关缓存
                if self.venv_path:
                    self._environment_manager.clear_venv_cache(self.venv_path)
            except Exception as e:
                if self.debug:
                    logger.error(f"[PERSISTENT] 清理环境缓存失败: {e}")

        # 清理持久化进程
        if self.process:
            try:
                with self.process_lock:
                    self.process.stdin.write("EXIT\n")
                    self.process.stdin.flush()
                    self.process.wait(timeout=5)
            except Exception as e:
                logger.debug(f"[PERSISTENT] Graceful exit failed ({e}), escalating to SIGTERM")
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                    self.process.wait(timeout=2)
                except Exception as e2:
                    logger.debug(f"[PERSISTENT] SIGTERM failed ({e2}), escalating to SIGKILL")
                    try:
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, OSError) as e3:
                        logger.warning(f"[PERSISTENT] SIGKILL also failed: {e3}")
            finally:
                self.process = None

        # 使用 WorkspaceManager 处理工作区清理
        if hasattr(self, '_workspace_manager') and hasattr(self, '_workspace_handle'):
            try:
                if os.path.exists(self._workspace_handle.work_dir):
                    snapshot_meta = {
                        "session_id": self.session_id,
                        "venv_path": self.venv_path,
                        "installed_packages": list(self._installed_packages) if self._installed_packages else []
                    }
                    snapshot_path = self._workspace_manager.finalize(self._workspace_handle, snapshot_meta=snapshot_meta)
                    
                    if snapshot_path and self.debug:
                        logger.info(f"[PERSISTENT] 工作区快照已保存: {snapshot_path}")
                
                try:
                    self._workspace_manager.purge_snapshots()
                except Exception:
                    pass
                
                # 清理工作区
                self._workspace_manager.cleanup(self._workspace_handle)
                
            except Exception as e:
                if self.debug:
                    logger.error(f"[PERSISTENT] 工作区清理失败: {e}")
        else:
            # 兼容旧的清理方式（如果WorkspaceManager未初始化）
            if self.work_dir and os.path.exists(self.work_dir):
                try:
                    shutil.rmtree(self.work_dir)
                except Exception as e:
                    if self.debug:
                        logger.error(f"[PERSISTENT] Cleanup error: {e}")
    
    def _get_python_executable(self) -> str:
        """获取Python可执行文件路径"""
        if self.venv_path:
            # 虚拟环境中的Python
            if os.name == 'nt':  # Windows
                candidate = os.path.join(self.venv_path, 'Scripts', 'python.exe')
            else:  # Unix/Linux/macOS
                candidate = os.path.join(self.venv_path, 'bin', 'python')
            if os.path.exists(candidate):
                return candidate
        # 系统Python
        return sys.executable
    
    def get_package_cache_stats(self) -> Dict[str, Any]:
        """获取环境缓存统计信息"""
        if hasattr(self, '_environment_manager') and self._environment_manager:
            return self._environment_manager.get_cache_stats()
        return {"error": "Environment manager not initialized"}
    
    def clear_package_cache(self, venv_only: bool = True) -> Dict[str, Any]:
        """清理环境缓存"""
        if not hasattr(self, '_environment_manager') or not self._environment_manager:
            return {"success": False, "error": "Environment manager not initialized"}
        
        try:
            if venv_only and self.venv_path:
                self._environment_manager.clear_venv_cache(self.venv_path)
                return {"success": True, "message": f"Cleared cache for venv: {self.venv_path}"}
            else:
                self._environment_manager.clear_all_cache()
                return {"success": True, "message": "Cleared all environment cache"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def exec_bash(
        self,
        command: str,
        timeout: int = 30,
        capture_output: bool = True,
        env_requirements: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if self.is_timeout():
            return {
                'success': False,
                'error': 'Session timeout',
                'stdout': '',
                'stderr': '',
                'exit_code': -1,
                'session_id': self.session_id
            }      
        self.touch()
        try:
            venv_path = self.venv_path or ""
            python_code = f'''
import subprocess
import os
import sys
import json

try:
    os.chdir(r"{self.work_dir}")
    env = os.environ.copy()
    venv_path = r"{venv_path}"
    debug_enabled = {repr(bool(self.debug))}
    bash_command = {repr(command)}
    
    if venv_path and os.path.exists(venv_path):
        env['VIRTUAL_ENV'] = venv_path
        env['PYTHONHOME'] = ''  # 清除 PYTHONHOME 避免冲突

        venv_bin = os.path.join(venv_path, "bin")
        if os.path.exists(venv_bin):
            env['PATH'] = venv_bin + ":" + env.get('PATH', '')

        env['PYTHONPATH'] = r"{self.work_dir}"

        activate_script = os.path.join(venv_path, "bin", "activate")
        if os.path.exists(activate_script):
            bash_command = "source " + activate_script + " && " + bash_command
        
        if debug_enabled:
            print(f"DEBUG: Using venv: {{venv_path}}", file=sys.stderr)
            print(f"DEBUG: Activate script exists: {{os.path.exists(activate_script)}}", file=sys.stderr)
    else:
        env['PYTHONPATH'] = r"{self.work_dir}"
        if debug_enabled:
            print("DEBUG: No venv found, using system environment", file=sys.stderr)
    
    if debug_enabled:
        print(f"DEBUG: Final bash command: {{bash_command[:100]}}...", file=sys.stderr)
        print(f"DEBUG: VIRTUAL_ENV = {{env.get('VIRTUAL_ENV', 'Not set')}}", file=sys.stderr)
        print(f"DEBUG: PATH prefix = {{env.get('PATH', '')[:100]}}...", file=sys.stderr)

    env.setdefault("CI", "1")
    env.setdefault("PIP_NO_INPUT", "1")
    env.setdefault("DEBIAN_FRONTEND", "noninteractive")
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("SUDO_ASKPASS", "/usr/bin/false")
    env.setdefault("SSH_ASKPASS", "/usr/bin/false")

    result = subprocess.run(
        ["bash", "-c", bash_command],
        capture_output=True,
        text=True,
        timeout={timeout},
        env=env,
        cwd=r"{self.work_dir}",
        input="\\n",
    )

    output = {{
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.returncode
    }}
    
    print("__BASH_RESULT_START__")
    print(json.dumps(output))
    print("__BASH_RESULT_END__")
    
except subprocess.TimeoutExpired:
    output = {{
        "success": False,
        "stdout": "",
        "stderr": "Command timeout after {timeout}s",
        "exit_code": -1,
        "error": "Command timeout after {timeout}s"
    }}
    print("__BASH_RESULT_START__")
    print(json.dumps(output))
    print("__BASH_RESULT_END__")
    
except Exception as e:
    import traceback
    output = {{
        "success": False,
        "stdout": "",
        "stderr": f"Exception: {{str(e)}}\\n{{traceback.format_exc()}}",
        "exit_code": -1,
        "error": str(e)
    }}
    print("__BASH_RESULT_START__")
    print(json.dumps(output))
    print("__BASH_RESULT_END__")
'''
        
            if self.debug:
                print(f"[BASH] Executing in venv via Python sandbox: {command[:50]}...")
            
            python_result = self.run_code(python_code, env_requirements or [])
            
            if not python_result['success']:
                return {
                    'success': False,
                    'error': f"Python sandbox error: {python_result.get('error', 'Unknown error')}",
                    'stdout': '',
                    'stderr': python_result.get('stderr', ''),
                    'exit_code': -1,
                    'command': command,
                    'session_id': self.session_id
                }
            
            stdout = python_result.get('stdout', '')
            stderr = python_result.get('stderr', '')

            start_marker = "__BASH_RESULT_START__"
            end_marker = "__BASH_RESULT_END__"
            
            start_idx = stdout.find(start_marker)
            end_idx = stdout.find(end_marker)
            
            if start_idx == -1 or end_idx == -1:
                return {
                    'success': False,
                    'error': 'Failed to parse bash result',
                    'stdout': stdout,
                    'stderr': stderr,
                    'exit_code': -1,
                    'command': command,
                    'session_id': self.session_id
                }

            json_str = stdout[start_idx + len(start_marker):end_idx].strip()
            
            try:
                import json
                bash_result = json.loads(json_str)
                
                debug_info = stderr
                original_stderr = bash_result.get('stderr', '')
                combined_stderr = f"{debug_info}\n{original_stderr}".strip()
                out_stdout = bash_result.get('stdout', '')
                out_stderr = combined_stderr
                if not capture_output:
                    out_stdout = ""
                    if bash_result.get('success', False):
                        out_stderr = ""
                
                return {
                    'success': bash_result.get('success', False),
                    'stdout': out_stdout,
                    'stderr': out_stderr,
                    'exit_code': bash_result.get('exit_code', -1),
                    'error': bash_result.get('error'),
                    'command': command,
                    'session_id': self.session_id
                }
                
            except json.JSONDecodeError as e:
                return {
                    'success': False,
                    'error': f'Failed to parse JSON result: {e}',
                    'stdout': stdout,
                    'stderr': stderr,
                    'exit_code': -1,
                    'command': command,
                    'session_id': self.session_id
                }
        
        except Exception as e:
            return {
                'success': False,
                'error': f'Bash execution error: {str(e)}',
                'stdout': '',
                'stderr': str(e),
                'exit_code': -1,
                'command': command,
                'session_id': self.session_id
            }


        
    def check_venv_status(self) -> Dict[str, Any]:
        """检查虚拟环境状态"""
        if self.is_timeout():
            return {
                'success': False,
                'error': 'Session timeout',
                'type': 'TIMEOUT'
            }
        
        self.touch()
        
        try:
            command = {
                'type': 'STATUS',
                'command_id': str(uuid.uuid4())
            }
            
            result = self._send_command(command, timeout=15)
            
            if result.get('type') == 'STATUS':
                return {
                    'success': True,
                    'type': 'VENV_STATUS',
                    'python_version': result.get('python_version', 'Unknown'),
                    'pip_version': result.get('pip_version', 'Unknown'),
                    'installed_packages': result.get('installed_packages', []),
                    'venv_path': result.get('venv_path', 'Unknown')
                }
            else:
                return {
                    'success': False,
                    'error': result.get('error', 'Unknown error'),
                    'type': 'VENV_ERROR'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'type': 'EXCEPTION'
            }

    # ========== 子进程管理API ==========
    
    def get_process_env(self, env: Dict[str, str] = None) -> Dict[str, str]:
        """获取进程环境变量配置（暴露给外部调用）"""
        return self._get_process_env(env)

    def _is_portal_not_running(self, exc: Exception) -> bool:
        try:
            return "portal is not running" in str(exc).lower()
        except Exception:
            return False

    def _reset_mcp_server_manager(self) -> None:
        try:
            if self._mcp_server_manager:
                self._mcp_server_manager.close()
        except Exception:
            pass
        self._mcp_server_manager = MCPServerManagerSync()

    def _ensure_mcp_server_manager(self) -> None:
        if self._mcp_server_manager is None:
            self._mcp_server_manager = MCPServerManagerSync()
            return
        try:
            self._mcp_server_manager.servers()
        except Exception as e:
            if self._is_portal_not_running(e):
                self._reset_mcp_server_manager()
            else:
                raise

    def create_mcp_server(self, config: MCPServerConfig) -> Dict[str, Any]:
        """创建MCP服务器"""
        try:
            self._ensure_mcp_server_manager()
            tools = self._mcp_server_manager.start_server(config=config)
            return {
                "success": True,
                "name": config.name,
                "tools": tools,
                "message": f"MCP服务器 '{config.name}' 创建成功，包含 {len(tools)} 个工具"
            }
        except Exception as e:
            if self._is_portal_not_running(e):
                try:
                    self._reset_mcp_server_manager()
                    tools = self._mcp_server_manager.start_server(config=config)
                    return {
                        "success": True,
                        "name": config.name,
                        "tools": tools,
                        "message": f"MCP服务器 '{config.name}' 创建成功，包含 {len(tools)} 个工具"
                    }
                except Exception as e2:
                    return {
                        "success": False,
                        "name": config.name,
                        "error": f"创建MCP服务器失败: {str(e2)}"
                    }
            return {
                "success": False,
                "name": config.name,
                "error": f"创建MCP服务器失败: {str(e)}"
            }

    def stop_mcp_server(self, name: str) -> Dict[str, Any]:
        """停止MCP服务器"""
        try:
            self._ensure_mcp_server_manager()
            success = self._mcp_server_manager.stop_server(name)
            return {
                "success": success,
                "name": name,
                "message": f"服务器 '{name}' 已停止" if success else f"服务器 '{name}' 停止失败"
            }
        except Exception as e:
            if self._is_portal_not_running(e):
                try:
                    self._reset_mcp_server_manager()
                    success = self._mcp_server_manager.stop_server(name)
                    return {
                        "success": success,
                        "name": name,
                        "message": f"服务器 '{name}' 已停止" if success else f"服务器 '{name}' 停止失败"
                    }
                except Exception as e2:
                    return {
                        "success": False,
                        "name": name,
                        "error": f"停止服务器失败: {str(e2)}"
                    }
            return {
                "success": False,
                "name": name,
                "error": f"停止服务器失败: {str(e)}"
            }

    def restart_mcp_server(self, name: str) -> Dict[str, Any]:
        """重启MCP服务器"""
        try:
            self._ensure_mcp_server_manager()
            self._mcp_server_manager.restart_server(name)
            return {
                "success": True,
                "name": name,
                "message": f"服务器 '{name}' 已重启"
            }
        except Exception as e:
            if self._is_portal_not_running(e):
                try:
                    self._reset_mcp_server_manager()
                    self._mcp_server_manager.restart_server(name)
                    return {
                        "success": True,
                        "name": name,
                        "message": f"服务器 '{name}' 已重启"
                    }
                except Exception as e2:
                    return {
                        "success": False,
                        "name": name,
                        "error": f"重启服务器失败: {str(e2)}"
                    }
            return {
                "success": False,
                "name": name,
                "error": f"重启服务器失败: {str(e)}"
            }

    def call_mcp_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float] = 30.0,
    ) -> Dict[str, Any]:
        """调用MCP服务器工具"""
        try:
            self._ensure_mcp_server_manager()
            return self._mcp_server_manager.call_tool(server_name, tool_name, arguments, timeout)
        except Exception as e:
            if self._is_portal_not_running(e):
                try:
                    self._reset_mcp_server_manager()
                    return self._mcp_server_manager.call_tool(server_name, tool_name, arguments, timeout)
                except Exception as e2:
                    return {
                        "success": False,
                        "error": str(e2),
                        "server_name": server_name,
                        "tool_name": tool_name
                    }
            return {
                "success": False,
                "error": str(e),
                "server_name": server_name,
                "tool_name": tool_name
            }

    def list_mcp_server_tools(self, server_name: str, public_only: bool = True) -> Dict[str, Any]:
        """列出MCP服务器工具"""
        try:
            self._ensure_mcp_server_manager()
            tools = self._mcp_server_manager.list_tools(server_name, public_only)
            return {
                "success": True,
                "tools": tools,
                "count": len(tools),
                "server_name": server_name
            }
        except Exception as e:
            if self._is_portal_not_running(e):
                try:
                    self._reset_mcp_server_manager()
                    tools = self._mcp_server_manager.list_tools(server_name, public_only)
                    return {
                        "success": True,
                        "tools": tools,
                        "count": len(tools),
                        "server_name": server_name
                    }
                except Exception as e2:
                    return {
                        "success": False,
                        "error": str(e2),
                        "tools": [],
                        "count": 0,
                        "server_name": server_name
                    }
            return {
                "success": False,
                "error": str(e),
                "tools": [],
                "count": 0,
                "server_name": server_name
            }

    def list_mcp_servers(self) -> Dict[str, Any]:
        """列出所有MCP服务器"""
        try:
            self._ensure_mcp_server_manager()
            server_names = self._mcp_server_manager.servers()
            servers = [{"name": name, "status": "running"} for name in server_names]
            return {
                "success": True,
                "servers": servers,
                "count": len(servers)
            }
        except Exception as e:
            if self._is_portal_not_running(e):
                try:
                    self._reset_mcp_server_manager()
                    server_names = self._mcp_server_manager.servers()
                    servers = [{"name": name, "status": "running"} for name in server_names]
                    return {
                        "success": True,
                        "servers": servers,
                        "count": len(servers)
                    }
                except Exception as e2:
                    return {
                        "success": False,
                        "error": f"获取服务器列表失败: {str(e2)}",
                        "servers": [],
                        "count": 0
                    }
            return {
                "success": False,
                "error": f"获取服务器列表失败: {str(e)}",
                "servers": [],
                "count": 0
            }

    def get_mcp_server_status(self, name: str) -> Dict[str, Any]:
        """获取MCP服务器状态"""
        try:
            self._ensure_mcp_server_manager()
            # 使用manager的get_status方法真正检查服务器运行状态
            is_running = self._mcp_server_manager.get_status(name)
            if is_running:
                return {
                    "success": True,
                    "name": name,
                    "status": "running"
                }
            else:
                # 检查服务器是否存在于注册列表中
                server_names = self._mcp_server_manager.servers()
                if name in server_names:
                    return {
                        "success": False,
                        "error": f"服务器 '{name}' 已停止或无响应",
                        "name": name,
                        "status": "stopped"
                    }
                else:
                    return {
                        "success": False,
                        "error": f"服务器 '{name}' 不存在",
                        "name": name,
                        "status": "not_found"
                    }
        except Exception as e:
            if self._is_portal_not_running(e):
                try:
                    self._reset_mcp_server_manager()
                    is_running = self._mcp_server_manager.get_status(name)
                    if is_running:
                        return {
                            "success": True,
                            "name": name,
                            "status": "running"
                        }
                    server_names = self._mcp_server_manager.servers()
                    if name in server_names:
                        return {
                            "success": False,
                            "error": f"服务器 '{name}' 已停止或无响应",
                            "name": name,
                            "status": "stopped"
                        }
                    return {
                        "success": False,
                        "error": f"服务器 '{name}' 不存在",
                        "name": name,
                        "status": "not_found"
                    }
                except Exception as e2:
                    return {
                        "success": False,
                        "error": f"获取服务器状态失败: {str(e2)}",
                        "name": name,
                        "status": "error"
                    }
            return {
                "success": False,
                "error": f"获取服务器状态失败: {str(e)}",
                "name": name,
                "status": "error"
            }

    def save_environment_template(self, template_name: str, description: str = "") -> Dict[str, Any]:
        """
        保存当前环境为模板
        
        Args:
            template_name: 模板名称
            description: 模板描述
            
        Returns:
            Dict包含操作结果和模板信息
        """
        if not self.venv_path or not os.path.exists(self.venv_path):
            return {
                "success": False,
                "error": "虚拟环境不存在，无法保存模板"
            }
        
        result = self._environment_manager.save_environment_template_with_result(
            self.venv_path,
            template_name,
            description
        )
        
        if result.get('success', False) and self.debug:
            logger.info(f"[PERSISTENT] Session {self.session_id} saved template: {template_name}")
        
        return result

    def load_environment_template(self, template_id: str) -> Dict[str, Any]:
        """
        从模板加载环境
        
        Args:
            template_id: 模板ID
            
        Returns:
            Dict包含操作结果
        """
        # 确保有venv_path
        if not self.venv_path:
            self.venv_path = os.path.join(self.work_dir, "venv")
        
        result = self._environment_manager.load_environment_template_with_result(template_id, self.venv_path)
        
        if result.get('success', False):
            # Fix shebangs in venv scripts to ensure they point to the correct python interpreter
            try:
                bin_path = os.path.join(self.venv_path, "bin")
                if os.path.isdir(bin_path):
                    # First, determine the correct path to the python executable in the new venv
                    new_python_path = os.path.join(bin_path, "python")
                    if not os.path.exists(new_python_path):
                        # In some systems or venv configurations, the executable might be python3
                        new_python_path = os.path.join(bin_path, "python3")

                    # If a python executable is found, proceed to fix the shebangs
                    if os.path.exists(new_python_path):
                        new_shebang = f"#!{new_python_path}"
                        for filename in os.listdir(bin_path):
                            filepath = os.path.join(bin_path, filename)
                            # Only process files, and ignore symlinks to avoid issues
                            if not os.path.islink(filepath) and os.path.isfile(filepath):
                                try:
                                    with open(filepath, "r+", encoding="utf-8", errors="ignore") as f:
                                        first_line = f.readline()
                                        # Check if the file is a script with a python shebang
                                        if first_line.startswith("#!") and "python" in first_line:
                                            # If the shebang is incorrect, rewrite the file with the correct one
                                            if new_shebang not in first_line:
                                                rest_of_content = f.read()
                                                f.seek(0)
                                                f.write(new_shebang + "\n")
                                                f.write(rest_of_content)
                                                f.truncate()
                                except Exception:
                                    # Ignore errors for files that can't be read/written (e.g. binary files)
                                    pass
            except Exception as e:
                # Log any unexpected errors during the shebang fixing process
                logger.error(f"[PERSISTENT] Failed to fix venv shebangs for session {self.session_id}: {e}")

            # 重启持久化进程以使用新的虚拟环境
            if self.process:
                try:
                    with self.process_lock:
                        self.process.terminate()
                        self.process.wait(timeout=5)
                except:
                    pass
                self.process = None
            
            if self.debug:
                logger.info(f"[PERSISTENT] Session {self.session_id} loaded template: {template_id}")
        
        return result

    def list_environment_templates(self) -> Dict[str, Any]:
        """
        列出所有可用的环境模板
        
        Returns:
            Dict包含模板列表
        """
        return self._environment_manager.list_environment_templates_with_result()
    
    def __del__(self):
        """自动清理资源"""
        try:
            if hasattr(self, 'session_id') and self.session_id:
                self.cleanup_session()
        except Exception:
            # 忽略清理过程中的异常，避免在程序退出时出现错误
            pass

class EnvironmentSandbox(PersistentEnvironmentSandbox):
    def __init__(self, **kwargs):
        kwargs['timeout_minutes'] = 0
        super().__init__(**kwargs)
     
    def run_environment(self, code: str, env_requirements: Optional[List[str]] = None) -> Dict[str, Any]:
        try:
            return self.run_code(code, env_requirements)
        finally:
            self.cleanup_session()
