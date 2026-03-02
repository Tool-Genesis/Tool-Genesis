"""
MCP Sandbox Bridge - MCP server management wrapper (supports batch creation)

This module provides a bridge layer that maps MCP server management
from persistent_sandbox to an external interface focused on MCP operations.

Highlights:
- create_mcp_server supports both single and multiple configurations.
- Single configuration: keeps the original return structure (backward compatible).
- Multiple configurations: returns aggregated results (success/created/failed/results/errors).
"""

import json
from loguru import logger
import os
import glob
import time
import re
import shlex
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Iterable, Tuple
from dataclasses import dataclass
from collections import defaultdict
from threading import Lock

from .persistent_sandbox import PersistentEnvironmentSandbox
from .managers.mcp_server_manager import MCPServerConfig

ServerConfigLike = Union[MCPServerConfig, Dict[str, Any], str]

import base64
from typing import Any, Dict, List, Set, Tuple, Union

_TEMPLATE_SETUP_LOCK = Lock()


class LogTruncator:
    @staticmethod
    def truncate_json_for_log(
        value: Any,
        max_total_chars: int,
        max_string_chars: int = 2000,
        max_depth: int = 12,
        max_container_items: int = 2000,
        bytes_mode: str = "base64",  # "base64" or "repr"
    ) -> Any:
        """
        Return a JSON-serializable object (dict/list/str/int/float/bool/None) for safe logging.
        Never raises; best-effort truncation by total char budget + per-string cap.

        - Prevents recursion cycles
        - Depth-limited
        - Container item limited
        - Converts non-serializable objects to safe strings
        """

        budget = [max(0, int(max_total_chars))]
        seen: Set[int] = set()

        def _safe_str(x: Any) -> str:
            try:
                return str(x)
            except Exception as e:
                return f"<unstringifiable {type(x).__name__}: {e.__class__.__name__}>"

        def _take_string(s: str) -> str:
            # Per-string cap
            if max_string_chars > 0 and len(s) > max_string_chars:
                omitted = len(s) - max_string_chars
                s = s[:max_string_chars] + f"... [{omitted} chars omitted]"

            # Total budget cap
            if budget[0] <= 0:
                return "... [log truncated: budget exhausted]"

            if len(s) > budget[0]:
                omitted_over = len(s) - budget[0]
                s = s[:budget[0]] + f"... [truncated, {omitted_over} chars over budget]"
                budget[0] = 0
                return s

            budget[0] -= len(s)
            return s

        def _exhausted_marker() -> str:
            return "... [log truncated: budget exhausted]"

        def _helper(v: Any, depth: int) -> Any:
            # Budget check first
            if budget[0] <= 0:
                return _exhausted_marker()

            # Depth limit
            if depth > max_depth:
                return _take_string(f"... [log truncated: max_depth={max_depth} reached]")

            # Primitives (JSON-safe)
            if v is None or isinstance(v, (bool, int, float)):
                return v

            # String
            if isinstance(v, str):
                return _take_string(v)

            # Bytes
            if isinstance(v, (bytes, bytearray, memoryview)):
                b = bytes(v)
                if bytes_mode == "base64":
                    b64 = base64.b64encode(b).decode("ascii")
                    return _take_string(f"<bytes len={len(b)} base64={b64}>")
                return _take_string(repr(b))

            # Detect cycles for container-ish objects
            obj_id = id(v)
            is_container = isinstance(v, (dict, list, tuple, set))
            if is_container:
                if obj_id in seen:
                    return _take_string("... [cycle detected]")
                seen.add(obj_id)

            # dict
            if isinstance(v, dict):
                out: Dict[str, Any] = {}
                try:
                    items = list(v.items())
                except Exception as e:
                    return _take_string(f"<dict items() failed: {e.__class__.__name__}>")

                # limit items
                if len(items) > max_container_items:
                    items = items[:max_container_items]
                    out["__log_note__"] = _take_string(f"... [{len(v) - max_container_items} dict items omitted]")

                for k, vv in items:
                    if budget[0] <= 0:
                        out["__log_truncated__"] = _exhausted_marker()
                        break
                    # JSON requires string keys
                    key = _safe_str(k)
                    out[key] = _helper(vv, depth + 1)

                return out

            # list/tuple/set
            if isinstance(v, (list, tuple, set)):
                try:
                    seq = list(v)
                except Exception as e:
                    return _take_string(f"<sequence iter failed: {e.__class__.__name__}>")

                # limit items
                omitted_n = 0
                if len(seq) > max_container_items:
                    omitted_n = len(seq) - max_container_items
                    seq = seq[:max_container_items]

                out_list: List[Any] = []
                for idx, vv in enumerate(seq):
                    if budget[0] <= 0:
                        out_list.append(_exhausted_marker())
                        break
                    out_list.append(_helper(vv, depth + 1))

                if omitted_n > 0 and budget[0] > 0:
                    out_list.append(_take_string(f"... [{omitted_n} items omitted]"))

                return out_list

            # Try common "to dict" protocols (best-effort, never throw)
            # pydantic v2: model_dump()
            if hasattr(v, "model_dump"):
                try:
                    dumped = v.model_dump()
                    return _helper(dumped, depth + 1)
                except Exception:
                    pass

            # dataclasses: __dict__ (careful: can be huge)
            if hasattr(v, "__dict__"):
                try:
                    d = dict(getattr(v, "__dict__", {}))
                    return _helper({"__type__": type(v).__name__, **d}, depth + 1)
                except Exception:
                    pass

            # Fallback: safe string summary (JSON-safe)
            return _take_string(_safe_str(v))

        try:
            return _helper(value, depth=0)
        except Exception as e:
            # Absolute last-resort: never crash logging
            return {"__log_truncator_error__": f"{e.__class__.__name__}: {e}"}


class MCPSandboxBridge:
    """
    MCP Sandbox bridge layer

    Provides a wrapper around MCP server management in persistent_sandbox
    and focuses on MCP-related operations.
    """

    def __init__(self, sandbox: PersistentEnvironmentSandbox, registry_path: Optional[str] = None):
        """
        Initialize the MCP Sandbox bridge layer

        Args:
            sandbox: PersistentEnvironmentSandbox instance
        """
        self.sandbox = sandbox
        self.registry_path = registry_path
        self._template_ready = False
        self._template_id = None
        logger.info("MCPSandboxBridge initialized")

    # === Private Utilities ===
    @staticmethod
    def _is_standard_library(package_name: str) -> bool:
        """
        Check whether a package is part of the Python standard library
        
        Args:
            package_name: Package name
            
        Returns:
            True if it is a standard library module, otherwise False
        """
        # Common Python standard libraries
        standard_libraries = {
            'os', 'sys', 'json', 'logging', 'pathlib', 'typing', 'dataclasses',
            'collections', 'datetime', 'hashlib', 'subprocess', 'shutil', 'tempfile',
            'tarfile', 'threading', 'venv', 're', 'glob', 'time', 'math', 'random',
            'string', 'itertools', 'functools', 'operator', 'copy', 'pickle',
            'base64', 'urllib', 'http', 'email', 'html', 'xml', 'csv', 'configparser',
            'argparse', 'getopt', 'unittest', 'doctest', 'pdb', 'profile', 'timeit',
            'trace', 'gc', 'weakref', 'abc', 'contextlib', 'warnings', 'inspect',
            'dis', 'ast', 'keyword', 'token', 'tokenize', 'parser', 'symbol',
            'compiler', 'py_compile', 'compileall', 'dis', 'pickletools'
        }
        
        # Remove version suffixes and operators
        clean_name = package_name.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0].strip()
        
        return clean_name.lower() in standard_libraries

    @staticmethod
    def _as_cfg(obj: ServerConfigLike) -> MCPServerConfig:
        """Normalize dict/MCPServerConfig/str into MCPServerConfig."""
        if isinstance(obj, MCPServerConfig):
            return obj
        if isinstance(obj, str):
            # Path/command script provided directly; use filename as server name
            inferred_name = Path(obj).stem or "mcp_server"
            return MCPServerConfig(name=inferred_name, server_config=obj)
        if isinstance(obj, dict):
            server_config = obj.get("server_config")
            if server_config is None:
                # Support common field aliases
                server_config = obj.get("config") or obj.get("config_path") or obj.get("path")
            if server_config is None:
                raise KeyError("MCP server config dict must include 'server_config' or an equivalent key")

            name = obj.get("name")
            if not name:
                if isinstance(server_config, str):
                    name = Path(server_config).stem
                else:
                    raise KeyError("MCP server config dict requires 'name' when server_config is not a path string")

            return MCPServerConfig(
                name=name,
                server_config=server_config,
                working_dir=obj.get("working_dir"),
                env=obj.get("env"),
            )
        raise TypeError(f"Unsupported config type: {type(obj)}")

    @staticmethod
    def _ensure_list(
        x: Union[ServerConfigLike, Iterable[ServerConfigLike]]
    ) -> List[ServerConfigLike]:
        """Normalize a single item or iterable to a list (strings treated as scalars)."""
        if isinstance(x, (MCPServerConfig, dict, str)):
            return [x]
        return list(x)


    # === MCP Server Management ===
    def _prepare_server_configs(
        self,
        server_names: Union[str, List[str]],
        init_states: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        registry_path: str = None,
        env: Dict[str, str] = None,
    ) -> Tuple[bool, Union[List, str]]:
        """
        Prepare the server configuration list
        
        Args:
            server_names: Single server name or a list of server names
            init_states: Initial state data; either a dict or a list of dicts
            registry_path: Path to the registry.json file
            env: Environment variables
            
        Returns:
            Tuple[bool, Union[List[MCPServerConfig], str]]: (success flag, configs or error message)
        """
        # Ensure server_names is a list
        if isinstance(server_names, str):
            server_names = [server_names]
            
        # Load registry configuration
        if registry_path is None:
            registry_path = Path("temp/run_benchmark/registry.json")
        try:
            with open(registry_path, 'r', encoding='utf-8') as f:
                registry = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load registry from {registry_path}: {e}")
            return False, f"Failed to load registry file: {str(e)}"

        # Check that all servers exist in the registry
        missing_servers = [name for name in server_names if name not in registry]
        if missing_servers:
            return False, f"Servers not found in registry: {', '.join(missing_servers)}"

        # Get environment variables
        try:
            full_env = self.sandbox.get_process_env(env)
        except Exception as e:
            return False, f"Failed to obtain environment variables: {str(e)}"
        try:
            if self.sandbox and getattr(self.sandbox, "work_dir", None):
                full_env["MCP_SANDBOX_WORKDIR"] = self.sandbox.work_dir
        except Exception:
            pass

        # Find and install dependencies per server (requirements.txt)
        logger.info("Installing dependencies...")
        
        try:
            project_root = Path(__file__).parent.parent.parent.parent
            # Control internal logging and progress via environment variables
            def _env_bool(name: str, default: str = "1") -> bool:
                val = os.environ.get(name, default)
                return str(val).lower() not in ("0", "false", "no", "off")

            dep_logs_enabled = _env_bool("MCP_DEP_LOGS", "1")
            dep_progress_enabled = _env_bool("MCP_DEP_PROGRESS", "0")
            dep_debug_enabled = _env_bool("MCP_DEP_DEBUG", "0")
            base_candidates = [
                project_root / "temp" / "agentic" / "envs"
            ]

            for server_name in server_names:
                req_file: Optional[Path] = None
                # Search by convention: <base>/<server_name>/requirements.txt
                for base in base_candidates:
                    candidate = base / server_name / "requirements.txt"
                    if candidate.exists():
                        req_file = candidate
                        break

                if req_file and req_file.exists():
                    try:
                        # Read raw lines for later writing back pinned versions
                        with open(req_file, "r", encoding="utf-8") as f:
                            raw_lines = f.readlines()

                        # Filter valid dependency lines (remove blanks/comments; keep content before inline comments)
                        packages: List[str] = []
                        for line in raw_lines:
                            stripped = line.strip()
                            if not stripped or stripped.startswith("#"):
                                continue
                            no_comment = stripped.split("#", 1)[0].strip()
                            if no_comment:
                                packages.append(no_comment)

                        if dep_logs_enabled:
                            logger.info(f"Preparing to install {len(packages)} dependencies for server {server_name}: {req_file}")

                        def _parse_spec(spec: str):
                            # Extract name[extras], operator, version, environment marker (; suffix)
                            marker_split = spec.split(";", 1)
                            base = marker_split[0].strip()
                            marker = marker_split[1] if len(marker_split) > 1 else None

                            m = re.match(r"^([A-Za-z0-9_.\-]+(?:\[[^\]]+\])?)(==|>=|<=|!=|~=)?\s*([^\s;#]+)?", base)
                            if m:
                                name_extras = m.group(1)
                                op = m.group(2)
                                ver = m.group(3)
                            else:
                                name_extras, op, ver = base, None, None

                            name_only = re.sub(r"\[.*\]", "", name_extras)
                            return name_extras, name_only, op, ver, marker

                        def _pip_install_one(spec: str, timeout: int = 60) -> bool:
                            start_ts = time.time()
                            if dep_progress_enabled or dep_logs_enabled:
                                logger.info(f"Starting install: {spec}")
                            # Use the persistent sandbox install API with streaming logs and debug
                            try:
                                res = self.sandbox.install_packages(
                                    [spec], timeout_sec=timeout,
                                    stream_logs=dep_progress_enabled,
                                    debug=dep_debug_enabled
                                )
                                elapsed = time.time() - start_ts
                                if res.get("success", False):
                                    if dep_logs_enabled:
                                        logger.info(f"pip install succeeded: {spec} | elapsed {elapsed:.1f}s")
                                    return True
                                else:
                                    last_err = res.get("error", "")
                                    logger.warning(f"pip install failed: {spec} | elapsed {elapsed:.1f}s | error: {last_err}")
                                    return False
                            except Exception as e:
                                elapsed = time.time() - start_ts
                                logger.warning(f"pip install exception: {spec} | elapsed {elapsed:.1f}s | error: {e}")
                                return False

                        installed_cache: Optional[Dict[str, str]] = None
                        python_exe_cached: Optional[str] = None

                        def _get_installed_version(name: str) -> Optional[str]:
                            # Query installed packages via environment manager directly to avoid blocking
                            nonlocal installed_cache, python_exe_cached
                            try:
                                if python_exe_cached is None:
                                    python_exe_cached = self.sandbox._get_python_executable()
                                if not python_exe_cached:
                                    logger.warning("Unable to get sandbox Python executable; skip version query")
                                    return None
                                if installed_cache is None:
                                    installed_cache = self.sandbox._environment_manager._get_installed_packages(python_exe_cached)
                                return installed_cache.get(name)
                            except Exception as e:
                                logger.warning(f"Failed to read installed version: {name} | error: {e}")
                                return None

                        def _build_specs(name_extras: str, op: Optional[str], ver: Optional[str], marker: Optional[str]):
                            """Construct install strings with/without version while preserving marker"""
                            base_spec = name_extras
                            versioned_spec = None
                            if op and ver:
                                versioned_spec = f"{name_extras}{op}{ver}"
                            if marker:
                                base_spec = f"{base_spec};{marker}"
                                versioned_spec = f"{versioned_spec};{marker}" if versioned_spec else None
                            return versioned_spec, base_spec

                        updated_specs: List[str] = []
                        if packages:
                            for spec in packages:
                                if dep_progress_enabled:
                                    logger.info(f"Processing: {spec}")
                                name_extras, name_only, op, ver, marker = _parse_spec(spec)

                                # Preserve marker and construct install strings
                                versioned_spec, base_spec = _build_specs(name_extras, op, ver, marker)

                                # Try version-pinned install first (if available); on failure, retry without version but keep marker
                                ok = _pip_install_one(versioned_spec) if versioned_spec else _pip_install_one(base_spec)
                                if not ok and versioned_spec:
                                    ok = _pip_install_one(base_spec)

                                if ok:
                                    # Optionally freeze installed versions to avoid long-term drift
                                    freeze_versions = os.environ.get("MCP_FREEZE_REQUIREMENTS", "1").lower() not in ("0", "false", "no", "off")
                                    if freeze_versions:
                                        installed_ver = _get_installed_version(name_only)
                                        if installed_ver:
                                            new_line = f"{name_extras}=={installed_ver}"
                                            if marker:
                                                new_line = f"{new_line};{marker}"
                                            updated_specs.append(new_line)
                                        else:
                                            # Keep original line if version cannot be obtained
                                            updated_specs.append(spec)
                                    else:
                                        updated_specs.append(spec)
                                else:
                                    logger.warning(f"Dependency installation failed for server {server_name} (version removed when retry): {spec}")
                                    updated_specs.append(spec)

                            # Write back requirements.txt with pinned versions after installation
                            try:
                                with open(req_file, "w", encoding="utf-8") as wf:
                                    wf.write("\n".join(updated_specs) + "\n")
                                if dep_logs_enabled:
                                    logger.info(f"Updated requirements.txt for {server_name}, {len(updated_specs)} entries: {req_file}")
                            except Exception as write_exc:
                                logger.warning(f"Failed to write back {server_name} requirements.txt: {write_exc}")
                        else:
                            logger.info(f"requirements.txt for {server_name} is empty, skipping installation")
                    except Exception as install_exc:
                        logger.warning(f"Failed to read/install requirements for {server_name}: {install_exc}")
                else:
                    logger.info(f"No requirements.txt found for {server_name}, skipping dependency installation")
        except Exception as e:
            logger.warning(f"Dependency installation step exception: {e}")

        # Process initial state data
        temp_dir = None
        load_state_files = {}
        save_state_files = {}
        if init_states:
            try:
                # Create a temporary directory inside the project root
                project_root = Path(__file__).parent.parent.parent.parent  # Back to project root
                temp_dir = project_root / "tmp" / "init_states"
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                # If init_states is a single dict, convert it to a list
                if isinstance(init_states, dict):
                    init_states = [init_states]
                
                # Save state files for each server in the local temporary directory
                for i, server_name in enumerate(server_names):
                    if i < len(init_states):
                        state_data = init_states[i]
                        local_file_path = temp_dir / f"{server_name}.json"
                        
                        # Save state to local JSON file
                        with open(local_file_path, 'w', encoding='utf-8') as f:
                            json.dump(state_data, f, ensure_ascii=False, indent=2)
                
                # Import temporary files into the Sandbox via import_file_map
                local_files = {
                    str(temp_dir) : "init_states/"
                }
                self.sandbox.import_file_map(local_files)
                logger.info(f"Imported {len(local_files)} init state files to sandbox")
                
                # Set state file paths inside the Sandbox
                for server_name in server_names:
                    load_state_files[server_name] = f"init_states/{server_name}.json"
                    save_state_files[server_name] = f"final_states/{server_name}.json"
                        
            except Exception as e:
                logger.error(f"Failed to save init states: {e}")
                return False, f"Failed to save initial states: {str(e)}"
            finally:
                # Clean up local temporary files
                if temp_dir and temp_dir.exists():
                    try:
                        import shutil
                        shutil.rmtree(temp_dir)
                        logger.info(f"Cleaned up local temp directory: {temp_dir}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup temp directory {temp_dir}: {cleanup_error}")

        # Create configuration list
        try:
            configs = []
            for server_name in server_names:
                auto_load = load_state_files.get(server_name)
                auto_save = save_state_files.get(server_name)
                if auto_load:
                    auto_load = os.path.join(self.sandbox.work_dir, auto_load)
                if auto_save:
                    auto_save = os.path.join(self.sandbox.work_dir, auto_save)
                config = MCPServerConfig(
                    name=server_name,
                    file_path=registry[server_name]["env_code_path"],
                    auto_load_path=auto_load,
                    auto_save_path=auto_save,
                    env=full_env,
                    python_exec=self.sandbox._get_python_executable()
                )
                configs.append(config)
            return True, configs
        except Exception as e:
            return False, f"Failed to create configuration objects: {str(e)}"

    def create_mcp_server(
        self,
        server_names: Union[str, List[str]],
        init_states: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        *,
        registry_path: str = None,
        working_dir: str = None,
        env: Dict[str, str] = None,
        fail_fast: bool = False,
    ) -> Dict[str, Any]:
        """
        Create MCP servers (supports single or multiple server names)
        
        Args:
            server_names: Single server name or a list of server names
            registry_path: Path to registry.json (defaults to project path)
            working_dir: Working directory (defaults to sandbox work directory)
            env: Environment variables
            fail_fast: When multiple configs, mark summary success False if any fail
        
        Returns:
            Dict[str, Any]: Creation summary
                {
                    "success": bool,            # True if all succeed (or per fail_fast)
                    "created": int,             # number of successes
                    "failed": int,              # number of failures
                    "results": { name: {...} }, # raw per-server results (pass-through)
                    "errors":  { name: "..." }  # error messages for failed items only
                }
        """
        # Auto-import simulation files and set up base template
        self.sandbox.import_file_map({"src/simulation/*": "src/simulation/"})
        template_result = self.setup_mcp_base_template()
        if not template_result.get("success", False):
            return {
                "success": False,
                "error": template_result.get("error") or template_result.get("message", "Template setup failed"),
                "template_result": template_result,
            }

        # Prepare server configurations
        success, configs = self._prepare_server_configs(server_names, init_states, registry_path, env)
        if not success:
            return {"success": False, "error": configs}
    
        # Process all configurations uniformly
        summary = {
            "success": True,
            "created": 0,
            "failed": 0,
            "results": {},
            "errors": {},
        }
        
        for config in configs:
            try:
                result = self.sandbox.create_mcp_server(config)
                server_name = config.name
                
                summary["results"][server_name] = result
                
                if result.get("success", False):
                    summary["created"] += 1
                else:
                    summary["failed"] += 1
                    summary["errors"][server_name] = result.get("error", "unknown error")
                    if fail_fast:
                        summary["success"] = False
                        
            except Exception as e:
                server_name = config.name
                error_msg = f"Exception creating MCP server: {str(e)}"
                logger.error(f"Exception creating server '{server_name}': {e}")
                
                summary["results"][server_name] = {"success": False, "error": error_msg}
                summary["failed"] += 1
                summary["errors"][server_name] = error_msg
                
                if fail_fast:
                    summary["success"] = False

        # If fail_fast is not enabled, overall success depends on whether all succeeded
        if not fail_fast:
            summary["success"] = summary["failed"] == 0

        return summary

    def stop_mcp_server(self, server_name: str) -> Dict[str, Any]:
        """
        Stop an MCP server
        
        Args:
            server_name: Server name
        
        Returns:
            Dict[str, Any]: Stop result
        """
        try:
            result = self.sandbox.stop_mcp_server(server_name)
            return result
        except Exception as e:
            logger.error(f"Failed to stop MCP server {server_name}: {e}")
            return {"success": False, "error": f"Exception stopping MCP server: {str(e)}"}

    def restart_mcp_server(self, server_name: str) -> Dict[str, Any]:
        """
        Restart an MCP server
        
        Args:
            server_name: Server name
        
        Returns:
            Dict[str, Any]: Restart result
        """
        try:
            result = self.sandbox.restart_mcp_server(server_name)
            return result
        except Exception as e:
            logger.error(f"Failed to restart MCP server {server_name}: {e}")
            return {"success": False, "error": f"Exception restarting MCP server: {str(e)}"}

    def list_mcp_servers(self) -> Dict[str, Any]:
        """
        List all MCP servers
        
        Returns:
            Dict[str, Any]: Server list
        """
        try:
            result = self.sandbox.list_mcp_servers()
            return result
        except Exception as e:
            logger.error(f"Failed to list MCP servers: {e}")
            return {"success": False, "error": f"Exception listing MCP servers: {str(e)}"}

    def get_mcp_server_status(self, server_name: str) -> Dict[str, Any]:
        """
        Get MCP server status
        
        Args:
            server_name: Server name
        
        Returns:
            Dict[str, Any]: Server status
        """
        try:
            result = self.sandbox.get_mcp_server_status(server_name)
            return result
        except Exception as e:
            logger.error(f"Failed to get MCP server status: {e}")
            return {"success": False, "error": f"Exception getting server status: {str(e)}"}

    # === Tool Management ===
    def list_mcp_server_tools(self, server_name: str, public_only: bool = True) -> Dict[str, Any]:
        """
        List tools of an MCP server
        
        Args:
            server_name: Server name
            public_only: Whether to return only public tools
        
        Returns:
            Dict[str, Any]: Tool list
        """
        try:
            result = self.sandbox.list_mcp_server_tools(server_name, public_only=public_only)
            return result
        except Exception as e:
            logger.error(f"Failed to list MCP server tools: {e}")
            return {"success": False, "error": f"Exception getting tool list: {str(e)}"}

    def _ensure_server_active(self, server_name: str) -> Tuple[bool, List[str], Dict[str, Any]]:
        active = self.get_active_servers()
        if server_name in active:
            return True, active, {}
        restart_res = self.restart_mcp_server(server_name)
        active = self.get_active_servers()
        if server_name in active:
            return True, active, {"restart": restart_res}
        create_res = self.create_mcp_server(
            server_names=server_name,
            registry_path=self.registry_path,
            fail_fast=True,
        )
        active = self.get_active_servers()
        if server_name in active:
            return True, active, {"restart": restart_res, "create": create_res}
        return False, active, {"restart": restart_res, "create": create_res}

    def call_mcp_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any] = None,
        timeout: Optional[float] = 30.0,
    ) -> Dict[str, Any]:
        """
        Call an MCP server tool
        
        Args:
            server_name: Server name
            tool_name: Tool name
            arguments: Tool arguments
        
        Returns:
            Dict[str, Any]: Tool execution result
        """
        try:
            active_servers = self.get_active_servers()
            if server_name not in active_servers:
                ok, active_servers, recovery = self._ensure_server_active(server_name)
                if not ok:
                    error_msg = f"Server '{server_name}' not found in active servers: {active_servers}"
                    logger.error(f"{error_msg} | recovery={recovery}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "server_name": server_name,
                        "tool_name": tool_name,
                        "recovery": recovery,
                    }
            if server_name == "terminal-controller" and tool_name == "execute_command":
                cmd = (arguments or {}).get("command", "")
                reason = self._validate_terminal_command(cmd)
                if reason:
                    return {
                        "success": False,
                        "error": reason,
                        "server_name": server_name,
                        "tool_name": tool_name,
                        "blocked": True,
                    }
            logger.debug(f"Calling MCP tool: {server_name}.{tool_name} with args: {arguments}")
            try:
                result = self.sandbox.call_mcp_tool(
                    server_name, tool_name, arguments or {}, timeout=timeout
                )
            except Exception as e:
                ok, active_servers, recovery = self._ensure_server_active(server_name)
                if ok:
                    result = self.sandbox.call_mcp_tool(
                        server_name, tool_name, arguments or {}, timeout=timeout
                    )
                else:
                    return {
                        "success": False,
                        "error": f"Exception calling tool: {str(e)}",
                        "server_name": server_name,
                        "tool_name": tool_name,
                        "recovery": recovery,
                    }
            safe_obj = LogTruncator.truncate_json_for_log(result, max_total_chars=20000)
            logger.debug(f"MCP tool call result: {safe_obj}")
            return safe_obj
        except Exception as e:
            logger.error(f"Failed to call MCP tool {server_name}.{tool_name}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"success": False, "error": f"Exception calling tool: {str(e)}"}

    def _validate_terminal_command(self, command: str) -> Optional[str]:
        if not isinstance(command, str) or not command.strip():
            return "Invalid command"
        lowered = command.lower()
        if re.search(r'(^|\\s)sudo\\b', lowered):
            return "sudo is not allowed"
        if re.search(r'\\b(passwd|chpasswd|dscl|security|launchctl|systemctl|service|shutdown|reboot|halt)\\b', lowered):
            return "system command is not allowed"
        if re.search(r'(^|\\s)find\\s+/', command) or re.search(r'(^|\\s)du\\s+/', command):
            return "root scan is not allowed"
        work_dir = ""
        if self.sandbox and getattr(self.sandbox, "work_dir", None):
            work_dir = os.path.abspath(self.sandbox.work_dir)
        try:
            tokens = shlex.split(command, posix=True)
        except Exception:
            tokens = command.split()
        for token in tokens:
            if "/dev/null" in token:
                continue
            if token.startswith("~"):
                expanded = os.path.abspath(os.path.expanduser(token))
                if work_dir and (expanded == work_dir or expanded.startswith(work_dir + os.sep)):
                    continue
                return f"path not allowed: {token}"
            if token.startswith("/"):
                if work_dir and (token == work_dir or token.startswith(work_dir + os.sep)):
                    continue
                if token == "/dev/null":
                    continue
                return f"path not allowed: {token}"
        return None

    # === Convenience Methods ===
    def get_all_tools(self) -> Dict[str, Any]:
        """Get tools from all active servers"""
        try:
            servers_result = self.list_mcp_servers()
            if not servers_result.get("success"):
                return servers_result

            all_tools: Dict[str, List[Dict[str, Any]]] = {}
            servers = servers_result.get("servers", [])

            # Query tools from each server serially
            for server in servers:
                # Handle two possible data formats
                if isinstance(server, dict):
                    # Standard format: {"name": "server_name", "status": "running"}
                    name = server.get("name")
                    status = server.get("status")
                elif isinstance(server, str):
                    # Simplified format: server name as string
                    name = server
                    status = "running"  # Assume string format servers are running
                else:
                    continue
                
                if name and status == "running":
                    try:
                        tools_result = self.list_mcp_server_tools(name)
                        if tools_result.get("success"):
                            all_tools[name] = tools_result.get("tools", [])
                    except Exception as exc:
                        logger.error(f"Failed to get tools from {name}: {exc}")
                        all_tools[name] = []

            return {"success": True, "tools": all_tools, "server_count": len(servers)}
        except Exception as e:
            logger.error(f"Failed to get all tools: {e}")
            return {"success": False, "error": f"Exception getting all tools: {str(e)}"}

    def get_active_servers(self) -> List[str]:
        """Get the list of active MCP server names"""
        try:
            result = self.list_mcp_servers()
            if result.get("success"):
                servers = result.get("servers", [])
                active_servers = []
                for server in servers:
                    # Handle two possible data formats
                    if isinstance(server, dict):
                        # Standard format: {"name": "server_name", "status": "running"}
                        if server.get("status") == "running":
                            server_name = server.get("name")
                            if server_name:
                                active_servers.append(server_name)
                    elif isinstance(server, str):
                        # Simplified format: server name as string
                        active_servers.append(server)
                return active_servers
            return []
        except Exception as e:
            logger.error(f"Error getting active servers: {e}")
            return []
    def check_template_exists(self, template_name: str = None) -> tuple[bool, str]:
        """
        Check whether an environment template already exists
        
        Args:
            template_name: Template name; use default if None
            
        Returns:
            tuple[bool, str]: (exists, template ID or error message)
        """
        if template_name is None:
            template_name = "mcp_base_template"  # Use default template name
            
        print(f"\n🔍 Checking if environment template exists: {template_name}")
        
        try:
            # Get all available environment templates
            templates_result = self.sandbox.list_environment_templates()
            
            if not templates_result.get("success", False):
                print(f"❌ Failed to get template list: {templates_result.get('error', 'Unknown error')}")
                return False, templates_result.get('error', 'Unknown error')
            
            templates = templates_result.get("templates", [])
            print(f"📋 Found {len(templates)} existing templates")
            
            # Find matching template
            for template in templates:
                if template.get("name") == template_name:
                    # Use the correct field name: template_id
                    template_id = template.get("template_id")
                    print(f"✅ Found existing template: {template_name} (ID: {template_id})")
                    print(f"   Created at: {template.get('created_at', 'Unknown')}")
                    print(f"   Size: {template.get('size_mb', 0):.2f}MB")
                    print(f"   Package count: {template.get('packages_count', 0)}")
                    print(f"   Template structure: {list(template.keys())}")
                    return True, template_id
            
            print(f"❌ Template not found: {template_name}")
            return False, f"Template {template_name} does not exist"
            
        except Exception as e:
            print(f"❌ Exception while checking template existence: {e}")
            return False, str(e)
    
    def setup_mcp_base_template(self) -> Dict[str, Any]:
        """
        Set up the MCP base template environment
        
        Try loading the mcp_base_template virtual environment; if that fails,
        automatically install the environment and save it as a template.
        
        Returns:
            Dict[str, Any]: Environment setup result
                {
                    "success": bool,
                    "message": str,
                    "template_id": str,
                    "packages_installed": int,
                    "error": str (only on failure)
                }
        """
        if self._template_ready and self._template_id:
            return {
                "success": True,
                "message": "Template already loaded",
                "template_id": self._template_id,
                "packages_installed": 0,
                "loaded_from_existing": True,
            }

        with _TEMPLATE_SETUP_LOCK:
            if self._template_ready and self._template_id:
                return {
                    "success": True,
                    "message": "Template already loaded",
                    "template_id": self._template_id,
                    "packages_installed": 0,
                    "loaded_from_existing": True,
                }

            template_name = "mcp_base_template"
            exists, result = self.check_template_exists(template_name)
            
            if exists:
                template_id = result
                print(f"✅ Template exists, loading directly: {template_name} (ID: {template_id})")
                
                try:
                    load_result = self.sandbox.load_environment_template(template_id)
                    if load_result.get("success", False):
                        print(f"🎉 Template loaded successfully: {template_name}")
                        self._template_ready = True
                        self._template_id = template_id
                        return {
                            "success": True,
                            "message": f"Successfully loaded existing template: {template_name}",
                            "template_id": template_id,
                            "packages_installed": 0,
                            "loaded_from_existing": True
                        }
                    else:
                        print(f"❌ Template load failed: {load_result.get('error', 'Unknown error')}")
                        print("🔄 Will attempt to recreate the template...")
                except Exception as e:
                    print(f"❌ Exception while loading template: {e}")
                    print("🔄 Will attempt to recreate the template...")
        
            print(f"🔧 Creating new base template: {template_name}")
            
            try:
                requirements_file = Path("src/utils/requirements.txt")
                if requirements_file.exists():
                    with open(requirements_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    base_packages = []
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            base_packages.append(line)
                    
                    print(f"📦 Read {len(base_packages)} packages from {requirements_file}")
                else:
                    print(f"⚠️  {requirements_file} not found; using default package list")
                    base_packages = [
                        "requests>=2.31.0",
                        "loguru>=0.7.0",
                        "fastapi>=0.104.0",
                        "uvicorn>=0.24.0",
                        "pydantic>=2.5.0",
                    ]
                
                third_party_packages = [
                    pkg for pkg in base_packages 
                    if not self._is_standard_library(pkg.split(">=")[0].split("==")[0])
                ]
                
                print(f"📋 Third-party packages to install: {len(third_party_packages)}")
                for pkg in third_party_packages:
                    print(f"  - {pkg}")
                
                print("🔧 Installing base dependencies...")
                start_time = time.time()
                
                install_code = f"""
import subprocess
import sys

packages = {third_party_packages}
for package in packages:
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Installed successfully: {{package}}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {{package}}")
        print(f"Error: {{e.stderr}}")
        raise e
"""
                
                result = self.sandbox.run_code(install_code)
                
                if not result.get("success", False):
                    print(f"❌ Batch installation failed")
                    print(f"Error: {result.get('error', 'Unknown error')}")
                    print(f"Stdout: {result.get('stdout', '')}")
                    print(f"Stderr: {result.get('stderr', '')}")
                    return {
                        "success": False,
                        "message": f"Package installation failed: {result.get('error', 'Unknown error')}",
                        "error": result.get('error', 'Unknown error')
                    }
                else:
                    print(f"✅ All packages installed successfully")
                    print(f"Install output: {result.get('stdout', '')}")
                
                install_time = time.time() - start_time
                print(f"⏱️  Base package installation completed in: {install_time:.2f}s")
                
                print(f"💾 Saving environment template: {template_name}")
                
                save_result = self.sandbox.save_environment_template(template_name)
                if save_result.get("success", False):
                    print(f"✅ Environment template saved successfully: {template_name}")
                    template_id = save_result.get("template_id", template_name)
                    print(f"📋 Template ID: {template_id}")
                    self._template_ready = True
                    self._template_id = template_id
                    return {
                        "success": True,
                        "message": f"Successfully created and saved new template: {template_name}",
                        "template_id": template_id,
                        "packages_installed": len(third_party_packages),
                        "loaded_from_existing": False
                    }
                else:
                    print(f"❌ Failed to save environment template: {save_result.get('error', 'Unknown error')}")
                    return {
                        "success": False,
                        "message": f"Environment template save failed: {save_result.get('error', 'Unknown error')}",
                        "error": save_result.get('error', 'Unknown error')
                    }
                    
            except Exception as e:
                print(f"❌ Exception while creating base environment template: {e}")
                return {
                    "success": False,
                    "message": f"Exception while creating base environment template: {str(e)}",
                    "error": str(e)
                }

    # === Lifecycle Management ===
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up MCPSandboxBridge")

        try:
            self.sandbox.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        logger.info("MCPSandboxBridge cleanup completed")

    def __enter__(self):
        """Synchronous context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Synchronous context manager exit"""
        self.cleanup()
