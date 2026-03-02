# -*- coding: utf-8 -*-
"""
SandboxToolkit
--------------
Expose ONLY the high-level APIs as tools:
  - file_tool(action, file_path, content=None)
  - code_tool(action, code=None, bash_cmd=None, env_requirements=None)

Additionally provides NON-tool programmatic APIs (NOT exposed via get_tools):
  - extend_default_file_map(file_map: dict[str, str])

For file import functionality, use session.import_file_map() directly:
  - session.import_file_map(file_map: dict[str, str], add_to_sys_path=False, merge=True)
    * Supports BOTH:
        - directory mapping:  "utils/llm.py" -> "utils"
        - file mapping:       "utils/llm.py" -> "utils/llm.py"  (exact dest path, can rename)
    * Left side may be a glob pattern. If destination is a single file path,
      there must be exactly ONE match on the left and it must be a file.

Behavior
--------
- A persistent sandbox session is created/reused.
- Default files (default_file_map) can be uploaded once on first use.
- Default requirements (pip) can be installed once on first use.

Notes
-----
- For exact file mapping we write text via session.save_file(dest, text).
  If you need binary files, extend this class with a save_binary helper.
"""

from __future__ import annotations

import glob
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import Literal
from loguru import logger
from pathlib import Path
from src.core.sandbox import create_persistent_sandbox
from src.core.toolkits import FunctionTool
from src.core.toolkits.base import BaseToolkit
from src.core.sandbox.persistent_sandbox import PersistPolicy
from src.utils.llm import call_llm

MAX_RETURN_CHARS = 20_000


class SandboxToolkit(BaseToolkit):
    """Provision & reuse a persistent sandbox. Expose only file_tool / code_tool."""

    def __init__(
        self,
        *,
        memory_limit_mb: int = 512,
        timeout_minutes: int = 5,
        cleanup_paths_on_close: Optional[list[str]] = None,
        mount_dir: Optional[str] = None,
        default_file_map: Optional[dict[str, str]] = None,
        default_requirements: Optional[list[str]] = None,
        session: Any = None,
        bootstrap_on_init: bool = True,
        on_bootstrap_error: Literal["ignore", "raise", "log"] = "ignore",
        persist_policy: PersistPolicy = PersistPolicy.EPHEMERAL,
        temp_dir: Optional[str] = None,
        session_id: Optional[str] = None,
        mcp_server_mode:bool = False
    ) -> None:
        """
        Args:
            memory_limit_mb (int): Memory limit (MB) used when creating a new sandbox.
            timeout_minutes (int): Idle timeout (minutes) used when creating a new sandbox.
            default_file_map (Optional[dict[str, str]]): Host->sandbox path mapping to upload on first use.
                Supports dir mapping (dest is a directory) and exact file mapping (dest is a file path).
            default_requirements (Optional[list[str]]): Pip packages to install once on first use.
            session (Any): Existing sandbox session to reuse instead of creating a new one.
            bootstrap_on_init (bool): If True, attempt eager bootstrap during initialization.
            on_bootstrap_error (Literal["ignore","raise","log"]): Behavior when eager bootstrap fails.
        """
        self._session = session
        self._initialized = session is not None

        self._memory_limit_mb = memory_limit_mb
        self._timeout_minutes = timeout_minutes
        self._cleanup_paths_on_close = cleanup_paths_on_close
        self._mount_dir = mount_dir
        self._default_file_map = default_file_map or {}
        self._default_requirements = default_requirements or []
        self._on_bootstrap_error = on_bootstrap_error
        self._persist_policy = persist_policy
        self._temp_dir = os.path.abspath(temp_dir) if temp_dir else None
        self._session_id = session_id
        self._mcp_server_mode = mcp_server_mode
        if bootstrap_on_init:
            try:
                # Only ensure sandbox and upload default files once.
                self._ensure_sandbox()
            except Exception as e:
                if self._on_bootstrap_error == "raise":
                    raise
                if self._on_bootstrap_error == "log":
                    logger.exception(f"Sandbox eager bootstrap failed: {e}")
        

    # ----------------------------- exported tool methods -----------------------------

    def file_tool(
        self,
        action: str,
        file_path: str,
        content: Optional[str] = None,
    ) -> dict[str, Any]:
        r"""
        Read or write a text/python file inside the shared sandbox workspace.

        Args:
            action (str): Operation to perform. Use **"save"** to write text, or **"read"** to read text.
            file_path (str): File path relative to the sandbox root.
            content (str,optional): Text to write when `action="save"`. If omitted, an empty string is written.

        Returns:
            dict[str, Any]: JSON object indicating success or failure.
                - On **save**: includes a success flag and a short message.
                - On **read**: includes a success flag, a (possibly truncated) text snippet, and total character length.
                - On error: includes a failure flag and a human-readable error message.
        """
        try:
            session = self._ensure_sandbox()
            if action == "save":
                session.save_file(file_path, "" if content is None else str(content))
                # Keep original wording for compatibility with upstream consumers
                return {"success": True, "content": f"You success save content in {file_path}"}
            elif action == "read":
                text = session.read_file(file_path)
                snippet, total_len = self._safe_snippet(text)
                return {"success": True, "content": snippet, "full_length": total_len}
            else:
                return {"success": False, "error": f"Unknown action '{action}'"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _normalize_result(self,result: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize keys so callers can rely on a consistent payload."""
        result.setdefault("returncode", result.get("returncode", 0))
        result.setdefault("stdout", result.get("stdout", ""))
        result.setdefault("stderr", result.get("stderr", ""))
        result.setdefault("success", result.get("returncode", 0) == 0)
        result.setdefault("error", None)
        return result

    def run_code(
        self,
        code: str = None,
        env_requirements: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        r"""
        Execute Python code or a shell command inside the shared sandbox.

        Args:
            code (str): Python source to execute when using this `run_code` tool.
            env_requirements (list[str],optional): Pip package names to install **for this call** before execution.

        Returns:
            dict[str, Any]: JSON object indicating execution success or failure,
                including textual outputs (stdout, stderr), process return code,
                and a human-readable error message on failure.
        """
        try:
            if not code:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": "",
                    "returncode": -1,
                    "error": "Missing 'code' for run_code",
                }
            session = self._ensure_sandbox()
            per_call_reqs = env_requirements or []
            result = session.run_code(code, per_call_reqs)
            return self._normalize_result(result)
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "returncode": -1,
                "error": str(e),
            }

    def run_bash(
        self,
        bash_cmd: str = None,
        env_requirements: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        r"""
        Execute Python code or a shell command inside the shared sandbox.

        Args:
            bash_cmd (str): Shell command to execute when using this `run_bash` tool.
            env_requirements (list[str],optional): Pip package names to install **for this call** before execution.

        Returns:
            dict[str, Any]: JSON object indicating execution success or failure,
                including textual outputs (stdout, stderr), process return code,
                and a human-readable error message on failure.
        """
        try:
            if not bash_cmd:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": "",
                    "returncode": -1,
                    "error": "Missing 'bash_cmd' for run_bash",
                }
            session = self._ensure_sandbox()
            per_call_reqs = env_requirements or []
            result = session.exec_bash(
                bash_cmd,
                timeout=60 * 20,
                env_requirements=per_call_reqs,
            )
            return self._normalize_result(result)
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "returncode": -1,
                "error": str(e),
            }

    def run_pytest_with_analysis(
        self,
        file_path: str,
        env_requirements: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        r"""
        Run pytest on a specific file or path inside the sandbox and return
        only an LLM-generated analysis of the test run.

        Args:
            file_path (str): Path to the tests to run, relative to sandbox root.
            env_requirements (list[str], optional): Extra pip packages to install before running.

        Returns:
            dict[str, Any]: A dictionary with a single key:
                - "analysis": str, the LLM summary of the pytest run.
        """
        try:
            parts: List[str] = ["pytest", file_path]
            cmd = " ".join(parts)
            per_call_reqs = list(env_requirements) if env_requirements else []
            if "pytest" not in per_call_reqs:
                per_call_reqs.append("pytest")
            result = self.run_bash(bash_cmd=cmd, env_requirements=per_call_reqs)
            stdout = result.get("stdout", "") or ""
            stderr = result.get("stderr", "") or ""
            combined_output = stdout
            if stderr:
                combined_output = f"{stdout}\n\n[stderr]\n{stderr}" if stdout else f"[stderr]\n{stderr}"
            snippet, total_len = self._safe_snippet(combined_output)
            default_system_prompt = (
                "You are a senior Python testing expert. You receive the full pytest output "
                "for a test run executed inside an isolated sandbox. Provide a concise analysis: "
                "1) overall status; 2) list main failing tests with test names and error types; "
                "3) likely root causes; 4) concrete next debugging or fixing steps. "
                "Focus only on information available from the output."
            )
            analysis_input = (
                f"Command: {cmd}\n"
                f"Return code: {result.get('returncode')}\n"
                f"Output length: {total_len} characters\n\n"
                f"Pytest output snippet:\n{snippet}"
            )
            analysis = call_llm(
                text=analysis_input,
                system_prompt=default_system_prompt,
                max_tokens=800,
                temperature=0.2,
            )
            return {"analysis": analysis}
        except Exception as e:
            return {
                "analysis": f"Pytest execution or analysis failed: {e}",
            }
    # ------------------------------- non-tool public APIs -------------------------------

    def only_read_file(self, file_path: str) -> str:
        session = self._ensure_sandbox()
        text = session.read_file(file_path)
        return text

    def extend_default_file_map(self, file_map: dict[str, str]) -> None:
        """
        Update the toolkit's default file_map (used during first bootstrap).
        This does NOT immediately upload; call `session.import_file_map(...)` to upload now.
        """
        self._default_file_map.update(file_map)

    # ------------------------------- internal helpers -------------------------------

    def _ensure_sandbox(self):
        """Create the persistent sandbox on first use; bootstrap files & default deps once."""
        if self._session is None:
            self._session = create_persistent_sandbox(
                memory_limit_mb=self._memory_limit_mb,
                timeout_minutes=self._timeout_minutes,
                workspace_persist_policy=self._persist_policy,
                cleanup_paths_on_close=self._cleanup_paths_on_close,  # Pass the new parameter
                mount_dir=self._mount_dir,  # Pass the new parameter
                temp_dir=self._temp_dir,
                session_id=self._session_id,
            )
            self.sandbox = self._session
            self._initialized = False

            if self._mcp_server_mode:
                self.setup_mcp_base_template()
        
        if not self._initialized:
            # Upload initial files (default_file_map) – supports dir and exact file mapping
            if self._default_file_map:
                res = self._session.import_file_map(self._default_file_map)
                if not res.get("success"):
                    logger.error(f"Default file_map import failed: {res.get('error')}")

            # Install default requirements once (if any)
            if self._default_requirements:
                try:
                    self._session.run_code("", self._default_requirements)
                except Exception as e:
                    logger.error(f"Default requirements installation failed: {e}")

            self._initialized = True

        return self._session

    def _safe_snippet(self, text: str) -> tuple[str, int]:
        """Return (snippet, total_length) with truncation for very large content."""
        total_len = len(text)
        if total_len > MAX_RETURN_CHARS:
            half = MAX_RETURN_CHARS // 2
            return (
                f"{text[:half]}\n\n... (omitted {total_len - MAX_RETURN_CHARS} chars) ...\n\n{text[-half:]}",
                total_len,
            )
        return (text, total_len)

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

    def _is_standard_library(self,package_name: str) -> bool:
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
        template_name = "mcp_base_template"
        exists, result = self.check_template_exists(template_name)
        
        
        # If the template exists, load it directly
        if exists:
            template_id = result
            print(f"✅ Template exists, loading directly: {template_name} (ID: {template_id})")
            
            try:
                load_result = self.sandbox.load_environment_template(template_id)
                if load_result.get("success", False):
                    print(f"🎉 Template loaded successfully: {template_name}")
                    return {
                        "success": True,
                        "message": f"Successfully loaded existing template: {template_name}",
                        "template_id": template_id,
                        "packages_installed": 0,  # Existing template; no new packages needed
                        "loaded_from_existing": True
                    }
                else:
                    print(f"❌ Template load failed: {load_result.get('error', 'Unknown error')}")
                    print("🔄 Will attempt to recreate the template...")
                    # Load failure; continue with creation logic
            except Exception as e:
                print(f"❌ Exception while loading template: {e}")
                print("🔄 Will attempt to recreate the template...")
                # Load exception; continue with creation logic
        
        # Template does not exist or load failed; create a new template
        print(f"🔧 Creating new base template: {template_name}")
        
        try:
            # Define required base dependencies
            # Read src/utils/requirements.txt
            requirements_file = Path("src/utils/requirements.txt")
            if requirements_file.exists():
                with open(requirements_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                base_packages = []
                for line in lines:
                    line = line.strip()
                    # Skip comment and empty lines
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
            
            # Filter third-party packages (exclude standard library)
            third_party_packages = [
                pkg for pkg in base_packages 
                if not self._is_standard_library(pkg.split(">=")[0].split("==")[0])
            ]
            
            print(f"📋 Third-party packages to install: {len(third_party_packages)}")
            for pkg in third_party_packages:
                print(f"  - {pkg}")
            
            # Install base packages
            print("🔧 Installing base dependencies...")
            start_time = time.time()
            
            # Use a simpler installation method
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
            
            result = self.sandbox.run_code(install_code,timeout_sec=600)
            
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
            
            # Save environment template
            print(f"💾 Saving environment template: {template_name}")
            
            save_result = self.sandbox.save_environment_template(template_name)
            if save_result.get("success", False):
                print(f"✅ Environment template saved successfully: {template_name}")
                # Retrieve actual template ID
                template_id = save_result.get("template_id", template_name)
                print(f"📋 Template ID: {template_id}")
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
    
    
    
    def cleanup(self):
        self._session.cleanup()

    # ------------------------------- tool exposure ----------------------------------

    def get_tools(self) -> list[FunctionTool]:
        """Expose only file_tool and code_tool as FunctionTool."""
        return [
            FunctionTool(self.file_tool),
            FunctionTool(self.run_code),
            FunctionTool(self.run_bash),
            FunctionTool(self.run_pytest_with_analysis),
        ]
