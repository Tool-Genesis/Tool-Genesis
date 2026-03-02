"""
Long-running subprocess manager module
Supports creating, managing and STDIO communication with long-running subprocesses
"""

import os
import sys
import subprocess
import threading
import time
import signal
import select
import fcntl
import weakref
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from queue import Queue, Empty
import logging

logger = logging.getLogger(__name__)


class ProcessStatus(Enum):
    """Process status enumeration"""
    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class ProcessInfo:
    """Process information dataclass"""
    pid: int
    name: str
    command: List[str]
    status: ProcessStatus
    start_time: float
    working_dir: str
    env: Dict[str, str]


class ManagedProcess:
    """Managed subprocess class"""
    
    def __init__(self, name: str, command: List[str], working_dir: str = None, 
                 env: Dict[str, str] = None, on_output: Callable = None,
                 on_error: Callable = None, on_exit: Callable = None):
        self.name = name
        self.command = command
        self.working_dir = working_dir or os.getcwd()
        self.env = env or os.environ.copy()
        self.on_output = on_output
        self.on_error = on_error
        self.on_exit = on_exit
        
        self.process: Optional[subprocess.Popen] = None
        self.status = ProcessStatus.STARTING
        self.start_time = 0
        self.stdout_buffer = Queue()
        self.stderr_buffer = Queue()
        # Backward-compatible aliases
        self.stdout_queue = self.stdout_buffer
        self.stderr_queue = self.stderr_buffer
        self.stdout_thread: Optional[threading.Thread] = None
        self.stderr_thread: Optional[threading.Thread] = None
        self.monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._stop_monitoring = threading.Event()
        self._lock = threading.Lock()
        
    def start(self) -> bool:
        """Start process"""
        if self.process is not None:
            return False
        
        try:
            # Prepare environment
            process_env = os.environ.copy()
            if self.env:
                process_env.update(self.env)
            
            # Start process
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.working_dir,
                env=process_env,
                text=True,
                bufsize=0,
                preexec_fn=os.setsid  # Create new process group
            )
            
            # Set non-blocking IO
            self._set_nonblocking(self.process.stdout.fileno())
            self._set_nonblocking(self.process.stderr.fileno())
            
            self.status = ProcessStatus.RUNNING
            self.start_time = time.time()
            
            # Start output monitoring threads
            self._start_monitoring_threads()
            
            logger.info(f"Process {self.name} (PID: {self.process.pid}) started successfully")
            return True
            
        except Exception as e:
            self.status = ProcessStatus.ERROR
            logger.error(f"Failed to start process {self.name}: {e}")
            return False
    
    def _set_nonblocking(self, fd):
        """Set file descriptor to non-blocking mode"""
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    
    def _start_monitoring_threads(self):
        """Start monitoring threads"""
        # Start stdout monitoring thread
        self.stdout_thread = threading.Thread(
            target=self._monitor_stdout,
            daemon=True
        )
        self.stdout_thread.start()
        
        # Start stderr monitoring thread
        self.stderr_thread = threading.Thread(
            target=self._monitor_stderr,
            daemon=True
        )
        self.stderr_thread.start()
        
        # Start process monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_process,
            daemon=True
        )
        self.monitor_thread.start()
    
    def _monitor_stdout(self):
        """Monitor standard output"""
        while self.process and self.process.poll() is None and not self._stop_monitoring.is_set():
            try:
                ready, _, _ = select.select([self.process.stdout], [], [], 0.1)
                if ready:
                    line = self.process.stdout.readline()
                    if line:
                        line = line.strip()
                        self.stdout_buffer.put(line)
                        if self.on_output:
                            self.on_output(line)
            except Exception as e:
                if not self._stop_monitoring.is_set():
                    logger.error(f"Error monitoring {self.name} stdout: {e}")
                break
    
    def _monitor_stderr(self):
        """Monitor standard error"""
        while self.process and self.process.poll() is None and not self._stop_monitoring.is_set():
            try:
                ready, _, _ = select.select([self.process.stderr], [], [], 0.1)
                if ready:
                    line = self.process.stderr.readline()
                    if line:
                        line = line.strip()
                        self.stderr_buffer.put(line)
                        if self.on_error:
                            self.on_error(line)
            except Exception as e:
                if not self._stop_monitoring.is_set():
                    logger.error(f"Error monitoring {self.name} stderr: {e}")
                break
    
    def _monitor_process(self):
        """Monitor process status"""
        while self.process and not self._stop_monitoring.is_set():
            try:
                exit_code = self.process.poll()
                if exit_code is not None:
                    # Process has exited
                    self.status = ProcessStatus.STOPPED
                    logger.info(f"Process {self.name} exited with code: {exit_code}")
                    if self.on_exit:
                        self.on_exit(exit_code)
                    break
                time.sleep(0.1)
            except Exception as e:
                if not self._stop_monitoring.is_set():
                    logger.error(f"Error monitoring process {self.name}: {e}")
                break
    
    def send_input(self, data: str) -> bool:
        """Send input to process"""
        if not self.process or self.process.poll() is not None:
            return False
        
        try:
            self.process.stdin.write(data)
            if not data.endswith('\n'):
                self.process.stdin.write('\n')
            self.process.stdin.flush()
            return True
        except Exception as e:
            logger.error(f"Failed to send input to process {self.name}: {e}")
            return False
    
    def read_output(self, timeout: float = 1.0) -> List[str]:
        """Read standard output"""
        lines = []
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                line = self.stdout_buffer.get_nowait()
                lines.append(line)
            except Empty:
                if lines:  # If we already have output, return immediately
                    break
                time.sleep(0.01)
        
        return lines
    
    def read_error(self, timeout: float = 1.0) -> List[str]:
        """Read standard error"""
        lines = []
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                line = self.stderr_buffer.get_nowait()
                lines.append(line)
            except Empty:
                if lines:  # If we already have output, return immediately
                    break
                time.sleep(0.01)
        
        return lines
    
    def stop(self, timeout: float = 5.0) -> bool:
        """Stop process and its entire process group"""
        if not self.process:
            return True

        # Signal monitoring threads to stop
        self._stop_monitoring.set()

        try:
            if self.process.poll() is None:
                # Try graceful shutdown of the entire process group
                # (start() uses os.setsid so the process is a group leader)
                try:
                    pgid = os.getpgid(self.process.pid)
                    os.killpg(pgid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    self.process.terminate()

                # Wait for process to exit
                try:
                    self.process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    # Force kill the entire process group
                    try:
                        pgid = os.getpgid(self.process.pid)
                        os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        self.process.kill()
                    try:
                        self.process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Process {self.name} could not be killed")
                        return False
            
            # Wait for monitoring threads to finish
            if self.stdout_thread and self.stdout_thread.is_alive():
                self.stdout_thread.join(timeout=1.0)
            if self.stderr_thread and self.stderr_thread.is_alive():
                self.stderr_thread.join(timeout=1.0)
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=1.0)
            
            self.process = None
            self.status = ProcessStatus.STOPPED
            logger.info(f"Process {self.name} stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop process {self.name}: {e}")
            return False
    
    def restart(self) -> bool:
        """Restart process"""
        self.stop()
        time.sleep(0.5)  # Wait a bit to ensure complete stop
        return self.start()
    
    def get_info(self) -> ProcessInfo:
        """Get process information"""
        return ProcessInfo(
            pid=self.process.pid if self.process else -1,
            name=self.name,
            command=self.command,
            status=self.status,
            start_time=self.start_time,
            working_dir=self.working_dir or os.getcwd(),
            env=self.env or {}
        )
    
    def is_alive(self) -> bool:
        """Check if process is alive"""
        if not self.process:
            return False
        return self.process.poll() is None


class SubprocessManager:
    """Subprocess manager"""
    
    def __init__(self, max_processes: int = 10):
        self.max_processes = max_processes
        self.processes: Dict[str, ManagedProcess] = {}
        self._lock = threading.Lock()
    
    def create_process(self, name: str, command: List[str], 
                      working_dir: str = None, env: Dict[str, str] = None,
                      on_output: Callable = None, on_error: Callable = None,
                      on_exit: Callable = None) -> bool:
        """Create new process"""
        with self._lock:
            if name in self.processes:
                logger.warning(f"Process {name} already exists")
                return False
            
            if len(self.processes) >= self.max_processes:
                logger.error(f"Maximum process limit reached ({self.max_processes})")
                return False
            
            try:
                process = ManagedProcess(
                    name=name,
                    command=command,
                    working_dir=working_dir,
                    env=env,
                    on_output=on_output,
                    on_error=on_error,
                    on_exit=on_exit
                )
                
                if process.start():
                    self.processes[name] = process
                    return True
                else:
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to create process {name}: {e}")
                return False
    
    def get_process(self, name: str) -> Optional[ManagedProcess]:
        """Get process"""
        return self.processes.get(name)
    
    def stop_process(self, name: str, timeout: float = 5.0) -> bool:
        """Stop process"""
        with self._lock:
            if name not in self.processes:
                return False
            
            process = self.processes[name]
            if process.stop(timeout):
                del self.processes[name]
                return True
            return False
    
    def restart_process(self, name: str) -> bool:
        """Restart process"""
        process = self.get_process(name)
        if not process:
            return False
        return process.restart()
    
    def send_input(self, name: str, data: str) -> bool:
        """Send input to process"""
        process = self.get_process(name)
        if not process:
            return False
        return process.send_input(data)
    
    def read_output(self, name: str, timeout: float = 1.0) -> List[str]:
        """Read process output"""
        process = self.get_process(name)
        if not process:
            return []
        return process.read_output(timeout)
    
    def read_error(self, name: str, timeout: float = 1.0) -> List[str]:
        """Read process error output"""
        process = self.get_process(name)
        if not process:
            return []
        return process.read_error(timeout)
    
    def list_processes(self) -> List[ProcessInfo]:
        """List all processes"""
        return [process.get_info() for process in self.processes.values()]
    
    def get_process_status(self, name: str) -> Optional[ProcessStatus]:
        """Get process status"""
        process = self.get_process(name)
        if not process:
            return None
        return process.status
    
    def cleanup_all(self):
        """Clean up all processes"""
        with self._lock:
            for name in list(self.processes.keys()):
                try:
                    self.stop_process(name)
                except Exception as e:
                    logger.error(f"Failed to clean up process {name}: {e}")
            
            self.processes.clear()
            logger.info("All subprocesses cleaned up")
    
    def __del__(self):
        """Destructor, ensure all processes are cleaned up"""
        try:
            self.cleanup_all()
        except:
            pass
