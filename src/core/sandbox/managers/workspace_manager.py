"""
WorkspaceManager - Workspace Manager

Responsible for managing sandbox workspace creation, file operations, snapshots and restore functionality.
Supports three persistence policies: temporary, snapshot save, continuous mount.
"""

import os
import sys
import shutil
import json
import tarfile
import tempfile
import hashlib
import glob
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Sequence, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .file_watcher import FileWatcher


logger = logging.getLogger(__name__)


class PersistPolicy(str, Enum):
    """Persistence policy"""
    EPHEMERAL = "ephemeral"                # Temporary directory, delete on close
    SNAPSHOT_ON_CLOSE = "snapshot_on_close" # Temporary directory + generate tar.gz snapshot on close (default)
    CONTINUOUS_MOUNT = "continuous_mount"   # Stable directory per session, persist across runs


@dataclass
class WorkspaceConfig:
    """Workspace configuration"""
    workspace_root: str                  # e.g., "cache/sandbox/ws/workspaces"
    snapshot_root: str                   # e.g., "cache/sandbox/ws/snapshots"
    persist_policy: PersistPolicy = PersistPolicy.SNAPSHOT_ON_CLOSE
    mount_dir: Optional[str] = None      # Directory to mount continuous mount snapshots
    cleanup_paths_on_close: Optional[List[str]] = None # Paths to clean up on close for continuous mount
    debug: bool = True
    snapshot_retention_days: Optional[int] = None


@dataclass
class WorkspaceHandle:
    """Workspace handle"""
    session_id: str
    work_dir: str                        # Absolute path
    policy: PersistPolicy
    created_at: datetime
    mount_dir: Optional[str] = None      # Mount directory for continuous mount
    file_watcher: Optional[FileWatcher] = None # File watcher for real-time sync


class WorkspaceManager:
    """Workspace manager"""
    
    def __init__(self, cfg: WorkspaceConfig):
        self.config = cfg
        self.debug = cfg.debug
        
        # Ensure root directories exist
        os.makedirs(cfg.workspace_root, mode=0o700, exist_ok=True)
        os.makedirs(cfg.snapshot_root, mode=0o700, exist_ok=True)
        
        if self.debug:
            logger.info(f"[WorkspaceManager] Initialization complete, workspace root: {cfg.workspace_root}")
    
    def create(self, session_id: str, policy: PersistPolicy = None, mount_dir: Optional[str] = None) -> WorkspaceHandle:
        """
        Create workspace
        
        Args:
            session_id: Session ID
            policy: Persistence policy, if None use default from config
            
        Returns:
            WorkspaceHandle: Workspace handle
        """
        if policy is None:
            policy = self.config.persist_policy
        
        # Always create a temporary directory for workspace operations
        work_dir = tempfile.mkdtemp(prefix=f"{session_id}_", dir=self.config.workspace_root)
        os.chmod(work_dir, 0o700)
        
        file_watcher = None
        if policy == PersistPolicy.CONTINUOUS_MOUNT:
            # For continuous mount, determine the mount directory
            unique_suffix = uuid.uuid4().hex[:8]
            if mount_dir:
                mount_dir = os.path.join(mount_dir, f"{session_id}_{unique_suffix}")
            elif self.config.mount_dir:
                mount_dir = os.path.join(self.config.mount_dir, f"{session_id}_{unique_suffix}")
            else:
                mount_dir = os.path.join(self.config.workspace_root, f"{session_id}_{unique_suffix}")
            os.makedirs(mount_dir, mode=0o700, exist_ok=True)

            # Combine ignore patterns
            ignore_patterns = ["__pycache__", "*.pyc"]
            if self.config.cleanup_paths_on_close:
                ignore_patterns.extend(self.config.cleanup_paths_on_close)

            # Initialize and start the file watcher
            def sync_callback():
                self._sync_paths(work_dir, mount_dir, ignore_patterns=ignore_patterns)

            file_watcher = FileWatcher(path=work_dir, callback=sync_callback, ignore_patterns=ignore_patterns)
            file_watcher.start()

        handle = WorkspaceHandle(
            session_id=session_id,
            work_dir=work_dir,
            policy=policy,
            created_at=datetime.now(),
            mount_dir=mount_dir,
            file_watcher=file_watcher,
        )
        
        if self.debug:
            logger.info(f"[WorkspaceManager] Created workspace: {work_dir} (policy: {policy.value}, mount_dir: {mount_dir})")
        
        return handle
    
    def finalize(self, handle: WorkspaceHandle, *, snapshot_meta: dict = None) -> Optional[str]:
        """
        Finalize workspace processing
        
        Args:
            handle: Workspace handle
            snapshot_meta: Snapshot metadata
            
        Returns:
            Optional[str]: Returns snapshot path when policy is SNAPSHOT_ON_CLOSE, otherwise None
        """
        if handle.file_watcher:
            handle.file_watcher.stop()

        if handle.policy == PersistPolicy.CONTINUOUS_MOUNT:
            # For continuous mount, sync files to the mount directory
            if handle.mount_dir:
                # Clean up specified paths before syncing
                if self.config.cleanup_paths_on_close:
                    for path_to_clean in self.config.cleanup_paths_on_close:
                        full_path = os.path.join(handle.work_dir, path_to_clean)
                        if os.path.exists(full_path):
                            try:
                                if os.path.isdir(full_path):
                                    shutil.rmtree(full_path)
                                else:
                                    os.remove(full_path)
                                if self.debug:
                                    logger.info(f"[WorkspaceManager] Cleaned up path: {full_path}")
                            except Exception as e:
                                logger.error(f"[WorkspaceManager] Failed to clean up path {full_path}: {e}")

                self._sync_paths(
                    handle.work_dir,
                    handle.mount_dir,
                    ignore_patterns=["__pycache__", "*.pyc"] + (self.config.cleanup_paths_on_close or [])
                )
            return None

        if handle.policy != PersistPolicy.SNAPSHOT_ON_CLOSE:
            return None
        
        if not os.path.exists(handle.work_dir):
            if self.debug:
                logger.warning(f"[WorkspaceManager] Workspace does not exist, skipping snapshot: {handle.work_dir}")
            return None
        
        # Generate snapshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate content hash (optional)
        spec_hash = self._calculate_workspace_hash(handle.work_dir)
        
        snapshot_name = f"mcp_ws_{spec_hash}_{handle.session_id}_{timestamp}.tar.gz"
        snapshot_path = os.path.join(self.config.snapshot_root, snapshot_name)
        temp_snapshot_path = snapshot_path + ".tmp"
        
        try:
            # Create manifest
            manifest = {
                "session_id": handle.session_id,
                "created_at": handle.created_at.isoformat(),
                "snapshot_at": datetime.now().isoformat(),
                "policy": handle.policy.value,
                "files_count": self._count_files(handle.work_dir),
                "spec_hash": spec_hash,
                "meta": snapshot_meta or {}
            }
            
            # Create tar.gz snapshot (atomic write)
            with tarfile.open(temp_snapshot_path, "w:gz") as tar:
                # First create temporary manifest file
                manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False)
                temp_manifest_path = temp_snapshot_path + ".manifest"
                
                with open(temp_manifest_path, 'w', encoding='utf-8') as f:
                    f.write(manifest_json)
                
                # Add manifest file to tar
                tar.add(temp_manifest_path, arcname="manifest.json")
                
                # Add workspace content
                for root, dirs, files in os.walk(handle.work_dir):
                    # Filter hidden files and cache directories
                    dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                    
                    for file in files:
                        if file.startswith('.') or file.endswith('.pyc'):
                            continue
                        
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, handle.work_dir)
                        try:
                            tar.add(file_path, arcname=arcname)
                        except (OSError, IOError) as e:
                            if self.debug:
                                logger.warning(f"[WorkspaceManager] Skipping file {file_path}: {e}")
            
            # Atomic move to final location
            os.replace(temp_snapshot_path, snapshot_path)
            
            # Clean up temporary files
            if os.path.exists(temp_manifest_path):
                os.remove(temp_manifest_path)
            
            if self.debug:
                logger.info(f"[WorkspaceManager] Snapshot created: {snapshot_path}")
            
            try:
                self.purge_snapshots()
            except Exception:
                pass
            
            return snapshot_path
            
        except Exception as e:
            # Clean up temporary files
            for temp_file in [temp_snapshot_path, temp_snapshot_path + ".manifest"]:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass
            
            logger.error(f"[WorkspaceManager] Failed to create snapshot: {e}")
            raise RuntimeError(f"Failed to create snapshot: {e}")
    
    def purge_snapshots(self) -> Dict[str, int]:
        deleted = 0
        total = 0
        rd = self.config.snapshot_retention_days
        if not rd or rd <= 0:
            return {"deleted": 0, "total": 0}
        cutoff = datetime.now() - timedelta(days=rd)
        root = self.config.snapshot_root
        if not os.path.exists(root):
            return {"deleted": 0, "total": 0}
        for name in os.listdir(root):
            if not name.endswith(".tar.gz"):
                continue
            total += 1
            base = name[:-7]
            parts = base.split("_")
            if len(parts) < 5:
                continue
            ts_str = parts[-1]
            try:
                ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
            except Exception:
                continue
            if ts < cutoff:
                p = os.path.join(root, name)
                try:
                    os.remove(p)
                    deleted += 1
                except Exception:
                    continue
        return {"deleted": deleted, "total": total}
    
    def restore(self, snapshot_path: str, session_id: str = None) -> WorkspaceHandle:
        """
        Restore workspace from snapshot
        
        Args:
            snapshot_path: Snapshot file path
            session_id: Session ID, if None read from snapshot
            
        Returns:
            WorkspaceHandle: New workspace handle
        """
        if not os.path.exists(snapshot_path):
            raise FileNotFoundError(f"Snapshot file does not exist: {snapshot_path}")
        
        # Read manifest
        try:
            with tarfile.open(snapshot_path, "r:gz") as tar:
                manifest_member = tar.getmember("manifest.json")
                manifest_file = tar.extractfile(manifest_member)
                manifest = json.loads(manifest_file.read().decode('utf-8'))
        except Exception as e:
            raise RuntimeError(f"Failed to read snapshot manifest: {e}")
        
        # Determine session ID
        if session_id is None:
            session_id = manifest.get("session_id", f"restored_{int(datetime.now().timestamp())}")
        
        # Create new workspace
        work_dir = tempfile.mkdtemp(prefix=f"ws_{session_id}_restored_", dir=self.config.workspace_root)
        os.chmod(work_dir, 0o700)
        
        try:
            # Extract snapshot content
            with tarfile.open(snapshot_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name == "manifest.json":
                        continue  # Skip manifest file
                    
                    # Security check: prevent path traversal attacks
                    if os.path.isabs(member.name) or ".." in member.name:
                        logger.warning(f"[WorkspaceManager] Skipping unsafe path: {member.name}")
                        continue
                    
                    tar.extract(member, work_dir)
            
            handle = WorkspaceHandle(
                session_id=session_id,
                work_dir=work_dir,
                policy=PersistPolicy.SNAPSHOT_ON_CLOSE,  # Restored workspace defaults to snapshot policy
                created_at=datetime.now()
            )
            
            if self.debug:
                logger.info(f"[WorkspaceManager] Restored workspace from snapshot: {work_dir}")
            
            return handle
            
        except Exception as e:
            # Clean up failed workspace
            if os.path.exists(work_dir):
                shutil.rmtree(work_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to restore snapshot: {e}")
    
    def cleanup(self, handle: WorkspaceHandle) -> None:
        """
        Clean up workspace
        
        Args:
            handle: Workspace handle
        """
        if handle.file_watcher:
            handle.file_watcher.stop()

        if not os.path.exists(handle.work_dir):
            return
        
        if handle.policy == PersistPolicy.CONTINUOUS_MOUNT:
            # For continuous mount, the temporary work_dir is always cleaned up.
            # The mount_dir is preserved.
            if self.debug:
                logger.info(f"[WorkspaceManager] Preserving continuous mount directory: {handle.mount_dir}")
        
        # For all policies, the temporary work_dir is deleted.
        try:
            shutil.rmtree(handle.work_dir)
            if self.debug:
                logger.info(f"[WorkspaceManager] Cleaned up temporary workspace: {handle.work_dir}")
        except Exception as e:
            logger.error(f"[WorkspaceManager] Failed to clean up workspace {handle.work_dir}: {e}")
    
    def _resolve(self, handle: WorkspaceHandle, rel_path: str) -> Path:
        """
        Resolve relative path within workspace
        
        Args:
            handle: Workspace handle
            rel_path: Relative path
            
        Returns:
            Path: Absolute path
            
        Raises:
            ValueError: If path is outside workspace
        """
        work_path = Path(handle.work_dir).resolve()
        
        # Normalize the relative path to prevent directory traversal
        rel_path = str(rel_path).strip()
        if os.path.isabs(rel_path):
            raise ValueError(f"Absolute paths not allowed: {rel_path}")
        
        # Check for directory traversal attempts
        if '..' in rel_path or rel_path.startswith('/'):
            raise ValueError(f"Directory traversal not allowed: {rel_path}")
        
        # Remove any leading slashes or dots that could cause issues
        rel_path = rel_path.lstrip('./')
        
        # Construct target path
        target_path = (work_path / rel_path).resolve()
        
        # Check if path is within workspace (double check after resolution)
        try:
            target_path.relative_to(work_path)
        except ValueError:
            raise ValueError(f"Path outside workspace: {rel_path}")
        
        return target_path
    
    def _sync_paths(
        self,
        source_dir: str,
        target_dir: str,
        paths_to_sync: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None
    ) -> None:
        """
        Synchronize files from source to target directory, with improved efficiency.

        Args:
            source_dir: Source directory
            target_dir: Target directory
            paths_to_sync: Specific paths to sync (not used in this implementation)
            ignore_patterns: Patterns to ignore during sync
        """
        if not os.path.exists(source_dir):
            if self.debug:
                logger.warning(f"[WorkspaceManager] Source directory does not exist, skipping sync: {source_dir}")
            return

        # Ensure target directory exists
        os.makedirs(target_dir, exist_ok=True)

        # Define a function for shutil.copytree's ignore argument
        def ignore_func(directory, contents):
            ignored = set()
            if ignore_patterns:
                for item in contents:
                    for pattern in ignore_patterns:
                        if glob.fnmatch.fnmatch(item, pattern):
                            ignored.add(item)
            return ignored

        # Use shutil.copytree for efficient synchronization
        try:
            shutil.copytree(source_dir, target_dir, ignore=ignore_func, dirs_exist_ok=True)
            if self.debug:
                logger.info(f"[WorkspaceManager] Sync completed for {target_dir} using copytree")
        except Exception as e:
            logger.error(f"[WorkspaceManager] Error during sync with copytree: {e}")
            errors: List[Tuple[str, str, str]] = []
            for root, dirs, files in os.walk(source_dir):
                if ignore_patterns:
                    dirs[:] = [
                        d for d in dirs
                        if not any(glob.fnmatch.fnmatch(d, pattern) for pattern in ignore_patterns)
                    ]
                    files = [
                        f for f in files
                        if not any(glob.fnmatch.fnmatch(f, pattern) for pattern in ignore_patterns)
                    ]
                rel_root = os.path.relpath(root, source_dir)
                target_root = target_dir if rel_root == "." else os.path.join(target_dir, rel_root)
                os.makedirs(target_root, exist_ok=True)
                for file in files:
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(target_root, file)
                    try:
                        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                        shutil.copy2(src_path, dst_path)
                    except Exception as copy_error:
                        errors.append((src_path, dst_path, str(copy_error)))
            if errors:
                logger.error(f"[WorkspaceManager] Fallback sync failed with {len(errors)} errors")

    def _calculate_workspace_hash(self, work_dir: str) -> str:
        """Calculate workspace content hash"""
        hasher = hashlib.md5()
        
        # Sort to ensure consistency
        for root, dirs, files in os.walk(work_dir):
            dirs.sort()
            files.sort()
            
            for file in files:
                if file.startswith('.') or file.endswith('.pyc'):
                    continue
                
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, work_dir)
                hasher.update(rel_path.encode('utf-8'))
                
                try:
                    with open(file_path, 'rb') as f:
                        hasher.update(f.read())
                except (OSError, IOError):
                    # Skip files that cannot be read
                    continue
        
        return hasher.hexdigest()[:8]  # Take first 8 characters
    
    def _count_files(self, work_dir: str) -> int:
        """Count files in workspace"""
        count = 0
        for root, dirs, files in os.walk(work_dir):
            count += len([f for f in files if not f.startswith('.') and not f.endswith('.pyc')])
        return count
    
    # ========== Secure file system operations ==========
    
    def save_text(self, handle: WorkspaceHandle, rel_path: str, content: str) -> str:
        """
        Save text content to file
        
        Args:
            handle: Workspace handle
            rel_path: Relative path
            content: Text content
            
        Returns:
            str: Absolute path of saved file
        """
        target_path = self._resolve(handle, rel_path)
        
        # Ensure parent directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Atomic write
        temp_path = target_path.with_suffix(target_path.suffix + '.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Atomic move
            os.replace(temp_path, target_path)
            
            if self.debug:
                logger.debug(f"[WorkspaceManager] Saved text file: {target_path}")
            
            return str(target_path)
            
        except Exception as e:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to save file {rel_path}: {e}")
    
    def read_text(self, handle: WorkspaceHandle, rel_path: str) -> str:
        """
        Read text content of relative path
        
        Args:
            handle: Workspace handle
            rel_path: Relative path
            
        Returns:
            str: File content
        """
        target_path = self._resolve(handle, rel_path)
        
        if not target_path.exists():
            raise FileNotFoundError(f"File does not exist: {rel_path}")
        
        try:
            with open(target_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if self.debug:
                logger.debug(f"[WorkspaceManager] Read file: {rel_path}")
            
            return content
            
        except Exception as e:
            raise RuntimeError(f"Read file failed {rel_path}: {e}")
    
    def put_many(
        self,
        handle: WorkspaceHandle,
        sources: Sequence[str],
        dest_rel: str,
        *,
        add_to_sys_path: bool = False,
        merge: bool = True,
        overwrite: bool = True,
        ignore_patterns: Sequence[str] = ("__pycache__", "*.pyc", "*.pyo", ".git", ".mypy_cache")
    ) -> List[str]:
        """
        Batch copy files/directories to workspace
        
        Args:
            handle: Workspace handle
            sources: Source paths list
            dest_rel: Destination relative path
            add_to_sys_path: Whether to add to sys.path (return path information)
            merge: Whether to merge to existing directory
            overwrite: Whether to overwrite existing files
            ignore_patterns: Ignore patterns
            
        Returns:
            List[str]: Copied files paths list
        """
        if not sources:
            return []
        
        dest_dir = self._resolve(handle, dest_rel)
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        copied: List[str] = []
        
        for src in sources:
            src_path = Path(src).resolve()
            if not src_path.exists():
                raise FileNotFoundError(f"Source path does not exist: {src_path}")
            
            target = dest_dir / src_path.name
            
            if src_path.is_dir():
                if target.exists() and not merge:
                    raise FileExistsError(f"Target already exists and merge=False: {target}")
                
                # Use shutil.ignore_patterns handle ignore rules
                ignore_func = shutil.ignore_patterns(*(ignore_patterns or ()))
                shutil.copytree(src_path, target, dirs_exist_ok=True, ignore=ignore_func)
                copied.append(str(target))
                
            else:
                if target.exists() and not overwrite:
                    raise FileExistsError(f"Target file already exists and overwrite=False: {target}")
                
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, target)
                copied.append(str(target))
        
        if self.debug:
            logger.info(f"[WorkspaceManager] Copied {len(copied)} items to {dest_rel}")
        
        # Return sys.path information (if needed)
        result = copied
        if add_to_sys_path:
            # Return paths that need to be added to sys.path
            sys_path_info = {
                'copied_paths': copied,
                'add_sys_paths': [str(dest_dir)]
            }
            # Here can extend return more complex structure, but for compatibility, temporarily return path list
            pass
        
        return result
    
    def import_file_map(
        self,
        handle: WorkspaceHandle,
        file_map: dict[str, str],
        *,
        add_to_sys_path: bool = False,
        merge: bool = True
    ) -> dict[str, Any]:
        """
        Import files mapping to workspace
        
        Support two mapping ways:
          - Directory mapping:  "utils/llm.py" -> "utils"
          - File mapping:  "utils/llm.py" -> "utils/llm.py"  (exact target path, can be renamed)
        
        Left can be glob mode. If target is single file path, must be:
          - Left has only one match item, and
          - Match item must be file(not directory)
        
        Args:
            handle: Workspace handle
            file_map: Host path or glob mode -> workspace target path mapping
            add_to_sys_path: For directory mapping, whether to add target directory to sys.path
            merge: Whether to merge to existing target directory
        
        Returns:
            dict[str,Any]: {
              "success": bool,
              "imported": List[Tuple[str, str]],  # (host path, target relative path or exact file)
              "missing": List[str],               # Missing host patterns
              "error": Optional[str],
              "add_sys_paths": List[str]          # Need to add to sys.path paths (if add_to_sys_path=True)
            }
        """
        from collections import defaultdict
        
        def _is_file_like_path(p: str) -> bool:
            # Heuristic check: if last segment has extension, then consider it as "file path"
            base = os.path.basename(p.rstrip("/"))
            root, ext = os.path.splitext(base)
            return bool(root) and bool(ext)

        try:
            imported: List[Tuple[str, str]] = []
            missing: List[str] = []
            add_sys_paths: List[str] = []

            # Divide: directory batches and exact file pairs
            dir_batches: Dict[str, List[str]] = defaultdict(list)  # dest_dir -> [host_paths]
            file_pairs: List[Tuple[str, str]] = []                 # [(host_file, dest_file)]

            for host_pattern, dest in file_map.items():
                matches = glob.glob(host_pattern)
                if not matches:
                    missing.append(host_pattern)
                    continue

                if _is_file_like_path(dest):
                    # Exact file target
                    if len(matches) != 1:
                        return {
                            "success": False,
                            "imported": [],
                            "missing": missing,
                            "add_sys_paths": [],
                            "error": (
                                f"Fuzzy mapping: '{host_pattern}' matched {len(matches)} paths, "
                                f"but target '{dest}' is a single file path."
                            ),
                        }
                    src = matches[0]
                    if os.path.isdir(src):
                        return {
                            "success": False,
                            "imported": [],
                            "missing": missing,
                            "add_sys_paths": [],
                            "error": (
                                f"Invalid mapping: source '{src}' is directory but target '{dest}' "
                                f"is a file path. Please map directory to target directory."
                            ),
                        }
                    file_pairs.append((src, dest))
                else:
                    # Target is directory - batch upload
                    dir_batches[dest].extend(matches)

            # Upload directory/file to target directory
            for dest_dir, host_paths in dir_batches.items():
                copied_paths = self.put_many(
                    handle,
                    host_paths,
                    dest_rel=dest_dir,
                    add_to_sys_path=False,  # Here don't handle sys.path directly
                    merge=merge,
                )
                
                # Record imported files
                for hp in host_paths:
                    imported.append((hp, os.path.join(dest_dir, os.path.basename(hp))))
                
                # If need to add to sys.path
                if add_to_sys_path:
                    dest_abs_path = str(self._resolve(handle, dest_dir))
                    add_sys_paths.append(dest_abs_path)

            # Exact file to file upload (text mode)
            for host_path, dest_file in file_pairs:
                with open(host_path, "r", encoding="utf-8") as f:
                    text = f.read()
                self.save_text(handle, dest_file, text)
                imported.append((host_path, dest_file))
                
                # If caller needs sys.path behavior, we consider parent directory
                if add_to_sys_path:
                    parent = os.path.dirname(dest_file).rstrip("/") or "."
                    parent_abs_path = str(self._resolve(handle, parent))
                    if parent_abs_path not in add_sys_paths:
                        add_sys_paths.append(parent_abs_path)

            if self.debug:
                logger.info(f"[WorkspaceManager] Imported {len(imported)} files, missing {len(missing)} modes")

            return {
                "success": True,
                "imported": imported,
                "missing": missing,
                "add_sys_paths": add_sys_paths,
                "error": None
            }

        except Exception as e:
            logger.error(f"[WorkspaceManager] Failed to import files mapping: {e}")
            return {
                "success": False,
                "imported": [],
                "missing": [],
                "add_sys_paths": [],
                "error": str(e)
            }
    
    def info(self, handle: WorkspaceHandle) -> dict[str, Any]:
        """
        Get workspace info
        
        Args:
            handle: Workspace handle
            
        Returns:
            dict: Workspace info
        """
        try:
            work_dir_path = Path(handle.work_dir)
            
            # Calculate directory size
            total_size = 0
            files_count = 0
            last_modified = handle.created_at
            
            if work_dir_path.exists():
                for root, dirs, files in os.walk(handle.work_dir):
                    for file in files:
                        if file.startswith('.') or file.endswith('.pyc'):
                            continue
                        
                        file_path = os.path.join(root, file)
                        try:
                            stat = os.stat(file_path)
                            total_size += stat.st_size
                            files_count += 1
                            
                            file_mtime = datetime.fromtimestamp(stat.st_mtime)
                            if file_mtime > last_modified:
                                last_modified = file_mtime
                        except (OSError, IOError):
                            continue
            
            size_mb = total_size / (1024 * 1024)
            
            return {
                "work_dir": handle.work_dir,
                "policy": handle.policy.value,
                "size_mb": round(size_mb, 2),
                "files_count": files_count,
                "created_at": handle.created_at.isoformat(),
                "last_modified": last_modified.isoformat(),
                "session_id": handle.session_id,
                "exists": work_dir_path.exists()
            }
            
        except Exception as e:
            logger.error(f"[WorkspaceManager] Failed to get workspace info: {e}")
            return {
                "work_dir": handle.work_dir,
                "policy": handle.policy.value,
                "error": str(e),
                "session_id": handle.session_id,
                "exists": False
            }
