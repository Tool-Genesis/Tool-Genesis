"""
Environment Manager - supports saving, reusing and fast deployment of virtual environments

This module provides complete virtual environment management functionality, including:
1. Environment template saving and reuse
2. Fast environment copying
3. Pre-built environment management
4. Environment version control
5. Incremental package installation optimization
"""

import os
import sys
import json
import shutil
import hashlib
import subprocess
import logging
import tarfile
import tempfile
import time
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from threading import Lock
import venv
from builtins import open

import json
from dataclasses import asdict

logger = logging.getLogger(__name__)


@dataclass
class PackageInfo:
    """Package information"""
    name: str
    version: str
    installed_at: datetime
    install_path: str


@dataclass
class EnvironmentTemplate:
    """Environment template information"""
    template_id: str
    name: str
    description: str
    packages: Dict[str, PackageInfo]
    python_version: str
    created_at: datetime
    last_used: datetime
    template_path: str
    size_mb: float
    usage_count: int


@dataclass
class CacheEntry:
    """Cache entry"""
    packages: Dict[str, PackageInfo]
    created_at: datetime
    last_used: datetime
    python_version: str
    venv_path: str


class EnvironmentManager:
    """Environment Manager - responsible for creating, caching and managing virtual environment templates"""
    
    # Common package list for creating general templates
    COMMON_PACKAGES = [
        'fsspec==2025.9.0',
        'PyYAML==6.0.2', 
        'ruamel.yaml==0.18.14',
        'toml==0.10.2',
        'tomlkit==0.13.3',
        'mcp',
        'xmltodict==0.14.2',
        'configobj',
        'requests',
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'scikit-learn'
    ]
    
    def __init__(self, 
                 cache_dir: Optional[str] = None, 
                 templates_dir: Optional[str] = None,
                 max_cache_age_days: int = 7,
                 max_templates: int = 10,
                 cache_update_threshold_minutes: int = 5,
                 enable_lazy_cache_write: bool = True,
                 memory_only_mode: bool = True):
        """
        Initialize environment manager
        
        Args:
            cache_dir: Cache directory, defaults to cache/env_cache in current project
            templates_dir: Template directory, defaults to cache/env_templates in current project
            max_cache_age_days: Maximum cache retention days
            max_templates: Maximum number of templates
            cache_update_threshold_minutes: Cache update threshold (minutes)
            enable_lazy_cache_write: Whether to enable lazy write
            memory_only_mode: Whether to enable memory-only mode (no file I/O)
        """
        # Get project root directory (assumed to be current working directory or its parent)
        project_root = self._find_project_root()
        
        default_cache_dir = project_root / "cache" / "env_cache"
        default_templates_dir = project_root / "cache" / "env_templates"
        
        self.cache_dir = Path(cache_dir or default_cache_dir)
        self.templates_dir = Path(templates_dir or default_templates_dir)
        
        self.max_cache_age = timedelta(days=max_cache_age_days)
        self.max_templates = max_templates
        self.cache_update_threshold = timedelta(minutes=cache_update_threshold_minutes)
        # Maintain both minute value and timedelta for compatibility across call sites
        self._cache_update_threshold_minutes = cache_update_threshold_minutes
        self.cache_update_threshold_minutes = cache_update_threshold_minutes
        # Keep a private mirror for templates limit to match downstream usage
        self._max_templates = max_templates
        self.enable_lazy_cache_write = enable_lazy_cache_write
        self.memory_only_mode = memory_only_mode
        self.cache_file = self.cache_dir / "package_cache.json"
        self.templates_file = self.templates_dir / "templates.json"
        self.lock = Lock()
        
        # Lazy write related state
        self._cache_dirty = False
        self._last_cache_save = datetime.now()
        
        # Initialize cache and templates
        if self.memory_only_mode:
            # Memory-only mode: don't read files, initialize empty cache directly
            self._cache: Dict[str, CacheEntry] = {}
            self._templates: Dict[str, EnvironmentTemplate] = {}
            logger.info("EnvironmentManager initialized in memory-only mode")
        else:
            # Ensure directories exist (including parent directories)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.templates_dir.mkdir(parents=True, exist_ok=True)
            
            # Load cache and templates
            self._cache: Dict[str, CacheEntry] = self._load_cache()
            self._templates: Dict[str, EnvironmentTemplate] = self._load_templates()
            
            logger.info(f"EnvironmentManager initialized:")
            logger.info(f"  Cache dir: {self.cache_dir}")
            logger.info(f"  Templates dir: {self.templates_dir}")
            logger.info(f"  Loaded {len(self._cache)} cache entries")
            logger.info(f"  Loaded {len(self._templates)} templates")

    def _get_cache_update_minutes(self) -> int:
        """Safely get cache update threshold (minutes) with backward compatibility."""
        return int(getattr(self, "_cache_update_threshold_minutes", getattr(self, "cache_update_threshold_minutes", 5)))
    
    def _find_project_root(self) -> Path:
        """
        Find project root directory
        
        Returns:
            Project root directory path
        """
        current = Path.cwd()
        
        # Search upward for directories containing specific files (like requirements.txt, setup.py, pyproject.toml, etc.)
        project_indicators = [
            'requirements.txt', 'setup.py', 'pyproject.toml', 
            '.git', 'src', 'main.py', 'README.md'
        ]
        
        for parent in [current] + list(current.parents):
            if any((parent / indicator).exists() for indicator in project_indicators):
                return parent
        
        # If project root directory not found, return current directory
        return current

    def _safe_rmtree(self, path: Path) -> None:
        target = Path(path)
        if not target.exists():
            return
        for _ in range(3):
            try:
                shutil.rmtree(target)
                return
            except Exception:
                time.sleep(0.05)
        shutil.rmtree(target, ignore_errors=True)
    
    def _load_cache(self) -> Dict[str, CacheEntry]:
        """Load cache file"""
        if not self.cache_file.exists():
            return {}
        
        try:
            from builtins import open
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            cache = {}
            for key, entry_data in data.items():
                # Convert timestamps
                entry_data['created_at'] = datetime.fromisoformat(entry_data['created_at'])
                entry_data['last_used'] = datetime.fromisoformat(entry_data['last_used'])
                
                # Convert package information
                packages = {}
                for pkg_name, pkg_data in entry_data['packages'].items():
                    pkg_data['installed_at'] = datetime.fromisoformat(pkg_data['installed_at'])
                    packages[pkg_name] = PackageInfo(**pkg_data)
                
                entry_data['packages'] = packages
                cache[key] = CacheEntry(**entry_data)
            
            return cache
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return {}
    
    def _load_templates(self) -> Dict[str, EnvironmentTemplate]:
        """Load environment templates"""
        if not self.templates_file.exists():
            return {}
        
        try:
            with open(self.templates_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            templates = {}
            for template_id, template_data in data.items():
                # Convert timestamps
                template_data['created_at'] = datetime.fromisoformat(template_data['created_at'])
                template_data['last_used'] = datetime.fromisoformat(template_data['last_used'])
                
                # Convert package information
                packages = {}
                for pkg_name, pkg_data in template_data['packages'].items():
                    pkg_data['installed_at'] = datetime.fromisoformat(pkg_data['installed_at'])
                    packages[pkg_name] = PackageInfo(**pkg_data)
                
                template_data['packages'] = packages
                templates[template_id] = EnvironmentTemplate(**template_data)
            
            return templates
        except Exception as e:
            logger.warning(f"Failed to load templates: {e}")
            return {}
    
    def _save_cache(self, force: bool = False):
        """
        Save cache to file
        
        Args:
            force: Whether to force save, ignoring lazy write settings
        """
        # Don't perform file I/O in memory-only mode
        if self.memory_only_mode:
            return
        
        builtin_open = open
        with self.lock:
            # Mark cache as dirty but don't write immediately
            self._cache_dirty = True
            
            # If force save or lazy write is disabled, save immediately
            if force or not self.enable_lazy_cache_write:
                try:
                    # Convert time to string
                    cache_data = {}
                    for key, entry in self._cache.items():
                        entry_dict = asdict(entry)
                        entry_dict['created_at'] = entry.created_at.isoformat()
                        entry_dict['last_used'] = entry.last_used.isoformat()
                        
                        # Convert package information
                        packages_dict = {}
                        for pkg_name, pkg_info in entry.packages.items():
                            pkg_dict = asdict(pkg_info)
                            pkg_dict['installed_at'] = pkg_info.installed_at.isoformat()
                            packages_dict[pkg_name] = pkg_dict
                        
                        entry_dict['packages'] = packages_dict
                        cache_data[key] = entry_dict
                    

                    with builtin_open(self.cache_file, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, indent=2, ensure_ascii=False)
                    
                    self._cache_dirty = False
                    self._last_cache_save = datetime.now()
                    
                except Exception as e:
                    logger.error(f"Failed to save cache: {e}")
    
    def _maybe_save_cache(self):
        """
        Decide whether to save cache based on strategy
        """
        # Don't perform file I/O in memory-only mode
        if self.memory_only_mode:
            return
        
        with self.lock:
            if not self._cache_dirty:
                return
            
            # If time since last save exceeds threshold, save
            if datetime.now() - self._last_cache_save > self.cache_update_threshold:
                self._save_cache(force=True)
    
    def _save_templates(self):
        """Save templates to file"""
        # Don't perform file I/O in memory-only mode
        if self.memory_only_mode:
            return
        
        try:
            # Convert time to string
            templates_data = {}
            for template_id, template in self._templates.items():
                template_dict = asdict(template)
                template_dict['created_at'] = template.created_at.isoformat()
                template_dict['last_used'] = template.last_used.isoformat()
                
                # Convert package information
                packages_dict = {}
                for pkg_name, pkg_info in template.packages.items():
                    pkg_dict = asdict(pkg_info)
                    pkg_dict['installed_at'] = pkg_info.installed_at.isoformat()
                    packages_dict[pkg_name] = pkg_dict
                
                template_dict['packages'] = packages_dict
                templates_data[template_id] = template_dict
            
            with open(self.templates_file, 'w', encoding='utf-8') as f:
                json.dump(templates_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save templates: {e}")
    
    def _get_cache_key(self, venv_path: str, python_version: str) -> str:
        """Generate cache key"""
        return hashlib.md5(f"{venv_path}:{python_version}".encode()).hexdigest()
    
    def _get_template_id(self, packages: List[str]) -> str:
        """Generate template ID based on package list"""
        packages_str = ":".join(sorted(packages))
        return hashlib.md5(packages_str.encode()).hexdigest()[:16]
    
    def _get_installed_packages(self, python_executable: str) -> Dict[str, str]:
        """Get list of installed packages"""
        try:
            result = subprocess.run(
                [python_executable, '-m', 'pip', 'list', '--format=json'],
                capture_output=True, text=True, check=True
            )
            packages = json.loads(result.stdout)
            return {pkg['name']: pkg['version'] for pkg in packages}
        except Exception as e:
            logger.error(f"Failed to get installed packages: {e}")
            return {}
    
    def _parse_package_spec(self, package_spec: str) -> Tuple[str, Optional[str]]:
        """Parse package specification"""
        if '==' in package_spec:
            name, version = package_spec.split('==', 1)
            return name.strip(), version.strip()
        elif '>=' in package_spec:
            name, _ = package_spec.split('>=', 1)
            return name.strip(), None
        else:
            return package_spec.strip(), None
    
    def _calculate_dir_size(self, path: Path) -> float:
        """Calculate directory size (MB)"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception as e:
            logger.warning(f"Failed to calculate directory size: {e}")
        return total_size / (1024 * 1024)  # Convert to MB
    
    def save_environment_template(self, 
                                venv_path: str, 
                                template_name: str,
                                description: str = "") -> Dict[str, Any]:
        """
        Save environment template (simplified version, directly calls create_environment_template)
        
        Args:
            venv_path: Virtual environment path
            template_name: Template name
            description: Template description
            
        Returns:
            Save result
        """
        # Get Python executable path
        if os.path.exists(f"{venv_path}/bin/python"):
            python_executable = f"{venv_path}/bin/python"
        elif os.path.exists(f"{venv_path}/Scripts/python.exe"):
            python_executable = f"{venv_path}/Scripts/python.exe"
        else:
            return {"success": False, "error": "Cannot find Python executable"}
        
        return self.create_environment_template(
            venv_path, python_executable, template_name, description
        )
    
    def create_environment_template(self,
                                  venv_path: str, 
                                  python_executable: str,
                                  template_name: str,
                                  description: str = "") -> Dict[str, Any]:
        """
        Create environment template
        
        Args:
            venv_path: Virtual environment path
            python_executable: Python executable path
            template_name: Template name
            description: Template description
            
        Returns:
            Creation result
        """
        try:
            # Get installed packages
            installed_packages = self._get_installed_packages(python_executable)
            if not installed_packages:
                return {"success": False, "error": "Cannot get installed package list"}
            
            # Generate template ID
            package_list = list(installed_packages.keys())
            template_id = self._get_template_id(package_list)
            
            # Check if template with same content already exists
            for existing_id, template in self._templates.items():
                if set(template.packages.keys()) == set(installed_packages.keys()):
                    # Template already exists, update usage count
                    template.usage_count += 1
                    template.last_used = datetime.now()
                    self._save_templates()
                    return {
                        "success": True,
                        "template_id": existing_id,
                        "message": f"Template already exists: {template.name}",
                        "existing": True
                    }
            
            # Create template directory
            template_dir = self.templates_dir / template_id
            template_dir.mkdir(parents=True, exist_ok=True)
            
            # Compress virtual environment
            archive_path = template_dir / "environment.tar.gz"
            logger.info(f"Creating environment template: {template_name}")
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(venv_path, arcname="venv")
            
            # Calculate size
            size_mb = self._calculate_dir_size(template_dir)
            
            # Create package info
            package_info = {}
            for name, version in installed_packages.items():
                package_info[name] = PackageInfo(
                    name=name,
                    version=version,
                    installed_at=datetime.now(),
                    install_path=""
                )
            
            # Create template object
            template = EnvironmentTemplate(
                template_id=template_id,
                name=template_name,
                description=description,
                packages=package_info,
                python_version=sys.version.split()[0],
                created_at=datetime.now(),
                last_used=datetime.now(),
                template_path=str(archive_path),
                size_mb=size_mb,
                usage_count=1
            )
            
            # Save template
            self._templates[template_id] = template
            self._save_templates()
            
            # If template count exceeds limit, delete oldest template
            if len(self._templates) > self._max_templates:
                # Find oldest template (by last_used time)
                oldest_id = min(self._templates.keys(), 
                              key=lambda x: self._templates[x].last_used)
                self._remove_template(oldest_id)
            
            logger.info(f"Environment template created successfully: {template_name} ({template_id})")
            return {
                "success": True,
                "template_id": template_id,
                "message": f"Environment template created successfully: {template_name}",
                "size_mb": size_mb,
                "package_count": len(installed_packages)
            }
            
        except Exception as e:
            logger.error(f"Failed to create environment template: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def deploy_from_template(self, 
                           template_id: str, 
                           target_venv_path: str) -> Dict[str, Any]:
        """
        Deploy environment from template (copy template to independent directory)
        
        Args:
            template_id: Template ID
            target_venv_path: Target virtual environment path
            
        Returns:
            Deployment result
        """
        try:
            template = self._templates.get(template_id)
            if not template:
                return {"success": False, "error": f"Template does not exist: {template_id}"}
            
            template_path = Path(template.template_path)
            
            # Handle both directory and file paths
            if template_path.is_dir():
                # If template_path is a directory, look for environment.tar.gz inside
                archive_path = template_path / "environment.tar.gz"
            else:
                # If template_path is a file, use it directly
                archive_path = template_path
            
            # Check if template file exists
            if not archive_path.exists():
                return {"success": False, "error": f"Template file does not exist: {archive_path}"}
            
            # Ensure target directory exists
            target_path = Path(target_venv_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # If target environment already exists, delete it first
            if target_path.exists():
                self._safe_rmtree(target_path)
            
            logger.info(f"Deploying environment from template: {template.name} -> {target_venv_path}")
            
            # If target path exists, remove it first to avoid conflicts
            if target_path.exists():
                self._safe_rmtree(target_path)
            
            # Create parent directory for extraction
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Extract environment to parent directory (with path traversal protection)
            with tarfile.open(archive_path, "r:gz") as tar:
                # Filter out members with absolute paths or path traversal components
                def _safe_members(tar_obj):
                    dest = str(target_path.parent.resolve())
                    for member in tar_obj.getmembers():
                        member_path = os.path.normpath(os.path.join(dest, member.name))
                        if not member_path.startswith(dest + os.sep) and member_path != dest:
                            logger.warning(f"Skipping unsafe tar member: {member.name}")
                            continue
                        if member.issym() or member.islnk():
                            link_target = os.path.normpath(os.path.join(dest, member.linkname))
                            if not link_target.startswith(dest + os.sep) and link_target != dest:
                                logger.warning(f"Skipping unsafe symlink tar member: {member.name} -> {member.linkname}")
                                continue
                        yield member
                tar.extractall(target_path.parent, members=_safe_members(tar))
            
            # Check extraction result and reorganize directory structure
            extracted_venv_path = target_path.parent / "venv"
            if extracted_venv_path.exists():
                # Move venv directory to target path
                if extracted_venv_path.resolve() != target_path.resolve():
                    if target_path.exists():
                        self._safe_rmtree(target_path)
                    shutil.move(str(extracted_venv_path), str(target_path))
            else:
                # If no venv directory was extracted, the tar might have been extracted directly
                logger.warning(f"No venv directory found after extraction, assuming direct extraction to {target_path}")
                # Create target directory if it doesn't exist
                target_path.mkdir(parents=True, exist_ok=True)
            
            # Fix virtual environment path references (only fix when Python executable exists)
            python_exe = self._get_python_executable(str(target_path))
            if python_exe:
                self._fix_venv_paths(str(target_path))
            else:
                logger.warning(f"Python executable not found in {target_path}, skipping path fix")
            
            # Update template usage statistics
            template.usage_count += 1
            template.last_used = datetime.now()
            self._save_templates()
            
            logger.info(f"Environment deployment successful: {template.name}")
            return {
                "success": True,
                "message": f"Environment deployment successful: {template.name}",
                "target_path": str(target_path),
                "package_count": len(template.packages)
            }
            
        except Exception as e:
            logger.error(f"Environment deployment failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _remove_template(self, template_id: str):
        """Delete template"""
        template = self._templates.get(template_id)
        if template:
            # Delete template file
            template_path = Path(template.template_path)
            if template_path.exists():
                shutil.rmtree(template_path.parent)
            
            # Remove from memory
            del self._templates[template_id]
            logger.info(f"Deleted template: {template.name} ({template_id})")
    
    def find_best_template(self, required_packages: List[str]) -> Optional[str]:
        """
        Find the best matching template
        
        Args:
            required_packages: List of required packages
            
        Returns:
            Best template ID, returns None if not found
        """
        if not self._templates or not required_packages:
            return None
        
        best_template = None
        best_score = 0
        
        required_set = set(pkg.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0] 
                          for pkg in required_packages)
        
        for template_id, template in self._templates.items():
            template_packages = set(template.packages.keys())
            
            # Calculate match score
            intersection = required_set & template_packages
            
            # Match score = intersection size / required package count
            match_ratio = len(intersection) / len(required_set) if required_set else 0
            
            # If template contains all required packages, give extra points
            if required_set.issubset(template_packages):
                match_ratio += 0.5
            
            # Consider usage frequency
            usage_bonus = min(template.usage_count * 0.01, 0.1)
            final_score = match_ratio + usage_bonus
            
            if final_score > best_score:
                best_score = final_score
                best_template = template_id
        
        # Only return template if match score is reasonable (>= 50%)
        return best_template if best_score >= 0.5 else None
    
    def get_missing_packages(self, 
                           required_packages: List[str], 
                           venv_path: str, 
                           python_executable: str) -> List[str]:
        """
        Get list of packages that need to be installed (incremental installation)
        
        Args:
            required_packages: List of required packages
            venv_path: Virtual environment path
            python_executable: Python executable path
            
        Returns:
            List of packages that need to be installed
        """
        # First try to use template
        best_template = self.find_best_template(required_packages)
        if best_template:
            logger.info(f"Found matching template: {self._templates[best_template].name}")
            # If template found, consider deploying template first then checking missing packages
        
        # Use original cache logic
        python_version = sys.version.split()[0]
        cache_key = self._get_cache_key(venv_path, python_version)
        
        # Check cache
        cache_entry = self._cache.get(cache_key)
        
        # If cache expired or doesn't exist, recheck
        if (not cache_entry or 
            datetime.now() - cache_entry.last_used > timedelta(minutes=self._get_cache_update_minutes())):
            
            # Get currently installed packages
            installed_packages = self._get_installed_packages(python_executable)
            
            # Update cache
            self._cache[cache_key] = CacheEntry(
                packages={name: PackageInfo(
                    name=name,
                    version=version,
                    installed_at=datetime.now(),
                    install_path=""
                ) for name, version in installed_packages.items()},
                created_at=datetime.now(),
                last_used=datetime.now(),
                python_version=python_version,
                venv_path=venv_path
            )
        else:
            # Update last used time
            cache_entry.last_used = datetime.now()
            installed_packages = {pkg.name: pkg.version for pkg in cache_entry.packages.values()}
        
        # Only update last_used time when time interval exceeds threshold
        if (not hasattr(self, '_last_cache_update') or 
            datetime.now() - self._last_cache_update > timedelta(minutes=self._get_cache_update_minutes())):
            self._last_cache_update = datetime.now()
            
            # Use lazy write mechanism to avoid frequent file I/O
            self._maybe_save_cache()
        
        # Check which packages need to be installed
        missing_packages = []
        for package_spec in required_packages:
            package_name = self._parse_package_spec(package_spec)[0]
            if package_name not in installed_packages:
                missing_packages.append(package_spec)
        
        return missing_packages
    
    def update_installed_packages(self, 
                                installed_packages: List[str], 
                                venv_path: str, 
                                python_executable: str):
        """
        Update cache information for installed packages
        
        Args:
            installed_packages: List of newly installed packages
            venv_path: Virtual environment path
            python_executable: Python executable path
        """
        python_version = sys.version.split()[0]
        cache_key = self._get_cache_key(venv_path, python_version)
        
        # Get current cache entry
        cache_entry = self._cache.get(cache_key)
        if not cache_entry:
            # Create new cache entry
            cache_entry = CacheEntry(
                packages={},
                created_at=datetime.now(),
                last_used=datetime.now(),
                python_version=python_version,
                venv_path=venv_path
            )
            self._cache[cache_key] = cache_entry
        
        # Update newly installed package information
        current_packages = self._get_installed_packages(python_executable)
        for package_spec in installed_packages:
            package_name = self._parse_package_spec(package_spec)[0]
            
            # If no version specified, try to get actual installed version
            if package_name in current_packages:
                version = current_packages[package_name]
            else:
                # Use version from package spec
                version = self._parse_package_spec(package_spec)[1] or "unknown"
            
            cache_entry.packages[package_name] = PackageInfo(
                name=package_name,
                version=version,
                installed_at=datetime.now(),
                install_path=""
            )
        
        cache_entry.last_used = datetime.now()
        self._maybe_save_cache()
    
    def create_common_template(self) -> Dict[str, Any]:
        """
        Create template containing common packages
        
        Returns:
            Creation result
        """
        try:
            # Create temporary virtual environment
            with tempfile.TemporaryDirectory() as temp_dir:
                venv_path = os.path.join(temp_dir, "common_env")
                
                # Create virtual environment
                venv.create(venv_path, with_pip=True)
                
                # Get Python executable path
                python_executable = self._get_python_executable(venv_path)
                if not python_executable:
                    return {"success": False, "error": "Cannot find Python executable"}
                
                # Install common packages
                logger.info("Installing common packages to template environment...")
                for package in self.COMMON_PACKAGES:
                    result = subprocess.run([
                        python_executable, "-m", "pip", "install", package
                    ], capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        return {
                            "success": False,
                            "error": f"Failed to install common packages: {result.stderr}"
                        }
                
                # Create template
                return self.create_environment_template(
                    venv_path=venv_path,
                    python_executable=python_executable,
                    template_name="common-packages",
                    description="Pre-built environment containing common Python packages"
                )
                
        except Exception as e:
            logger.error(f"Failed to create common package template: {e}")
            return {"success": False, "error": str(e)}
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List all templates"""
        templates = []
        for template_id, template in self._templates.items():
            templates.append({
                "template_id": template_id,
                "name": template.name,
                "description": template.description,
                "package_count": len(template.packages),
                "python_version": template.python_version,
                "created_at": template.created_at.isoformat(),
                "last_used": template.last_used.isoformat(),
                "size_mb": template.size_mb,
                "usage_count": template.usage_count
            })
        
        # Sort by usage frequency
        templates.sort(key=lambda x: x["usage_count"], reverse=True)
        return templates
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache and template statistics"""
        total_cache_size = 0
        total_template_size = 0
        
        for entry in self._cache.values():
            # Estimate cache entry size (simplified calculation)
            total_cache_size += len(entry.packages) * 0.001  # MB
        
        for template in self._templates.values():
            total_template_size += template.size_mb
        
        return {
            "cache": {
                "entries": len(self._cache),
                "estimated_size_mb": round(total_cache_size, 2)
            },
            "templates": {
                "count": len(self._templates),
                "total_size_mb": round(total_template_size, 2),
                "templates": [
                    {
                        "name": t.name,
                        "size_mb": t.size_mb,
                        "usage_count": t.usage_count
                    }
                    for t in self._templates.values()
                ]
            }
        }
    
    def clear_venv_cache(self, venv_path: str):
        """Clear cache for specific virtual environment"""
        keys_to_remove = []
        for key, entry in self._cache.items():
            if entry.venv_path == venv_path:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._cache[key]
        
        # Only save to file in non-memory-only mode
        if not self.memory_only_mode:
            self._save_cache(force=True)
    
    def clear_all_cache(self):
        """Clear all cache"""
        self._cache.clear()
        
        # Only save to file in non-memory-only mode
        if not self.memory_only_mode:
            self._save_cache(force=True)
    
    # Environment cloning methods
    
    def clone_environment_direct(self, 
                               source_venv_path: str, 
                               target_venv_path: str) -> Dict[str, Any]:
        """
        Clone environment directly, no need to pass package list
        
        Args:
            source_venv_path: Source virtual environment path
            target_venv_path: Target virtual environment path
            
        Returns:
            Clone result
        """
        try:
            source_path = Path(source_venv_path)
            target_path = Path(target_venv_path)
            
            # Check if source environment exists
            if not source_path.exists():
                return {"success": False, "error": f"Source environment does not exist: {source_venv_path}"}
            
            # If target environment already exists, delete it first
            if target_path.exists():
                shutil.rmtree(target_path)
            
            # Copy entire virtual environment directory directly
            logger.info(f"Cloning environment: {source_venv_path} -> {target_venv_path}")
            shutil.copytree(source_venv_path, target_venv_path)
            
            # Fix Python paths (important: paths in virtual environment are hardcoded)
            self._fix_venv_paths(target_venv_path)
            
            # Get environment information
            python_executable = self._get_python_executable(target_venv_path)
            if python_executable:
                installed_packages = self._get_installed_packages(python_executable)
                package_count = len(installed_packages)
            else:
                package_count = 0
            
            return {
                "success": True,
                "message": f"Environment cloned successfully",
                "source_path": source_venv_path,
                "target_path": target_venv_path,
                "package_count": package_count
            }
            
        except Exception as e:
            logger.error(f"Environment cloning failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_python_executable(self, venv_path: str) -> Optional[str]:
        """Get Python executable path in virtual environment"""
        # Unix/Linux/macOS
        unix_python = os.path.join(venv_path, "bin", "python")
        if os.path.exists(unix_python):
            return unix_python
        
        # Windows
        windows_python = os.path.join(venv_path, "Scripts", "python.exe")
        if os.path.exists(windows_python):
            return windows_python
        
        return None
    
    def _fix_venv_paths(self, venv_path: str):
        """
        Fix hardcoded paths in virtual environment
        This is necessary because activation scripts etc. in virtual environment contain absolute paths
        """
        if not venv_path:
            logger.warning("venv_path is empty, skipping path fix")
            return
        
        # Fix pyvenv.cfg file
        pyvenv_cfg = os.path.join(venv_path, "pyvenv.cfg")
        if os.path.exists(pyvenv_cfg):
            try:
                with open(pyvenv_cfg, 'r') as f:
                    content = f.read()
                
                # Update home path to current Python's path
                import sys
                python_home = os.path.dirname(sys.executable)
                
                # Replace home path
                lines = content.split('\n')
                new_lines = []
                for line in lines:
                    if line.startswith('home = '):
                        new_lines.append(f'home = {python_home}')
                    else:
                        new_lines.append(line)
                
                with open(pyvenv_cfg, 'w') as f:
                    f.write('\n'.join(new_lines))
                    
            except Exception as e:
                logger.warning(f"Failed to fix pyvenv.cfg: {e}")
        
        # Fix activation script paths (Unix/Linux/macOS)
        activate_script = os.path.join(venv_path, "bin", "activate")
        if os.path.exists(activate_script):
            try:
                with open(activate_script, 'r') as f:
                    content = f.read()
                
                # Replace VIRTUAL_ENV path
                old_pattern = r'VIRTUAL_ENV="[^"]*"'
                new_content = re.sub(old_pattern, f'VIRTUAL_ENV="{venv_path}"', content)
                
                if new_content != content:
                    with open(activate_script, 'w') as f:
                        f.write(new_content)
                        
            except Exception as e:
                logger.warning(f"Failed to fix activate script: {e}")
        
        # Fix Windows activation script
        activate_bat = os.path.join(venv_path, "Scripts", "activate.bat")
        if os.path.exists(activate_bat):
            try:
                # Here you can add Windows-specific path fix logic
                # But usually Windows virtual environments are more portable
                pass
            except Exception as e:
                logger.warning(f"Warning when fixing virtual environment path: {e}")
                # Don't throw exception, because even if path fix fails, environment is usually still usable
    
    def clear_all_templates(self):
        """Clear all templates"""
        for template_id in list(self._templates.keys()):
            self._remove_template(template_id)
        
        self._templates.clear()
        self._save_templates()
    
    # Structured return environment template methods - for persistent_sandbox calls
    
    def save_environment_template_with_result(self, 
                                            venv_path: str, 
                                            template_name: str,
                                            description: str = "") -> Dict[str, Any]:
        """
        Save environment template (with structured return)
        
        Args:
            venv_path: Virtual environment path
            template_name: Template name
            description: Template description
            
        Returns:
            Dict containing operation result and template information
        """
        try:
            result = self.save_environment_template(venv_path, template_name, description)
            return result
        except Exception as e:
            logger.error(f"Failed to save environment template: {e}")
            return {
                "success": False,
                "error": f"Failed to save environment template: {str(e)}"
            }
    
    def load_environment_template_with_result(self, template_id: str, target_venv_path: str) -> Dict[str, Any]:
        """
        Load environment from template (with structured return)
        
        Args:
            template_id: Template ID
            target_venv_path: Target virtual environment path
            
        Returns:
            Dict containing operation result
        """
        try:
            result = self.deploy_from_template(template_id, target_venv_path)
            return result
        except Exception as e:
            logger.error(f"Failed to load environment template: {e}")
            return {
                "success": False,
                "error": f"Failed to load environment template: {str(e)}"
            }
    
    def list_environment_templates_with_result(self) -> Dict[str, Any]:
        """
        List all available environment templates (with structured return)
        
        Returns:
            Dict containing template list
        """
        try:
            templates = self.list_templates()
            return {
                "success": True,
                "templates": templates,
                "count": len(templates)
            }
        except Exception as e:
            logger.error(f"Failed to list environment templates: {e}")
            return {
                "success": False,
                "error": f"Failed to list environment templates: {str(e)}"
            }
