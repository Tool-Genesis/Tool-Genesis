"""
static_checker.py - Environment Code Static Checker

This module provides static checks for generated environment code, including:
1. Function and method implementation completeness checks
2. JSON schema alignment checks
3. Code structure compliance checks
4. Dependency checks

Author: Assistant 2025-01-27
"""
from __future__ import annotations

import ast
import json
import re
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set, Union
from dataclasses import dataclass
from enum import Enum

import jsonschema
from jsonschema import validate, ValidationError


class IssueLevel(Enum):
    """Issue severity"""
    ERROR = "error"      # Critical error; prevents code from running
    WARNING = "warning"  # Warning; may affect functionality
    INFO = "info"        # Info; recommended improvements


@dataclass
class StaticCheckIssue:
    """Static check issue"""
    level: IssueLevel
    category: str
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    function_name: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class StaticCheckResult:
    """Static check result"""
    passed: bool
    issues: List[StaticCheckIssue]
    summary: Dict[str, int]
    
    def add_issue(self, issue: StaticCheckIssue):
        """Add issue"""
        self.issues.append(issue)
        self.summary[issue.level.value] = self.summary.get(issue.level.value, 0) + 1
        if issue.level == IssueLevel.ERROR:
            self.passed = False


class FunctionImplementationChecker:
    """Function implementation checker"""
    
    def __init__(self):
        self.stub_patterns = [
            r'pass\s*$',
            r'raise\s+NotImplementedError',
            r'return\s+None\s*$',
            r'return\s*$',
            r'TODO',
            r'FIXME',
            r'placeholder',
            r'not\s+implemented',
        ]
        self.stub_regex = re.compile('|'.join(self.stub_patterns), re.IGNORECASE | re.MULTILINE)
    
    def check_file(self, file_path: Union[str, Path]) -> List[StaticCheckIssue]:
        """Check function implementations in a file"""
        issues = []
        file_path = Path(file_path)
        
        if not file_path.exists():
            issues.append(StaticCheckIssue(
                level=IssueLevel.ERROR,
                category="file_missing",
                message=f"File not found: {file_path}",
                file_path=str(file_path)
            ))
            return issues
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content, filename=str(file_path))
            
            # Check classes and functions
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    issues.extend(self._check_function(node, content, str(file_path)))
                elif isinstance(node, ast.ClassDef):
                    issues.extend(self._check_class(node, content, str(file_path)))
                    
        except SyntaxError as e:
            issues.append(StaticCheckIssue(
                level=IssueLevel.ERROR,
                category="syntax_error",
                message=f"Syntax error: {e.msg}",
                file_path=str(file_path),
                line_number=e.lineno
            ))
        except Exception as e:
            issues.append(StaticCheckIssue(
                level=IssueLevel.ERROR,
                category="parse_error",
                message=f"Parse error: {str(e)}",
                file_path=str(file_path)
            ))
        
        return issues
    
    def _check_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], 
                       content: str, file_path: str) -> List[StaticCheckIssue]:
        """Check a single function"""
        issues = []
        
        # Get function source code
        lines = content.split('\n')
        func_lines = lines[node.lineno-1:node.end_lineno] if hasattr(node, 'end_lineno') else []
        func_source = '\n'.join(func_lines)
        
        # Check for stub or placeholder implementation
        if self._is_stub_implementation(func_source):
            issues.append(StaticCheckIssue(
                level=IssueLevel.WARNING,
                category="stub_implementation",
                message=f"Function '{node.name}' may be a placeholder implementation",
                file_path=file_path,
                line_number=node.lineno,
                function_name=node.name
            ))
        
        # Check if function body is empty
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            issues.append(StaticCheckIssue(
                level=IssueLevel.WARNING,
                category="empty_function",
                message=f"Function '{node.name}' only contains a pass statement",
                file_path=file_path,
                line_number=node.lineno,
                function_name=node.name
            ))
        
        # Check docstring
        if not ast.get_docstring(node):
            issues.append(StaticCheckIssue(
                level=IssueLevel.INFO,
                category="missing_docstring",
                message=f"Function '{node.name}' is missing a docstring",
                file_path=file_path,
                line_number=node.lineno,
                function_name=node.name
            ))
        
        return issues
    
    def _check_class(self, node: ast.ClassDef, content: str, file_path: str) -> List[StaticCheckIssue]:
        """Check class implementation"""
        issues = []
        
        # Check whether class defines methods
        methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if not methods:
            issues.append(StaticCheckIssue(
                level=IssueLevel.INFO,
                category="empty_class",
                message=f"Class '{node.name}' defines no methods",
                file_path=file_path,
                line_number=node.lineno
            ))
        
        # Check whether it inherits UnifiedBaseEnv
        base_names = [base.id for base in node.bases if isinstance(base, ast.Name)]
        if 'UnifiedBaseEnv' not in base_names:
            issues.append(StaticCheckIssue(
                level=IssueLevel.WARNING,
                category="inheritance_issue",
                message=f"Class '{node.name}' does not inherit UnifiedBaseEnv",
                file_path=file_path,
                line_number=node.lineno
            ))
        
        return issues
    
    def _is_stub_implementation(self, source: str) -> bool:
        """Check whether it's a placeholder implementation"""
        return bool(self.stub_regex.search(source))


class SchemaAlignmentChecker:
    """JSON Schema alignment checker"""
    
    def __init__(self):
        self.required_methods = {
            '_get_environment_state',
            '_reset_environment_state', 
            '_initialize_mcp_server'
        }
    
    def check_schema_alignment(self, env_file: Union[str, Path], 
                             schema_file: Union[str, Path]) -> List[StaticCheckIssue]:
        """Check alignment between environment code and JSON schema"""
        issues = []
        env_file = Path(env_file)
        schema_file = Path(schema_file)
        
        # Check file existence
        if not env_file.exists():
            issues.append(StaticCheckIssue(
                level=IssueLevel.ERROR,
                category="file_missing",
                message=f"Environment file not found: {env_file}",
                file_path=str(env_file)
            ))
            return issues
        
        if not schema_file.exists():
            issues.append(StaticCheckIssue(
                level=IssueLevel.ERROR,
                category="file_missing", 
                message=f"Schema file not found: {schema_file}",
                file_path=str(schema_file)
            ))
            return issues
        
        try:
            # Load schema
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            
            # Parse environment code
            with open(env_file, 'r', encoding='utf-8') as f:
                env_content = f.read()
            
            tree = ast.parse(env_content, filename=str(env_file))
            
            # Check schema alignment
            issues.extend(self._check_state_schema_alignment(tree, schema, str(env_file)))
            issues.extend(self._check_tool_schema_alignment(tree, schema, str(env_file)))
            
        except json.JSONDecodeError as e:
            issues.append(StaticCheckIssue(
                level=IssueLevel.ERROR,
                category="json_error",
                message=f"Schema file has invalid JSON format: {e.msg}",
                file_path=str(schema_file)
            ))
        except Exception as e:
            issues.append(StaticCheckIssue(
                level=IssueLevel.ERROR,
                category="schema_check_error",
                message=f"Schema check error: {str(e)}",
                file_path=str(env_file)
            ))
        
        return issues
    
    def _check_state_schema_alignment(self, tree: ast.AST, schema: Dict, 
                                    file_path: str) -> List[StaticCheckIssue]:
        """Check state schema alignment"""
        issues = []
        
        # Find state-related methods
        state_methods = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in self.required_methods:
                state_methods[node.name] = node
        
        # Check if required methods exist
        for method_name in self.required_methods:
            if method_name not in state_methods:
                issues.append(StaticCheckIssue(
                    level=IssueLevel.ERROR,
                    category="missing_method",
                    message=f"Required method missing: {method_name}",
                    file_path=file_path
                ))
        
        # Check state schema structure
        if 'state_schema' in schema:
            state_schema = schema['state_schema']
            if 'properties' in state_schema:
                issues.extend(self._validate_state_properties(
                    state_schema['properties'], tree, file_path
                ))
        
        return issues
    
    def _check_tool_schema_alignment(self, tree: ast.AST, schema: Dict,
                                   file_path: str) -> List[StaticCheckIssue]:
        """Check tool schema alignment"""
        issues = []
        
        # Collect all defined methods
        defined_methods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                defined_methods.add(node.name)
        
        # Check whether tool methods are implemented
        if 'tool_strategies' in schema:
            for tool_name in schema['tool_strategies']:
                # Convert tool name to method name
                method_name = self._tool_name_to_method_name(tool_name)
                if method_name not in defined_methods:
                    issues.append(StaticCheckIssue(
                        level=IssueLevel.WARNING,
                        category="missing_tool_method",
                        message=f"Tool '{tool_name}' corresponding method '{method_name}' is not implemented",
                        file_path=file_path
                    ))
        
        return issues
    
    def _validate_state_properties(self, properties: Dict, tree: ast.AST,
                                 file_path: str) -> List[StaticCheckIssue]:
        """Validate state properties"""
        issues = []
        
        # Find attribute initialization in __init__
        init_method = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                init_method = node
                break
        
        if not init_method:
            issues.append(StaticCheckIssue(
                level=IssueLevel.WARNING,
                category="missing_init",
                message="Missing __init__ method",
                file_path=file_path
            ))
            return issues
        
        # Check attribute initialization
        initialized_attrs = set()
        for node in ast.walk(init_method):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                        if target.value.id == 'self':
                            initialized_attrs.add(target.attr)
        
        # Check whether properties defined in schema are initialized
        for prop_name in properties:
            if prop_name not in initialized_attrs:
                issues.append(StaticCheckIssue(
                    level=IssueLevel.WARNING,
                    category="missing_property_init",
                    message=f"Property defined in schema '{prop_name}' is not initialized in __init__",
                    file_path=file_path
                ))
        
        return issues
    
    def _tool_name_to_method_name(self, tool_name: str) -> str:
        """Convert tool name to method name"""
        # Simple conversion rules; adjust as needed
        return tool_name.replace('-', '_').replace(' ', '_').lower()


class DependencyChecker:
    """Dependency checker"""
    
    def check_imports(self, file_path: Union[str, Path]) -> List[StaticCheckIssue]:
        """Check import dependencies"""
        issues = []
        file_path = Path(file_path)
        
        if not file_path.exists():
            return [StaticCheckIssue(
                level=IssueLevel.ERROR,
                category="file_missing",
                message=f"File not found: {file_path}",
                file_path=str(file_path)
            )]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            
            # Collect all imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            imports.append(f"{node.module}.{alias.name}")
            
            # Check key dependencies
            required_imports = [
                'src.utils.unified_base_env',
                'json',
                'pathlib'
            ]
            
            for required in required_imports:
                if not any(required in imp for imp in imports):
                    issues.append(StaticCheckIssue(
                        level=IssueLevel.WARNING,
                        category="missing_import",
                        message=f"Possibly missing required import: {required}",
                        file_path=str(file_path)
                    ))
            
        except Exception as e:
            issues.append(StaticCheckIssue(
                level=IssueLevel.ERROR,
                category="import_check_error",
                message=f"Import check error: {str(e)}",
                file_path=str(file_path)
            ))
        
        return issues


class StaticChecker:
    """Environment static checker (main class)"""
    
    def __init__(self):
        self.function_checker = FunctionImplementationChecker()
        self.schema_checker = SchemaAlignmentChecker()
        self.dependency_checker = DependencyChecker()
    
    def check_environment(self, env_dir: Union[str, Path], 
                         schema_file: Optional[Union[str, Path]] = None) -> StaticCheckResult:
        """Check environment directory"""
        result = StaticCheckResult(passed=True, issues=[], summary={})
        env_dir = Path(env_dir)
        
        if not env_dir.exists():
            result.add_issue(StaticCheckIssue(
                level=IssueLevel.ERROR,
                category="directory_missing",
                message=f"Environment directory does not exist: {env_dir}"
            ))
            return result
        
        # Find Python files
        py_files = list(env_dir.glob("*.py"))
        if not py_files:
            result.add_issue(StaticCheckIssue(
                level=IssueLevel.WARNING,
                category="no_python_files",
                message=f"No Python files found in environment directory: {env_dir}"
            ))
            return result
        
        # Check each Python file
        for py_file in py_files:
            # Function implementation check
            issues = self.function_checker.check_file(py_file)
            for issue in issues:
                result.add_issue(issue)
            
            # Dependency check
            issues = self.dependency_checker.check_imports(py_file)
            for issue in issues:
                result.add_issue(issue)
            
            # Schema alignment check
            if schema_file:
                issues = self.schema_checker.check_schema_alignment(py_file, schema_file)
                for issue in issues:
                    result.add_issue(issue)
        
        return result
    
    def check_single_file(self, file_path: Union[str, Path],
                         schema_file: Optional[Union[str, Path]] = None) -> StaticCheckResult:
        """Check a single file"""
        result = StaticCheckResult(passed=True, issues=[], summary={})
        
        # Function implementation check
        issues = self.function_checker.check_file(file_path)
        for issue in issues:
            result.add_issue(issue)
        
        # Dependency check
        issues = self.dependency_checker.check_imports(file_path)
        for issue in issues:
            result.add_issue(issue)
        
        # Schema alignment check
        if schema_file:
            issues = self.schema_checker.check_schema_alignment(file_path, schema_file)
            for issue in issues:
                result.add_issue(issue)
        
        return result


def format_check_result(result: StaticCheckResult) -> str:
    """Format check results into a readable string"""
    lines = []
    
    # Summary
    lines.append("=" * 60)
    lines.append("Environment Static Check Result")
    lines.append("=" * 60)
    lines.append(f"Status: {'Passed' if result.passed else 'Failed'}")
    lines.append(f"Total issues: {len(result.issues)}")
    
    if result.summary:
        lines.append("\nIssue summary:")
        for level, count in result.summary.items():
            lines.append(f"  {level}: {count}")
    
    # Detailed issues
    if result.issues:
        lines.append("\nDetailed issues:")
        lines.append("-" * 60)
        
        for i, issue in enumerate(result.issues, 1):
            lines.append(f"\n{i}. [{issue.level.value.upper()}] {issue.category}")
            lines.append(f"   Message: {issue.message}")
            if issue.file_path:
                lines.append(f"   File: {issue.file_path}")
            if issue.line_number:
                lines.append(f"   Line: {issue.line_number}")
            if issue.function_name:
                lines.append(f"   Function: {issue.function_name}")
    
    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
