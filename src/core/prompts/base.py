# src/prompts/base.py
from __future__ import annotations

import inspect
import platform
import string
import subprocess
import sys
from functools import wraps
from typing import Any, Callable, Dict, Optional, Set, TypeVar, Union

T = TypeVar("T")

__all__ = [
    "get_system_information",
    "get_prompt_template_key_words",
    "BaseInterpreter",
    "SubprocessInterpreter",
    "TextPrompt",
    "CodePrompt",
    "render_prompt",
    "as_text_prompt",
]

# -----------------------------------------------------------------------------
# System info & template utils
# -----------------------------------------------------------------------------
def get_system_information() -> Dict[str, str]:
    """Return a small set of system information for prompt context."""
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }


def get_prompt_template_key_words(template: str) -> Set[str]:
    """Extract placeholders like {name} from a format string safely."""
    keys: Set[str] = set()
    for _, field_name, _, _ in string.Formatter().parse(template):
        if field_name:
            keys.add(field_name)
    return keys


# -----------------------------------------------------------------------------
# Interpreter abstraction (used by CodePrompt.execute)
# -----------------------------------------------------------------------------
class BaseInterpreter:
    """Interface for code execution backends."""

    def run(self, code: str, code_type: Optional[str] = None, **kwargs: Any) -> str:
        raise NotImplementedError


class SubprocessInterpreter(BaseInterpreter):
    """
    Minimal subprocess-based interpreter.

    - code_type in {"python", None}: execute with current Python
    - code_type in {"bash", "sh"}: execute via system shell
    """

    def run(
        self,
        code: str,
        code_type: Optional[str] = None,
        timeout: Optional[int] = 120,
        cwd: Optional[str] = None,
        max_output_chars: Optional[int] = None,
        **_: Any,
    ) -> str:
        if code_type in (None, "", "python"):
            cmd = [sys.executable, "-c", code]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd
            )
            return self._fmt(proc, max_output_chars)

        if code_type in ("bash", "sh"):
            # NOTE: shell=True has security risks, external input needs strict validation
            proc = subprocess.run(
                code, shell=True, capture_output=True, text=True, timeout=timeout, cwd=cwd
            )
            return self._fmt(proc, max_output_chars)

        return f"[SubprocessInterpreter] Unsupported code_type: {code_type}"

    @staticmethod
    def _truncate(s: str, n: Optional[int]) -> str:
        if n is None or n <= 0 or len(s) <= n:
            return s
        return s[: n - 3] + "..."

    @classmethod
    def _fmt(cls, proc: subprocess.CompletedProcess, max_output_chars: Optional[int]) -> str:
        out = []
        if proc.stdout:
            out.append("STDOUT:\n" + cls._truncate(proc.stdout, max_output_chars))
        if proc.stderr:
            out.append("STDERR:\n" + cls._truncate(proc.stderr, max_output_chars))
        out.append(f"Return code: {proc.returncode}")
        return "\n".join(out).strip()


# -----------------------------------------------------------------------------
# Prompt classes
# -----------------------------------------------------------------------------
def return_prompt_wrapper(
    cls: Any,
    func: Callable,
) -> Callable[..., Union[Any, tuple]]:
    """If a wrapped method returns str (or tuple[str,...]), coerce to cls."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Union[Any, tuple]:
        result = func(*args, **kwargs)
        if isinstance(result, str) and not isinstance(result, cls):
            return cls(result)
        if isinstance(result, tuple):
            new_result = tuple(
                cls(item) if isinstance(item, str) and not isinstance(item, cls) else item
                for item in result
            )
            return new_result
        return result
    return wrapper


def wrap_prompt_functions(cls: T) -> T:
    """
    Decorator: auto-wrap class methods so any str return value becomes `cls`.
    - Skip property / descriptor
    - Preserve staticmethod/classmethod descriptor semantics
    """
    excluded = {"__init__", "__new__", "__str__", "__repr__"}
    for name, attr in list(cls.__dict__.items()):
        if name in excluded or name.startswith("__"):
            continue
        # Skip property / descriptor
        if isinstance(attr, property):
            continue

        # classmethod / staticmethod need to extract underlying function then wrap
        if isinstance(attr, classmethod):
            fn = attr.__func__
            wrapped = classmethod(return_prompt_wrapper(cls, fn))
            setattr(cls, name, wrapped)
            continue
        if isinstance(attr, staticmethod):
            fn = attr.__func__
            wrapped = staticmethod(return_prompt_wrapper(cls, fn))
            setattr(cls, name, wrapped)
            continue

        # Regular function/method
        if inspect.isfunction(attr) or inspect.ismethod(attr):
            setattr(cls, name, return_prompt_wrapper(cls, attr))

    return cls


@wrap_prompt_functions
class TextPrompt(str):
    """A thin str subclass with tolerant `.format()` and placeholder introspection."""

    @property
    def key_words(self) -> Set[str]:
        # Use local utility function
        return get_prompt_template_key_words(self)

    def format(self, *args: Any, **kwargs: Any) -> "TextPrompt":
        """
        Tolerant format: missing keys remain as {key} instead of KeyError.
        """
        default_kwargs = {key: f"{{{key}}}" for key in self.key_words}
        default_kwargs.update(kwargs)
        return TextPrompt(super().format(*args, **default_kwargs))


@wrap_prompt_functions
class CodePrompt(TextPrompt):
    """
    Code prompt with optional `code_type` and an `execute()` helper.
    `code_type`: "python" (default) or "bash"/"sh".
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "CodePrompt":
        code_type = kwargs.pop("code_type", None)
        inst = super().__new__(cls, *args, **kwargs)
        inst._code_type = code_type
        return inst

    @property
    def code_type(self) -> Optional[str]:
        return getattr(self, "_code_type", None)

    def set_code_type(self, code_type: str) -> None:
        self._code_type = code_type

    def execute(self, interpreter: Optional[BaseInterpreter] = None, **kwargs: Any) -> str:
        interp = interpreter or SubprocessInterpreter()
        return interp.run(self, self._code_type, **kwargs)


def render_prompt(template: Union[str, TextPrompt], **kwargs: Any) -> str:
    """
    Render a prompt with tolerant formatting.
    Accepts both plain strings and TextPrompt instances.
    """
    tpl = TextPrompt(template) if isinstance(template, str) else template
    return tpl.format(**kwargs)


def as_text_prompt(value: Union[str, TextPrompt]) -> TextPrompt:
    """Coerce any string-like value to TextPrompt."""
    return value if isinstance(value, TextPrompt) else TextPrompt(value)