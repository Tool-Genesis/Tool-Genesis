import json
import re
import secrets
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

# ────────────────────────────────────────────────
# Retry decorator (unchanged)
# ────────────────────────────────────────────────
def async_retry(backoff=2, max_delay=60):
    import asyncio
    from functools import wraps
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay, attempt = backoff, 0
            while True:
                try:
                    attempt += 1
                    return await func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {attempt} failed: {e}. Retrying in {delay}s")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, max_delay)
        return wrapper
    return decorator

# ────────────────────────────────────────────────
# Lightweight container for one tool call
# ────────────────────────────────────────────────
class ToolCall:
    def __init__(self, call_id: str, name: str, args: Dict[str, Any]):
        self.id = call_id
        self.function = SimpleNamespace(name=name,
                                        arguments=json.dumps(args))
    def __repr__(self):
        return (f"ToolCall(id={self.id!r}, name={self.function.name!r}, "
                f"args={self.function.arguments!r})")

def _new_id() -> str:
    return "call_" + secrets.token_urlsafe(16)

# ────────────────────────────────────────────────
# Helpers: minimal unescape + robust attribute parsing
# ────────────────────────────────────────────────
def _unescape_attr_val(s: str) -> str:
    # minimal, targeted unescaping of typical cases in serialized prompts
    return s.replace(r'\"', '"').replace(r"\'", "'").replace(r'\\', '\\')

_name_attr_rx = re.compile(
    r'name\s*=\s*\\?(?P<q>["\'])'
    r'(?P<val>(?:\\.|(?!\1).)*?)'
    r'\\?(?P=q)',
    re.DOTALL | re.IGNORECASE
)

def _extract_name(attr_blob: str) -> Optional[str]:
    m = _name_attr_rx.search(attr_blob)
    if not m:
        return None
    return _unescape_attr_val(m.group('val'))

# <parameter name="..."> ... </parameter>  (support single/double quotes)
_param_rx = re.compile(
    r'<parameter\s+name\s*=\s*(?P<q>["\'])(?P<name>[^"\']+)(?P=q)\s*>'
    r'(?P<val>.*?)</\s*parameter\s*>',
    re.DOTALL | re.IGNORECASE
)

def _parse_xml_params(body: str) -> Dict[str, Any]:
    """Return dict of <parameter …>value</parameter> inside *body*."""
    out: Dict[str, Any] = {}
    for m in _param_rx.finditer(body):
        out[m.group('name')] = m.group('val').strip()
    return out

# ────────────────────────────────────────────────
# 1) <multi_tool_use.parallel> … </multi_tool_use.parallel>
#    parse inner <tool_use name="...">…</tool_use> (robust to escaped quotes)
# ────────────────────────────────────────────────
def parse_multi_tool_use(body: str) -> List[ToolCall]:
    calls: List[ToolCall] = []
    inner_rx = re.compile(
        r'<\s*tool_use\s+name\s*=\s*\\?(?P<q>["\'])'
        r'(?P<name>(?:\\.|(?!\1).)*?)\\?(?P=q)\s*>'
        r'(?P<body>.*?)</\s*tool_use\s*>',
        re.DOTALL | re.IGNORECASE
    )
    for m in inner_rx.finditer(body):
        raw_name = _unescape_attr_val(m.group('name'))
        tool = raw_name.split('.')[-1]
        args = _parse_xml_params(m.group('body'))
        calls.append(ToolCall(_new_id(), tool, args))
    return calls

# ────────────────────────────────────────────────
# 2) <tool_use …> … </tool_use>  (classic wrapper)
# ────────────────────────────────────────────────
def parse_single_tool_use(body: str, wrapper_name: str) -> List[ToolCall]:
    args = _parse_xml_params(body)
    return [ToolCall(_new_id(), wrapper_name.split('.')[-1], args)] if args else []

# ────────────────────────────────────────────────
# 3) <functions.xxx> … </functions.xxx>
# ────────────────────────────────────────────────
def parse_functions_tool_use(body: str,
                             tag_type: str,
                             attr_blob: str) -> List[ToolCall]:
    """
    • tool name = substring after the dot (functions.code_execution → code_execution)
    • args = opening-tag attributes  +  <parameter …> entries inside body
    • if no <parameter>, the *entire* body is returned as {'code': body.strip()}
    """
    tool_name = tag_type.split('.')[-1]
    # a) attributes on opening tag → initial args (support escaped & single quotes)
    attr_pairs = re.findall(
        r'(\w+)\s*=\s*\\?(["\'])(.*?)\\?\2',
        attr_blob, re.DOTALL
    )
    args: Dict[str, Any] = {k: _unescape_attr_val(v) for k, _, v in attr_pairs}

    # b) body
    body = body.strip()
    xml_args = _parse_xml_params(body)
    if xml_args:
        args.update(xml_args)
    elif body:
        args["code"] = body

    return [ToolCall(_new_id(), tool_name, args)]

# ────────────────────────────────────────────────
# Strip helpers for analysis/final & multi-blocks
# ────────────────────────────────────────────────
def _strip_analysis_final(text: str) -> str:
    return re.sub(r'<(analysis|final)\b[^>]*>.*?</\1\s*>',
                  '', text, flags=re.DOTALL | re.IGNORECASE)

_multi_block_rx = re.compile(
    r'<\s*multi_tool_use\.parallel\b[^>]*>(?P<body>.*?)</\s*multi_tool_use\.parallel\s*>',
    re.DOTALL | re.IGNORECASE
)

def _extract_and_strip_multi_blocks(text: str) -> (List[ToolCall], str):
    calls: List[ToolCall] = []
    def _repl(m: re.Match) -> str:
        calls.extend(parse_multi_tool_use(m.group('body')))
        return ''  # remove the whole multi block from text
    stripped = _multi_block_rx.sub(_repl, text)
    return calls, stripped

# ────────────────────────────────────────────────
# Generic scanners for remaining wrappers
# ────────────────────────────────────────────────
_functions_rx = re.compile(
    r'<\s*(functions\.[\w\-]+)\b([^>]*)>(.*?)</\s*\1\s*>',
    re.DOTALL | re.IGNORECASE
)

_single_tool_rx = re.compile(
    r'<\s*tool_use\b([^>]*)>(.*?)</\s*tool_use\s*>',
    re.DOTALL | re.IGNORECASE
)

def _scan_remaining_wrappers(text: str) -> List[ToolCall]:
    """
    Scan remaining text (after multi blocks stripped) in document order,
    capturing <functions.xxx> and classic <tool_use ...> blocks.
    """
    calls: List[ToolCall] = []
    # Combined opener to preserve original order
    open_rx = re.compile(
        r'<\s*(functions\.[\w\-]+|tool_use)\b([^>]*)>',
        re.DOTALL | re.IGNORECASE
    )
    pos = 0
    n = len(text)
    while True:
        m = open_rx.search(text, pos)
        if not m:
            break
        tag = m.group(1)
        attr_blob = m.group(2)
        body_start = m.end()
        close_rx = re.compile(rf'</\s*{re.escape(tag)}\s*>', re.DOTALL | re.IGNORECASE)
        m_close = close_rx.search(text, body_start)
        body = text[body_start:m_close.start()] if m_close else text[body_start:]

        if tag.startswith('functions.'):
            calls.extend(parse_functions_tool_use(body, tag, attr_blob))
        else:  # 'tool_use'
            name = _extract_name(attr_blob)
            if name:
                calls.extend(parse_single_tool_use(body, name))

        pos = (m_close.end() if m_close else n)
    return calls

# ────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────
def extract_tool_calls(text: str) -> List[ToolCall]:
    """
    Robust extraction:
      ① strip <analysis>/<final> blocks
      ② parse & strip ALL <multi_tool_use.parallel> blocks (collect inner <tool_use>)
      ③ on the remaining text, scan in order for <functions.xxx> and classic <tool_use>
      ④ return concatenated ToolCall list (multi first, then remaining in-order)
    """
    # 0) remove analysis/final regions
    text = _strip_analysis_final(text)

    # 1) parse & strip multi-tool blocks
    calls_multi, rest = _extract_and_strip_multi_blocks(text)

    # 2) scan remaining wrappers in original order
    calls_rest = _scan_remaining_wrappers(rest)

    # 3) combine
    return calls_multi + calls_rest

# NEW: scan + strip remaining wrappers (<functions.xxx> & <tool_use ...>)
def _scan_and_strip_remaining_wrappers(text: str) -> tuple[List[ToolCall], str]:
    """
    Parse the remaining text (AFTER multi-tool blocks have been removed) in
    document order and simultaneously strip matched blocks from the text.

    Captures:
      • <functions.xxx>...</functions.xxx>
      • <tool_use ...>...</tool_use>

    Returns:
      (calls, cleaned_text)
        calls        -> list of ToolCall extracted in order
        cleaned_text -> original text with the matched blocks removed
    """
    calls: List[ToolCall] = []
    # Unified opener to preserve original ordering between <functions.*> and <tool_use>
    open_rx = re.compile(
        r'<\s*(functions\.[\w\-]+|tool_use)\b([^>]*)>',
        re.DOTALL | re.IGNORECASE
    )

    parts: List[str] = []  # chunks of text to keep
    pos = 0
    n = len(text)

    while True:
        m = open_rx.search(text, pos)
        if not m:
            break

        tag = m.group(1)
        attr_blob = m.group(2)
        body_start = m.end()

        # Find the corresponding closing tag
        close_rx = re.compile(rf'</\s*{re.escape(tag)}\s*>', re.DOTALL | re.IGNORECASE)
        m_close = close_rx.search(text, body_start)

        # If no closing tag is found, stop stripping to avoid destroying content
        if not m_close:
            break

        body = text[body_start:m_close.start()]

        # Parse & collect the tool calls
        if tag.startswith('functions.'):
            calls.extend(parse_functions_tool_use(body, tag, attr_blob))
        else:  # classic <tool_use>
            name = _extract_name(attr_blob)
            if name:
                calls.extend(parse_single_tool_use(body, name))

        # Keep the text BEFORE the matched block
        parts.append(text[pos:m.start()])
        # Skip the matched block
        pos = m_close.end()

    # Append any trailing text after the last match
    parts.append(text[pos:])
    cleaned_text = ''.join(parts)
    return calls, cleaned_text


# NEW: public API that returns (calls, cleaned_text)
def extract_tool_calls_and_clean(text: str) -> tuple[List[ToolCall], str]:
    """
    Full pipeline (does NOT replace the existing extract_tool_calls):
      1) Strip <analysis>/<final> blocks
      2) Parse & remove ALL <multi_tool_use.parallel> blocks (collect inner <tool_use>)
      3) On the remaining text, parse & remove <functions.xxx> and classic <tool_use> in order
      4) Return (all_calls, cleaned_text)

    Args:
      text: raw assistant message containing potential tool-use wrappers

    Returns:
      (calls, cleaned_text)
        calls        -> concatenated list of ToolCall (multi-block calls first, then the rest in order)
        cleaned_text -> original text with all tool-use wrappers removed
    """

    # 1) parse & strip all <multi_tool_use.parallel> blocks
    calls_multi, rest = _extract_and_strip_multi_blocks(text)

    # 2) parse & strip remaining wrappers (<functions.*> and <tool_use ...>)
    calls_rest, cleaned = _scan_and_strip_remaining_wrappers(rest)

    # 3) combine results
    return calls_multi + calls_rest, cleaned


# Example usage (optional self-test)
if __name__ == "__main__":
    demo = '''
<multi_tool_use.parallel>
  <tool_use name="functions.browser_search">
    <parameter name="query">A</parameter>
    <parameter name="topn">3</parameter>
  </tool_use>
  <tool_use name='functions.browser_search'>
    <parameter name="query">B</parameter>
    <parameter name="topn">2</parameter>
  </tool_use>
</multi_tool_use.parallel>

<tool_use name="functions.file_tool">
  <parameter name="action">save</parameter>
  <parameter name="file_path">x.py</parameter>
  <parameter name="content">print("hi")</parameter>
</tool_use>

<functions.code_execution>
  <parameter name="code">print("run")</parameter>
</functions.code_execution>

(This free text should remain after cleaning)
<analysis>\nThe initial verification test for the environment has passed successfully. The game can be instantiated, the task description is returned, possible actions are generated, and the step function works as expected with valid return types. \n\nNext, I will proceed with more robust and edge case testing to ensure that the environment handles invalid actions, boundary conditions, and error states gracefully. This will include:\n- Handling invalid actions not in possibleActions\n- Trying to feed baby with milk at wrong temperature\n- Trying to take objects that don't exist or are not in inventory\n- Using thermometer on objects other than milk\n- Opening/closing objects that cannot be opened/closed\n- Turning on/off non-device objects\n- Ensuring scores update correctly and no state corruption occurs\n- Confirming reproducibility with fixed random seeds\n\nI will write a test script to cover these scenarios and execute it.\n\nOnce all robustness tests pass without errors or exceptions, I will produce the final deliverable code.\n\nNo code changes are needed at this stage unless tests fail.\n\n</analysis>
'''
    calls, cleaned = extract_tool_calls_and_clean(demo)
    for c in calls:
        print(c)
    print("\n--- CLEANED TEXT ---\n")
    print(cleaned.strip())