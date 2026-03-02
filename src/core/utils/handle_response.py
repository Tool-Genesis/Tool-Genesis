import json
import re
from typing import Any, Dict, List, Optional, Tuple, Type, Union

try:
    # Optional: only used if enable_yaml=True and PyYAML is installed.
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

try:
    # Optional: if the caller passes a Pydantic model class
    from pydantic import BaseModel, ValidationError  # type: ignore
    _HAS_PYDANTIC = True
except Exception:
    _HAS_PYDANTIC = False
    BaseModel = object  # type: ignore
    class ValidationError(Exception): ...  # fallback


def _extract_last_message_text(response: Any) -> str:
    """
    Get the textual content we should parse from a ChatAgent-like response.
    Falls back to str(...) if structure is unknown.

    Expected (but not required):
      - response.msgs: list of message objects
      - last message has .content (str or dict/list)
    """
    # Common pattern: response.msgs is a list; last message is target
    msgs = getattr(response, "msgs", None)
    if msgs and isinstance(msgs, list) and len(msgs) > 0:
        last_msg = msgs[-1]
        content = getattr(last_msg, "content", "")
        if isinstance(content, (dict, list)):
            try:
                return json.dumps(content, ensure_ascii=False)
            except Exception:
                return str(content)
        return str(content)

    # If response itself looks like a dict/list/string
    if isinstance(response, (dict, list)):
        try:
            return json.dumps(response, ensure_ascii=False)
        except Exception:
            return str(response)
    if isinstance(response, str):
        return response

    # Fallback
    return str(getattr(response, "content", ""))


def _slice_by_final_tags(
    text: str,
    final_mode: str = "last",  # 'last' | 'first' | 'all' | 'none'
) -> Tuple[List[str], str]:
    """
    Extract segments inside <final>...</final> according to final_mode.
    Returns (segments, prefix_text).

    - If final_mode == 'none' or no <final> tags found: segments = [text], prefix_text = "".
    - If 'first': segments = [first_final_content]
    - If 'last':  segments = [last_final_content]
    - If 'all':   segments = [each_final_content, ...]
    """
    pattern = re.compile(r"<final>(.*?)</final>", flags=re.S | re.I)
    all_matches = pattern.findall(text)
    prefix_text = ""

    if final_mode.lower() == "none" or not all_matches:
        return [text], prefix_text

    if final_mode.lower() == "first":
        return [all_matches[0].strip()], prefix_text
    if final_mode.lower() == "last":
        return [all_matches[-1].strip()], prefix_text
    if final_mode.lower() == "all":
        return [m.strip() for m in all_matches], prefix_text

    # default to 'last' if unknown mode
    return [all_matches[-1].strip()], prefix_text


def _strip_fences(
    segment: str,
    fence_mode: str = "last",  # 'last' | 'first' | 'all' | 'none'
    joiner: str = "\n\n---\n\n",
) -> str:
    """
    Extract fenced code blocks (```...```) according to fence_mode.
    - If 'none' or no fences found: return original segment.
    - If 'first'/'last': return that block's inner content.
    - If 'all': join all block inner contents with joiner.
    """
    if fence_mode.lower() == "none":
        return segment

    # Match code fences: ```lang? + content + ```
    # lang is optional, content is captured
    blocks = re.findall(r"```(?:[a-zA-Z0-9_-]+)?\s*([\s\S]*?)```", segment, flags=re.I)
    if not blocks:
        return segment

    if fence_mode.lower() == "first":
        return blocks[0].strip()
    if fence_mode.lower() == "last":
        return blocks[-1].strip()
    if fence_mode.lower() == "all":
        return joiner.join(b.strip() for b in blocks)

    # default to 'last'
    return blocks[-1].strip()


def _maybe_parse_structured(
    text: str,
    *,
    auto_json: bool = True,
    enable_yaml: bool = False,
) -> Union[str, Dict[str, Any], List[Any]]:
    """
    Try to parse text into JSON (and YAML if enabled). If parsing fails, return the original string.
    """
    if auto_json:
        try:
            return json.loads(text)
        except Exception:
            pass

    if enable_yaml and _HAS_YAML:
        try:
            data = yaml.safe_load(text)  # type: ignore
            # Only return structured types; if it's a scalar, keep original text
            if isinstance(data, (dict, list)):
                return data
        except Exception:
            pass

    return text


def extract_all_finals(
    response: Any,
    *,
    strip_fence: bool = True,
    fence_mode: str = "all",      # 'last' | 'first' | 'all' | 'none'
    joiner: str = "\n\n---\n\n",
    auto_json: bool = False,      # default False to avoid mis-parsing YAML as JSON
    enable_yaml: bool = False,    # set True to parse YAML if PyYAML is installed
) -> List[Union[str, Dict[str, Any], List[Any]]]:
    """
    Return a list with ALL <final>...</final> contents from the last message, optionally
    stripped to fenced blocks and optionally parsed to JSON/YAML.

    Typical usage:
        finals = extract_all_finals(resp, strip_fence=True, fence_mode="all", auto_json=False)

    Parameters
    ----------
    response : Any
        Chat agent response object OR raw text; expected to have response.msgs[-1].content.
    strip_fence : bool
        If True, extract fenced code blocks from each final segment according to `fence_mode`.
    fence_mode : str
        'last' | 'first' | 'all' | 'none' — how to pick blocks when multiple fences exist.
    joiner : str
        Joiner used when fence_mode == 'all'.
    auto_json : bool
        Try json.loads() on each segment after fence processing.
    enable_yaml : bool
        If True and PyYAML is installed, attempt yaml.safe_load() when JSON parsing fails.

    Returns
    -------
    List[Union[str, dict, list]]
        One element per <final> segment (in the order they appear).
        Each element is either a string (raw) or a parsed JSON/YAML object.
    """
    text = _extract_last_message_text(response)
    segments, _ = _slice_by_final_tags(text, final_mode="all")

    processed: List[Union[str, Dict[str, Any], List[Any]]] = []
    for seg in segments:
        seg_clean = _strip_fences(seg, fence_mode=fence_mode, joiner=joiner) if strip_fence else seg
        parsed = _maybe_parse_structured(seg_clean, auto_json=auto_json, enable_yaml=enable_yaml)
        processed.append(parsed)
    return processed


def extract_payload_from_response(
    response: Any,
    model_cls: Optional[Type[BaseModel]] = None,
    *,
    final_mode: str = "last",     # 'last' | 'first' | 'all' | 'none'
    fence_mode: str = "last",     # 'last' | 'first' | 'all' | 'none'
    joiner: str = "\n\n---\n\n",
    strip_fence: bool = True,
    auto_json: bool = True,
    enable_yaml: bool = False,
    return_mode: str = "auto",    # 'auto' | 'list' | 'dict' | 'string'
) -> Union[BaseModel, Dict[str, Any], List[Any], str]:
    """
    A flexible extractor that replaces the original single-<final> behavior.

    - You can set final_mode='all' to capture EVERY <final> segment.
    - fence_mode controls how multiple fenced blocks inside a segment are handled.
    - return_mode:
        'auto'   -> return a single item if 1 segment, else a list of items
        'list'   -> always return a list
        'dict'   -> return {'finals': [...], 'raw_text': original_text}
        'string' -> force a single string (join with `joiner` if multiple)

    Pydantic:
      If `model_cls` is provided and final_mode != 'all':
        - Try to validate the single parsed payload against model_cls.
      If final_mode == 'all':
        - Return a list; each element is either validated (if dict/list) or raw string. If an element
          fails validation, keep the original parsed object/string in the list.

    Backward compatibility:
      Defaults (final_mode='last', fence_mode='last') emulate the old “take only the last <final> and last fence block”.

    Parameters
    ----------
    response, model_cls, final_mode, fence_mode, joiner, strip_fence, auto_json, enable_yaml, return_mode
      See functions above for semantics.

    Returns
    -------
    Union[BaseModel, dict, list, str]
    """
    original_text = _extract_last_message_text(response)

    # If model_cls is provided and the last message already has a parsed object, try that first.
    if model_cls is not None and _HAS_PYDANTIC:
        msgs = getattr(response, "msgs", None)
        if msgs and isinstance(msgs, list) and len(msgs) > 0:
            last_msg = msgs[-1]
            parsed_obj = getattr(last_msg, "parsed", None)
            if parsed_obj is not None:
                # If it's already an instance of the desired model, return it
                if isinstance(parsed_obj, model_cls):
                    return parsed_obj
                # Otherwise, try to coerce via model_dump or direct dict/list
                try:
                    if hasattr(parsed_obj, "model_dump"):
                        return model_cls.model_validate(parsed_obj.model_dump())
                    if isinstance(parsed_obj, (dict, list)):
                        return model_cls.model_validate(parsed_obj)
                except ValidationError:
                    pass  # fall through to text parsing

    # Gather segments as per final_mode
    segments, _ = _slice_by_final_tags(original_text, final_mode=final_mode)

    # Process fences and parse
    parsed_segments: List[Union[str, Dict[str, Any], List[Any]]] = []
    for seg in segments:
        seg_clean = _strip_fences(seg, fence_mode=fence_mode, joiner=joiner) if strip_fence else seg
        parsed = _maybe_parse_structured(seg_clean, auto_json=auto_json, enable_yaml=enable_yaml)
        parsed_segments.append(parsed)

    # If Pydantic model requested, try to validate segment(s)
    if model_cls is not None and _HAS_PYDANTIC:
        if final_mode.lower() == "all":
            validated_list: List[Any] = []
            for item in parsed_segments:
                if isinstance(item, (dict, list)):
                    try:
                        validated_list.append(model_cls.model_validate(item))
                        continue
                    except ValidationError:
                        pass
                validated_list.append(item)  # keep as-is if validation fails
            # Return according to return_mode
            if return_mode == "dict":
                return {"finals": validated_list, "raw_text": original_text}
            if return_mode == "string":
                # Best-effort string join
                return joiner + joiner.join([json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else str(x) for x in validated_list])
            return validated_list  # 'auto' or 'list'
        else:
            # Single-segment case
            item = parsed_segments[0] if parsed_segments else ""
            if isinstance(item, (dict, list)):
                try:
                    return model_cls.model_validate(item)
                except ValidationError:
                    pass
            return item  # return raw parsed object/string

    # No model_cls case: format per return_mode
    if final_mode.lower() == "all":
        if return_mode == "dict":
            return {"finals": parsed_segments, "raw_text": original_text}
        if return_mode == "string":
            # Join all segments into a single string
            def _to_str(x: Any) -> str:
                if isinstance(x, (dict, list)):
                    try:
                        return json.dumps(x, ensure_ascii=False)
                    except Exception:
                        return str(x)
                return str(x)
            return joiner.join(_to_str(x) for x in parsed_segments)
        # 'auto' or 'list'
        if return_mode in ("auto", "list"):
            return parsed_segments

    # Not 'all' mode → single segment semantics
    single = parsed_segments[0] if parsed_segments else ""
    if return_mode == "dict":
        return {"finals": [single], "raw_text": original_text}
    if return_mode == "list":
        return [single]
    if return_mode == "string":
        return str(single)
    # 'auto'
    return single