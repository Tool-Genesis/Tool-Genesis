from pathlib import Path
import threading
import tempfile
import os
from typing import Optional, Dict, Any, Union
from unidiff import PatchSet, UnidiffParseError

class TextFileHandler:
    def write(
        self,
        path: Path,
        content: Optional[Union[str, bytes]]
    ) -> None:
        """
        Overwrite the file at `path` with the given content.

        Args:
            path (Path): The target file path.
            content (str | bytes | None):
                - str: encoded as UTF-8
                - bytes: written verbatim
                - None: creates an empty file

        Raises:
            TypeError: If content is not str, bytes, or None.
            OSError: On any filesystem I/O error.
        """
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Normalize to bytes
        if content is None:
            data = b""
        elif isinstance(content, str):
            data = content.encode("utf-8")
        elif isinstance(content, bytes):
            data = content
        else:
            raise TypeError(f"Unsupported content type: {type(content)}")

        # Overwrite the file
        with open(path, "wb") as f:
            f.write(data)
    def read(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    
    def edit(
            self,
            path: Path,
            patch: str,
            desc: Optional[str] = None
        ) -> Dict[str, Any]:
        """
        Apply a unified-diff to the file at `path`, returning whether it changed
        and echoing back the diff.

        Args:
            path:  Absolute Path to the file to patch.
            patch: Unified-diff text (---/+++/@@/-/+ lines).
            desc:  Optional label used in diff headers (defaults to path.as_posix()).

        Returns:
            A dict with:
            'changed': True if any hunks added or removed lines.
            'diff':    The original patch text.

        Raises:
            OSError:    If the diff is invalid or I/O errors occur.
        """
        # Ensure only one thread edits this file at a time
        lock = self._locks.setdefault(path, threading.Lock())
        with lock:
            # 1. Parse the unified-diff
            try:
                patch_set = PatchSet.from_string(patch)
            except UnidiffParseError as e:
                raise OSError(f"Invalid unified diff: {e}")

            # 2. Determine the a/ and b/ prefixes for this file
            label = desc or path.as_posix()
            prefix_a = f"a/{label}"
            prefix_b = f"b/{label}"

            # 3. Select only the hunks that target this file
            relevant = [
                pf for pf in patch_set
                if pf.source_file in (prefix_a, prefix_b)
                or pf.target_file in (prefix_a, prefix_b)
            ]
            if not relevant:
                # No hunks for this file => no change
                return {"changed": False, "diff": patch}

            # 4. Read the original file lines
            original_text = path.read_text(encoding="utf-8")
            lines = original_text.splitlines(keepends=True)

            merged = lines[:]  # Start with the original
            changed = False

            # 5. Apply each PatchedFile entry
            for pf in relevant:
                # Handle files created by the patch
                if pf.is_added_file:
                    merged = [ln.value for h in pf for ln in h if ln.is_added]
                    changed = True
                    continue
                # Handle files deleted by the patch
                if pf.is_removed_file:
                    merged = []
                    changed = True
                    continue

                # For modified files, apply each hunk in sequence
                new_merged = []
                idx = 0
                for hunk in pf:
                    # Copy unchanged lines up to the hunk start
                    while idx < hunk.source_start - 1 and idx < len(merged):
                        new_merged.append(merged[idx])
                        idx += 1
                    # Add context and added lines
                    for ln in hunk:
                        if ln.is_context or ln.is_added:
                            new_merged.append(ln.value)
                    # Skip removed lines
                    idx = min(idx + hunk.source_length, len(merged))
                    if hunk.added or hunk.removed:
                        changed = True
                # Append any remaining lines after the last hunk
                new_merged.extend(merged[idx:])
                merged = new_merged

            # 6. If changed, write back atomically
            if changed:
                new_text = "".join(merged)
                fd, tmp_path_str = tempfile.mkstemp(suffix=path.suffix, dir=str(path.parent))
                os.close(fd)
                tmp_path = Path(tmp_path_str)
                try:
                    tmp_path.write_text(new_text, encoding="utf-8")
                    os.replace(tmp_path, path)
                finally:
                    tmp_path.unlink(missing_ok=True)

            return {"changed": changed, "diff": patch}
