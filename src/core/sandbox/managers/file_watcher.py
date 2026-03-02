import logging
import threading
import time
import fnmatch
import os
from typing import Callable, Optional, List

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

class _DebounceEventHandler(FileSystemEventHandler):
    """An event handler that debounces and filters events."""

    def __init__(self, callback: Callable, debounce_delay: float, ignore_patterns: Optional[List[str]] = None):
        self.callback = callback
        self.debounce_delay = debounce_delay
        self.ignore_patterns = ignore_patterns or []
        self._timer: Optional[threading.Timer] = None

    def is_ignored(self, path: str) -> bool:
        """Check if a path should be ignored."""
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(os.path.basename(path), pattern) or \
               fnmatch.fnmatch(path, pattern):
                logger.info(f"[FileWatcher] Ignoring path: {path} (matched pattern: {pattern})")
                return True
        return False

    def on_any_event(self, event):
        """Catches all events, filters them, and then debounces."""
        if self.is_ignored(event.src_path):
            logger.debug(f"Ignoring event for path: {event.src_path}")
            return

        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(self.debounce_delay, self.callback)
        self._timer.start()

class FileWatcher:
    """A file watcher that triggers a callback on changes with debouncing and filtering."""

    def __init__(
        self,
        path: str,
        callback: Callable,
        debounce_delay: float = 1.0,
        ignore_patterns: Optional[List[str]] = None,
    ):
        self.path = path
        self.callback = callback
        self.debounce_delay = debounce_delay
        self.ignore_patterns = ignore_patterns
        self._observer: Optional[Observer] = None

    def start(self):
        """Starts the file watcher."""
        if self._observer is None:
            event_handler = _DebounceEventHandler(self.callback, self.debounce_delay, self.ignore_patterns)
            self._observer = Observer()
            self._observer.schedule(event_handler, self.path, recursive=True)
            self._observer.start()
            logger.info(f"Started watching {self.path} (ignoring: {self.ignore_patterns})")

    def stop(self):
        """Stops the file watcher."""
        if self._observer and self._observer.is_alive():
            self._observer.stop()
            self._observer.join()  # Wait for the observer to terminate
            logger.info(f"Stopped watching {self.path}")
        self._observer = None