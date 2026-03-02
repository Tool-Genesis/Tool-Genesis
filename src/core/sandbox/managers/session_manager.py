"""
Session Manager - responsible for managing sandbox session lifecycle
"""
import threading
import time
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class SessionConfig:
    """Session configuration"""
    session_id: Optional[str] = None
    timeout_minutes: int = 60
    auto_cleanup: bool = True
    debug: bool = True
    
    def __post_init__(self):
        if self.session_id is None:
            self.session_id = str(uuid.uuid4())


class SessionManager:
    """Session Manager - responsible for managing session lifecycle, timeout and cleanup"""
    
    def __init__(self, config: SessionConfig, cleanup_callback: Optional[Callable] = None):
        """
        Initialize session manager
        
        Args:
            config: Session configuration
            cleanup_callback: Cleanup callback function, called when session times out
        """
        self.config = config
        self.cleanup_callback = cleanup_callback
        
        # Session state
        self.session_id = self.config.session_id
        self.created_at = datetime.now()
        self.last_used = datetime.now()
        self.last_accessed = time.time()
        self.is_active = True
        
        # Timer management
        self.cleanup_timer: Optional[threading.Timer] = None
        self.lock = threading.RLock()
        
        # Start auto cleanup timer
        if self.config.auto_cleanup:
            self._reset_cleanup_timer()
    
    def is_timeout(self) -> bool:
        """Check if session has timed out"""
        if not self.is_active:
            return True
        
        timeout_seconds = self.config.timeout_minutes * 60
        if timeout_seconds <= 0:
            return False
        current_time = time.time()
        
        # Check if session has exceeded timeout
        if current_time - self.last_accessed > timeout_seconds:
            return True
        
        # Check if session has exceeded maximum lifetime (optional additional check)
        max_lifetime_hours = 24  # Maximum 24 hours
        if (datetime.now() - self.created_at).total_seconds() > max_lifetime_hours * 3600:
            return True
        
        return False
    
    def get_remaining_time(self) -> float:
        """Get remaining time (minutes)"""
        if not self.is_active:
            return 0.0
        
        timeout_seconds = self.config.timeout_minutes * 60
        if timeout_seconds <= 0:
            return float("inf")
        current_time = time.time()
        elapsed = current_time - self.last_accessed
        remaining_seconds = max(0, timeout_seconds - elapsed)
        
        return remaining_seconds / 60.0  # Convert to minutes
    
    def extend_timeout(self, additional_minutes: int = 5):
        """Extend session timeout"""
        with self.lock:
            if not self.is_active:
                return
            
            self.config.timeout_minutes += additional_minutes
            self.touch()  # Update last accessed time and reset timer
    
    def touch(self):
        """Update last access time and reset cleanup timer"""
        with self.lock:
            if not self.is_active:
                return
            
            self.last_accessed = time.time()
            self.last_used = datetime.now()
            
            # Reset cleanup timer
            if self.config.auto_cleanup:
                self._reset_cleanup_timer()
    
    def deactivate(self):
        """Deactivate session manager"""
        with self.lock:
            if not self.is_active:
                return
            
            self.is_active = False
            
            # Cancel timer
            if self.cleanup_timer:
                self.cleanup_timer.cancel()
                self.cleanup_timer = None
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get session information"""
        return {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'last_used': self.last_used.isoformat(),
            'is_active': self.is_active,
            'timeout_minutes': self.config.timeout_minutes,
            'remaining_time': self._get_time_remaining_str(),
            'is_timeout': self.is_timeout()
        }
    
    def _get_time_remaining_str(self) -> str:
        """Get string representation of remaining time"""
        remaining_minutes = self.get_remaining_time()
        
        if remaining_minutes <= 0:
            return "Expired"
        elif remaining_minutes < 1:
            return f"{int(remaining_minutes * 60)}s"
        else:
            return f"{remaining_minutes:.1f}min"
    
    def _reset_cleanup_timer(self):
        """Reset auto cleanup timer"""
        with self.lock:
            timeout_seconds = self.config.timeout_minutes * 60
            if timeout_seconds <= 0:
                # 0 or negative timeout means caller manages lifecycle explicitly.
                return
            # Cancel existing timer
            if self.cleanup_timer:
                self.cleanup_timer.cancel()
            
            # Create new timer
            self.cleanup_timer = threading.Timer(
                timeout_seconds,
                self._auto_cleanup_callback
            )
            # Prevent session timers from blocking interpreter shutdown.
            self.cleanup_timer.daemon = True
            self.cleanup_timer.start()
    
    def _auto_cleanup_callback(self):
        """Auto cleanup callback"""
        with self.lock:
            if self.is_timeout():
                # breakpoint()
                self.is_active = False
                # Call cleanup callback
                if self.cleanup_callback:
                    try:
                        self.cleanup_callback(self.session_id)
                    except Exception as e:
                        logger.error(f"Cleanup callback failed: {e}")


class GlobalSessionManager:
    """Global session manager"""
    _instance = None
    
    def __init__(self):
        self.sessions: Dict[str, SessionManager] = {}
    
    def register_session(self, session_manager: SessionManager):
        """Register session"""
        self.sessions[session_manager.session_id] = session_manager
    
    def unregister_session(self, session_id: str):
        """Unregister session"""
        self.sessions.pop(session_id, None)
    
    def cleanup_all_sessions(self):
        """Cleanup all sessions"""
        for session_manager in list(self.sessions.values()):
            session_manager.deactivate()
        self.sessions.clear()
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
