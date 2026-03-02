"""
Data models for core state management.

This module defines the data structures used for saving and loading
agent states, conversation histories, and related metadata.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# Version constant for state format
STATE_V1_VERSION: str = "core_state_v1"


class StateMetadata(BaseModel):
    """Metadata for exported states."""
    version: str = STATE_V1_VERSION
    created_at: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class ExportedMessage(BaseModel):
    """Exported message structure."""
    role: str
    content: str
    timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class ExportedToolCall(BaseModel):
    """Exported tool call structure."""
    tool_name: str
    function_name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    timestamp: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None


class ExportedMemoryBlock(BaseModel):
    """Exported memory block structure."""
    block_type: str
    block_id: str
    content: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class ExportedAgentState(BaseModel):
    """Exported agent state structure."""
    agent_id: str
    agent_type: str
    configuration: Dict[str, Any]
    memory_blocks: List[ExportedMemoryBlock] = Field(default_factory=list)
    tool_calls_history: List[ExportedToolCall] = Field(default_factory=list)
    current_context: Optional[Dict[str, Any]] = None
    metadata: StateMetadata


class ExportedConversationState(BaseModel):
    """Exported conversation state structure."""
    conversation_id: str
    messages: List[ExportedMessage] = Field(default_factory=list)
    participants: List[str] = Field(default_factory=list)
    context: Optional[Dict[str, Any]] = None
    metadata: StateMetadata


class ExportedSessionState(BaseModel):
    """Exported session state structure containing multiple agents and conversations."""
    session_id: str
    agents: List[ExportedAgentState] = Field(default_factory=list)
    conversations: List[ExportedConversationState] = Field(default_factory=list)
    global_context: Optional[Dict[str, Any]] = None
    metadata: StateMetadata


# Internal state models (for runtime use)
class AgentState(BaseModel):
    """Internal agent state model."""
    agent_id: str
    agent_type: str
    configuration: Dict[str, Any]
    memory_blocks: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls_history: List[Dict[str, Any]] = Field(default_factory=list)
    current_context: Optional[Dict[str, Any]] = None


class ConversationState(BaseModel):
    """Internal conversation state model."""
    conversation_id: str
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    participants: List[str] = Field(default_factory=list)
    context: Optional[Dict[str, Any]] = None


class SessionState(BaseModel):
    """Internal session state model."""
    session_id: str
    agents: List[AgentState] = Field(default_factory=list)
    conversations: List[ConversationState] = Field(default_factory=list)
    global_context: Optional[Dict[str, Any]] = None