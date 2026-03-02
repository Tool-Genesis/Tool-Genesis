"""
State exporter for core data handler.

This module provides functionality to export agent states, conversation histories,
and session data to JSON format for persistence.
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .models import (
    AgentState,
    ConversationState,
    ExportedAgentState,
    ExportedConversationState,
    ExportedMemoryBlock,
    ExportedMessage,
    ExportedSessionState,
    ExportedToolCall,
    SessionState,
    StateMetadata,
    STATE_V1_VERSION,
)

logger = logging.getLogger(__name__)


class StateExporter:
    """
    Exports agent states, conversations, and sessions to JSON format.
    
    This class provides methods to serialize and save various state objects
    to persistent storage, similar to the simulation data handler but focused
    on core agent and conversation states.
    """

    @staticmethod
    def convert_message(message: Dict[str, Any]) -> ExportedMessage:
        """Convert internal message format to exported format."""
        if not isinstance(message, dict):
            raise ValueError(f"Message must be a dictionary, got {type(message)}")
        
        return ExportedMessage(
            role=message.get("role", "unknown"),
            content=message.get("content", ""),
            timestamp=message.get("timestamp"),
            metadata=message.get("metadata")
        )

    @staticmethod
    def convert_tool_call(tool_call: Dict[str, Any]) -> ExportedToolCall:
        """Convert internal tool call format to exported format."""
        if not isinstance(tool_call, dict):
            raise ValueError(f"Tool call must be a dictionary, got {type(tool_call)}")
        
        return ExportedToolCall(
            tool_name=tool_call.get("tool_name", ""),
            function_name=tool_call.get("function_name", ""),
            arguments=tool_call.get("arguments", {}),
            result=tool_call.get("result"),
            timestamp=tool_call.get("timestamp"),
            success=tool_call.get("success", True),
            error_message=tool_call.get("error_message")
        )

    @staticmethod
    def convert_memory_block(memory_block: Dict[str, Any]) -> ExportedMemoryBlock:
        """Convert internal memory block format to exported format."""
        if not isinstance(memory_block, dict):
            raise ValueError(f"Memory block must be a dictionary, got {type(memory_block)}")
        
        return ExportedMemoryBlock(
            block_type=memory_block.get("block_type", "unknown"),
            block_id=memory_block.get("block_id", ""),
            content=memory_block.get("content", {}),
            metadata=memory_block.get("metadata")
        )

    @staticmethod
    def convert_agent_state(
        agent_state: AgentState,
        metadata: Optional[StateMetadata] = None
    ) -> ExportedAgentState:
        """Convert internal agent state to exported format."""
        if not isinstance(agent_state, AgentState):
            raise ValueError(f"Expected AgentState, got {type(agent_state)}")
        
        if metadata is None:
            metadata = StateMetadata(agent_id=agent_state.agent_id)

        return ExportedAgentState(
            agent_id=agent_state.agent_id,
            agent_type=agent_state.agent_type,
            configuration=agent_state.configuration,
            memory_blocks=[
                StateExporter.convert_memory_block(block)
                for block in agent_state.memory_blocks
            ],
            tool_calls_history=[
                StateExporter.convert_tool_call(call)
                for call in agent_state.tool_calls_history
            ],
            current_context=agent_state.current_context,
            metadata=metadata
        )

    @staticmethod
    def convert_conversation_state(
        conversation_state: ConversationState,
        metadata: Optional[StateMetadata] = None
    ) -> ExportedConversationState:
        """Convert internal conversation state to exported format."""
        if not isinstance(conversation_state, ConversationState):
            raise ValueError(f"Expected ConversationState, got {type(conversation_state)}")
        
        if metadata is None:
            metadata = StateMetadata()

        return ExportedConversationState(
            conversation_id=conversation_state.conversation_id,
            messages=[
                StateExporter.convert_message(msg)
                for msg in conversation_state.messages
            ],
            participants=conversation_state.participants,
            context=conversation_state.context,
            metadata=metadata
        )

    def export_agent_state_to_json(
        self,
        agent_state: AgentState,
        output_path: Optional[str] = None,
        metadata: Optional[StateMetadata] = None,
        indent: Optional[int] = 2
    ) -> str:
        """
        Export agent state to JSON format.
        
        Args:
            agent_state: The agent state to export
            output_path: Optional file path to save the JSON
            metadata: Optional metadata to include
            indent: JSON indentation level
            
        Returns:
            JSON string representation of the agent state
        """
        exported_state = self.convert_agent_state(agent_state, metadata)
        json_str = exported_state.model_dump_json(indent=indent)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            logger.info(f"Agent state exported to {output_path}")
        
        return json_str

    def export_conversation_state_to_json(
        self,
        conversation_state: ConversationState,
        output_path: Optional[str] = None,
        metadata: Optional[StateMetadata] = None,
        indent: Optional[int] = 2
    ) -> str:
        """
        Export conversation state to JSON format.
        
        Args:
            conversation_state: The conversation state to export
            output_path: Optional file path to save the JSON
            metadata: Optional metadata to include
            indent: JSON indentation level
            
        Returns:
            JSON string representation of the conversation state
        """
        exported_state = self.convert_conversation_state(conversation_state, metadata)
        json_str = exported_state.model_dump_json(indent=indent)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            logger.info(f"Conversation state exported to {output_path}")
        
        return json_str

    def export_session_state_to_json(
        self,
        session_state: SessionState,
        output_path: Optional[str] = None,
        metadata: Optional[StateMetadata] = None,
        indent: Optional[int] = 2
    ) -> str:
        """
        Export complete session state to JSON format.
        
        Args:
            session_state: The session state to export
            output_path: Optional file path to save the JSON
            metadata: Optional metadata to include
            indent: JSON indentation level
            
        Returns:
            JSON string representation of the session state
        """
        if metadata is None:
            metadata = StateMetadata(session_id=session_state.session_id)

        exported_session = ExportedSessionState(
            session_id=session_state.session_id,
            agents=[
                self.convert_agent_state(agent, StateMetadata(agent_id=agent.agent_id))
                for agent in session_state.agents
            ],
            conversations=[
                self.convert_conversation_state(conv, StateMetadata())
                for conv in session_state.conversations
            ],
            global_context=session_state.global_context,
            metadata=metadata
        )
        
        json_str = exported_session.model_dump_json(indent=indent)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            logger.info(f"Session state exported to {output_path}")
        
        return json_str

    def export_to_file(
        self,
        state: Union[AgentState, ConversationState, SessionState],
        output_path: str,
        metadata: Optional[StateMetadata] = None,
        indent: Optional[int] = 2
    ) -> bool:
        """
        Export any state object to a file.
        
        Args:
            state: The state object to export
            output_path: File path to save the JSON
            metadata: Optional metadata to include
            indent: JSON indentation level
            
        Returns:
            True if export was successful, False otherwise
        """
        try:
            if isinstance(state, AgentState):
                self.export_agent_state_to_json(state, output_path, metadata, indent)
            elif isinstance(state, ConversationState):
                self.export_conversation_state_to_json(state, output_path, metadata, indent)
            elif isinstance(state, SessionState):
                self.export_session_state_to_json(state, output_path, metadata, indent)
            else:
                logger.error(f"Unsupported state type: {type(state)}")
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to export state to {output_path}: {e}")
            return False

    def create_backup(
        self,
        state: Union[AgentState, ConversationState, SessionState],
        backup_dir: str = "backups",
        prefix: str = "state_backup"
    ) -> Optional[str]:
        """
        Create a timestamped backup of the state.
        
        Args:
            state: The state object to backup
            backup_dir: Directory to store backups
            prefix: Filename prefix for the backup
            
        Returns:
            Path to the created backup file, or None if failed
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if isinstance(state, AgentState):
                filename = f"{prefix}_agent_{state.agent_id}_{timestamp}.json"
            elif isinstance(state, ConversationState):
                filename = f"{prefix}_conversation_{state.conversation_id}_{timestamp}.json"
            elif isinstance(state, SessionState):
                filename = f"{prefix}_session_{state.session_id}_{timestamp}.json"
            else:
                filename = f"{prefix}_{timestamp}.json"
            
            backup_path = os.path.join(backup_dir, filename)
            
            if self.export_to_file(state, backup_path):
                return backup_path
            return None
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None