"""
State importer for core data handler.

This module provides functionality to import agent states, conversation histories,
and session data from JSON format for restoration.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .models import (
    AgentState,
    ConversationState,
    ExportedAgentState,
    ExportedConversationState,
    ExportedSessionState,
    SessionState,
    STATE_V1_VERSION,
)

logger = logging.getLogger(__name__)


class StateImporter:
    """
    Imports agent states, conversations, and sessions from JSON format.
    
    This class provides methods to deserialize and restore various state objects
    from persistent storage, similar to the simulation data handler but focused
    on core agent and conversation states.
    """

    SUPPORTED_VERSIONS = [STATE_V1_VERSION]

    @staticmethod
    def convert_exported_message(exported_message: Dict[str, Any]) -> Dict[str, Any]:
        """Convert exported message format to internal format."""
        if not isinstance(exported_message, dict):
            raise ValueError(f"Exported message must be a dictionary, got {type(exported_message)}")
        
        return {
            "role": exported_message.get("role", "unknown"),
            "content": exported_message.get("content", ""),
            "timestamp": exported_message.get("timestamp"),
            "metadata": exported_message.get("metadata")
        }

    @staticmethod
    def convert_exported_tool_call(exported_tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Convert exported tool call format to internal format."""
        if not isinstance(exported_tool_call, dict):
            raise ValueError(f"Exported tool call must be a dictionary, got {type(exported_tool_call)}")
        
        return {
            "tool_name": exported_tool_call.get("tool_name", ""),
            "function_name": exported_tool_call.get("function_name", ""),
            "arguments": exported_tool_call.get("arguments", {}),
            "result": exported_tool_call.get("result"),
            "timestamp": exported_tool_call.get("timestamp"),
            "success": exported_tool_call.get("success", True),
            "error_message": exported_tool_call.get("error_message")
        }

    @staticmethod
    def convert_exported_memory_block(exported_memory_block: Dict[str, Any]) -> Dict[str, Any]:
        """Convert exported memory block format to internal format."""
        if not isinstance(exported_memory_block, dict):
            raise ValueError(f"Exported memory block must be a dictionary, got {type(exported_memory_block)}")
        
        return {
            "block_type": exported_memory_block.get("block_type", "unknown"),
            "block_id": exported_memory_block.get("block_id", ""),
            "content": exported_memory_block.get("content", {}),
            "metadata": exported_memory_block.get("metadata")
        }

    def import_agent_state_from_json(
        self,
        json_data: Union[str, bytes, Dict[str, Any]]
    ) -> Tuple[AgentState, Dict[str, Any]]:
        """
        Import agent state from JSON data.
        
        Args:
            json_data: JSON string, bytes, or dictionary containing the agent state
            
        Returns:
            Tuple of (AgentState, metadata_dict)
            
        Raises:
            ValueError: If the data format is unsupported or invalid
        """
        if isinstance(json_data, (str, bytes)):
            data = json.loads(json_data)
        else:
            data = json_data

        # Validate version if present
        metadata = data.get("metadata", {})
        version = metadata.get("version", STATE_V1_VERSION)
        if version not in self.SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported version: {version}")

        # Convert exported format to internal format
        agent_state = AgentState(
            agent_id=data["agent_id"],
            agent_type=data["agent_type"],
            configuration=data.get("configuration", {}),
            memory_blocks=[
                self.convert_exported_memory_block(block)
                for block in data.get("memory_blocks", [])
            ],
            tool_calls_history=[
                self.convert_exported_tool_call(call)
                for call in data.get("tool_calls_history", [])
            ],
            current_context=data.get("current_context")
        )

        return agent_state, metadata

    def import_conversation_state_from_json(
        self,
        json_data: Union[str, bytes, Dict[str, Any]]
    ) -> Tuple[ConversationState, Dict[str, Any]]:
        """
        Import conversation state from JSON data.
        
        Args:
            json_data: JSON string, bytes, or dictionary containing the conversation state
            
        Returns:
            Tuple of (ConversationState, metadata_dict)
            
        Raises:
            ValueError: If the data format is unsupported or invalid
        """
        if isinstance(json_data, (str, bytes)):
            data = json.loads(json_data)
        else:
            data = json_data

        # Validate version if present
        metadata = data.get("metadata", {})
        version = metadata.get("version", STATE_V1_VERSION)
        if version not in self.SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported version: {version}")

        # Convert exported format to internal format
        conversation_state = ConversationState(
            conversation_id=data["conversation_id"],
            messages=[
                self.convert_exported_message(msg)
                for msg in data.get("messages", [])
            ],
            participants=data.get("participants", []),
            context=data.get("context")
        )

        return conversation_state, metadata

    def import_session_state_from_json(
        self,
        json_data: Union[str, bytes, Dict[str, Any]]
    ) -> Tuple[SessionState, Dict[str, Any]]:
        """
        Import session state from JSON data.
        
        Args:
            json_data: JSON string, bytes, or dictionary containing the session state
            
        Returns:
            Tuple of (SessionState, metadata_dict)
            
        Raises:
            ValueError: If the data format is unsupported or invalid
        """
        if isinstance(json_data, (str, bytes)):
            data = json.loads(json_data)
        else:
            data = json_data

        # Validate version if present
        metadata = data.get("metadata", {})
        version = metadata.get("version", STATE_V1_VERSION)
        if version not in self.SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported version: {version}")

        # Convert agents
        agents = []
        for agent_data in data.get("agents", []):
            agent_state, _ = self.import_agent_state_from_json(agent_data)
            agents.append(agent_state)

        # Convert conversations
        conversations = []
        for conv_data in data.get("conversations", []):
            conv_state, _ = self.import_conversation_state_from_json(conv_data)
            conversations.append(conv_state)

        # Create session state
        session_state = SessionState(
            session_id=data["session_id"],
            agents=agents,
            conversations=conversations,
            global_context=data.get("global_context")
        )

        return session_state, metadata

    def import_from_file(
        self,
        file_path: str
    ) -> Tuple[Union[AgentState, ConversationState, SessionState], Dict[str, Any]]:
        """
        Import state from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Tuple of (state_object, metadata_dict)
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid or unsupported
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"State file not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Determine the type of state based on the data structure
            if "agent_id" in data and "agent_type" in data:
                state, metadata = self.import_agent_state_from_json(data)
                logger.info(f"Agent state imported from {file_path}")
            elif "conversation_id" in data:
                state, metadata = self.import_conversation_state_from_json(data)
                logger.info(f"Conversation state imported from {file_path}")
            elif "session_id" in data:
                state, metadata = self.import_session_state_from_json(data)
                logger.info(f"Session state imported from {file_path}")
            else:
                raise ValueError(f"Unknown state format in file: {file_path}")

            return state, metadata

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Failed to import state from {file_path}: {e}")
            raise

    def list_backups(
        self,
        backup_dir: str = "backups",
        state_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available backup files.
        
        Args:
            backup_dir: Directory containing backup files
            state_type: Optional filter by state type ("agent", "conversation", "session")
            
        Returns:
            List of backup file information dictionaries
        """
        backup_path = Path(backup_dir)
        if not backup_path.exists():
            return []

        backups = []
        for file_path in backup_path.glob("*.json"):
            try:
                # Extract information from filename
                filename = file_path.stem
                parts = filename.split("_")
                
                if len(parts) >= 3:
                    backup_type = parts[2] if len(parts) > 2 else "unknown"
                    
                    # Filter by state type if specified
                    if state_type and backup_type != state_type:
                        continue
                    
                    backup_info = {
                        "file_path": str(file_path),
                        "filename": file_path.name,
                        "type": backup_type,
                        "created_at": file_path.stat().st_mtime,
                        "size": file_path.stat().st_size
                    }
                    
                    # Try to extract ID from filename
                    if len(parts) >= 4:
                        backup_info["id"] = parts[3]
                    
                    backups.append(backup_info)
                    
            except Exception as e:
                logger.warning(f"Could not process backup file {file_path}: {e}")
                continue

        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x["created_at"], reverse=True)
        return backups

    def restore_from_backup(
        self,
        backup_path: str
    ) -> Tuple[Union[AgentState, ConversationState, SessionState], Dict[str, Any]]:
        """
        Restore state from a backup file.
        
        Args:
            backup_path: Path to the backup file
            
        Returns:
            Tuple of (restored_state, metadata_dict)
        """
        return self.import_from_file(backup_path)

    def validate_state_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate a state file without fully importing it.
        
        Args:
            file_path: Path to the state file
            
        Returns:
            Dictionary with validation results
        """
        result = {
            "valid": False,
            "error": None,
            "version": None,
            "type": None,
            "metadata": {}
        }

        try:
            if not Path(file_path).exists():
                result["error"] = "File not found"
                return result

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check version
            metadata = data.get("metadata", {})
            version = metadata.get("version", STATE_V1_VERSION)
            result["version"] = version
            result["metadata"] = metadata

            if version not in self.SUPPORTED_VERSIONS:
                result["error"] = f"Unsupported version: {version}"
                return result

            # Determine type
            if "agent_id" in data and "agent_type" in data:
                result["type"] = "agent"
            elif "conversation_id" in data:
                result["type"] = "conversation"
            elif "session_id" in data:
                result["type"] = "session"
            else:
                result["error"] = "Unknown state format"
                return result

            result["valid"] = True

        except json.JSONDecodeError as e:
            result["error"] = f"Invalid JSON: {e}"
        except Exception as e:
            result["error"] = f"Validation error: {e}"

        return result