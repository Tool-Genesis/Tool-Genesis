"""
Model adapter for unified tool calling interfaces across different models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum

from .function_tool import FunctionTool


class ModelType(Enum):
    """Supported model types for tool calling."""
    OPENAI = "openai"
    TONGYI_DEEPRESEARCH = "tongyi_deepresearch"
    # Add more model types as needed


class BaseModelAdapter(ABC):
    """Base class for model-specific tool adapters."""
    
    @abstractmethod
    def adapt_tool_schema(self, func_tool: FunctionTool) -> FunctionTool:
        """Adapt a FunctionTool to the model-specific format."""
        pass
    
    @abstractmethod
    def adapt_function_call(self, func: Callable, args: Dict[str, Any]) -> Any:
        """Adapt function call arguments from model format to function format."""
        pass


class OpenAIAdapter(BaseModelAdapter):
    """Adapter for OpenAI-compatible models (default format)."""
    
    def adapt_tool_schema(self, func_tool: FunctionTool) -> FunctionTool:
        """OpenAI format is the default, no adaptation needed."""
        return func_tool
    
    def adapt_function_call(self, func: Callable, args: Dict[str, Any]) -> Any:
        """Direct call with original arguments."""
        return func(**args)


class TongyiDeepResearchAdapter(BaseModelAdapter):
    """Adapter for Tongyi DeepResearch model."""
    
    # Mapping from original function names to DeepResearch names
    FUNCTION_NAME_MAPPING = {
        "browser_search": "search",
        "browser_open": "Visit"
    }
    
    def adapt_tool_schema(self, func_tool: FunctionTool) -> FunctionTool:
        """Adapt tool schema for Tongyi DeepResearch format."""
        original_schema = func_tool.get_openai_tool_schema()
        func_name = original_schema["function"]["name"]
        
        # Create adapted schema based on function name
        if func_name == "browser_search":
            adapted_schema = {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web for information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of search queries."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        elif func_name == "browser_open":
            adapted_schema = {
                "type": "function",
                "function": {
                    "name": "Visit",
                    "description": "Open a URL to visit a webpage.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of URLs to visit."
                            },
                            "goal": {
                                "type": "string",
                                "description": "Purpose of visiting the URL."
                            }
                        },
                        "required": ["url"]
                    }
                }
            }
        else:
            # For unknown functions, use original schema
            adapted_schema = original_schema
        
        # Create new FunctionTool with adapted schema
        return FunctionTool(
            func=func_tool.func,
            openai_tool_schema=adapted_schema
        )
    
    def adapt_function_call(self, func: Callable, args: Dict[str, Any]) -> Any:
        """Adapt function call arguments from DeepResearch format to original format."""
        func_name = func.__name__
        
        if func_name == "browser_search":
            # Convert array of queries to single query
            query_list = args.get("query", [])
            if isinstance(query_list, list) and query_list:
                # Use the first query or join multiple queries
                query = query_list[0] if len(query_list) == 1 else " ".join(query_list)
            else:
                query = str(query_list) if query_list else ""
            
            # Extract topn if provided, otherwise use default
            topn = args.get("topn", 10)
            return func(query=query, topn=topn)
            
        elif func_name == "browser_open":
            # Convert array of URLs to single URL
            url_list = args.get("url", [])
            if isinstance(url_list, list) and url_list:
                url = url_list[0]  # Use the first URL
            else:
                url = str(url_list) if url_list else ""
            
            # goal parameter is ignored as original function doesn't use it
            return func(url=url)
        
        else:
            # For unknown functions, use original arguments
            return func(**args)


class ModelAdapterFactory:
    """Factory for creating model-specific adapters."""
    
    _adapters = {
        ModelType.OPENAI: OpenAIAdapter,
        ModelType.TONGYI_DEEPRESEARCH: TongyiDeepResearchAdapter,
    }
    
    @classmethod
    def create_adapter(cls, model_type: Union[ModelType, str]) -> BaseModelAdapter:
        """Create an adapter for the specified model type."""
        if isinstance(model_type, str):
            try:
                model_type = ModelType(model_type.lower())
            except ValueError:
                raise ValueError(f"Unsupported model type: {model_type}")
        
        adapter_class = cls._adapters.get(model_type)
        if not adapter_class:
            raise ValueError(f"No adapter available for model type: {model_type}")
        
        return adapter_class()
    
    @classmethod
    def register_adapter(cls, model_type: ModelType, adapter_class: type):
        """Register a new adapter for a model type."""
        cls._adapters[model_type] = adapter_class
    
    @classmethod
    def get_supported_models(cls) -> List[ModelType]:
        """Get list of supported model types."""
        return list(cls._adapters.keys())


class AdaptiveToolWrapper:
    """Wrapper that adapts tools for different models."""
    
    def __init__(self, model_type: Union[ModelType, str] = ModelType.OPENAI):
        """Initialize with a specific model type."""
        self.adapter = ModelAdapterFactory.create_adapter(model_type)
        self.model_type = model_type
    
    def adapt_tools(self, tools: List[FunctionTool]) -> List[FunctionTool]:
        """Adapt a list of tools for the current model type."""
        return [self.adapter.adapt_tool_schema(tool) for tool in tools]
    
    def adapt_function_call(self, func: Callable, args: Dict[str, Any]) -> Any:
        """Adapt and execute a function call."""
        return self.adapter.adapt_function_call(func, args)