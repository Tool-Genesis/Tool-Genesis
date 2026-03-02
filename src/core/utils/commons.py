import functools
import importlib
from loguru import logger
import os
import platform
import re
import socket
import subprocess
import threading
import time
import zipfile
from functools import wraps
from http import HTTPStatus
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    cast,
)
from urllib.parse import urlparse

import pydantic
try:
    import requests
except Exception:
    requests = None
from pydantic import BaseModel

def get_pydantic_object_schema(pydantic_params: Type[BaseModel]) -> Dict:
    r"""Get the JSON schema of a Pydantic model.

    Args:
        pydantic_params (Type[BaseModel]): The Pydantic model class to retrieve
            the schema for.

    Returns:
        dict: The JSON schema of the Pydantic model.
    """
    return pydantic_params.model_json_schema()


def to_pascal(snake: str) -> str:
    """Convert a snake_case string to PascalCase.

    Args:
        snake (str): The snake_case string to be converted.

    Returns:
        str: The converted PascalCase string.
    """
    # Check if the string is already in PascalCase
    if re.match(r"^[A-Z][a-zA-Z0-9]*([A-Z][a-zA-Z0-9]*)*$", snake):
        return snake
    # Remove leading and trailing underscores
    snake = snake.strip("_")
    # Replace multiple underscores with a single one
    snake = re.sub("_+", "_", snake)
    # Convert to PascalCase
    return re.sub(
        "_([0-9A-Za-z])",
        lambda m: m.group(1).upper(),
        snake.title(),
    )
