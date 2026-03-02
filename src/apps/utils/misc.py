# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import itertools
import json
import os
import random
from functools import wraps
from typing import Iterator, TypeVar

T = TypeVar("T")


def add_reset(cls):
    """
    Decorator that adds a reset method to a class.

    The reset method reinitializes the instance with the same arguments
    that were used to create it initially.

    :param cls: The class to decorate
    :return: The decorated class with a reset method
    """
    original_init = cls.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        self._initial_args = args
        self._initial_kwargs = kwargs
        original_init(self, *args, **kwargs)

    def reset(self):
        self.__init__(*self._initial_args, **self._initial_kwargs)

    cls.__init__ = new_init
    cls.reset = reset
    return cls


def get_function_name(func):
    """
    Get the name of a function, handling partial functions.

    :param func: The function to get the name of
    :return: The function name
    :raises ValueError: If the function name cannot be determined
    """
    if hasattr(func, "__name__"):
        return func.__name__
    # partial function
    elif hasattr(func, "func"):
        return func.func.__name__
    else:
        raise ValueError("Could not find function name")


def helper_delay_range(min_delay, max_delay, unit="sec"):
    """
    Generate a random delay within the specified range.

    :param min_delay: Minimum delay value
    :param max_delay: Maximum delay value
    :param unit: Time unit ('sec', 'min', 'hr', 'day')
    :return: Random delay in seconds
    :raises AssertionError: If unit is not one of the supported units
    """
    assert unit in ["sec", "min", "hr", "day"]
    d_factor = {
        "sec": 1,
        "min": 60,
        "hr": 3600,
        "day": 86400,
    }
    return random.randint(min_delay, max_delay) * d_factor[unit]


def save_jsonl(data: list[dict], filename: str, overwrite: bool = False) -> None:
    """
    Save a list of dictionaries to a JSONL file.

    :param data: List of dictionaries to save
    :type data: list[dict]
    :param filename: Path to the output file
    :type filename: str
    :param overwrite: Whether to overwrite an existing file
    :type overwrite: bool
    :raises FileExistsError: If the file exists and overwrite is False
    """
    if os.path.exists(filename) and not overwrite:
        raise FileExistsError(
            f"File '{filename}' already exists. Set overwrite=True to overwrite."
        )

    with open(filename, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")


def truncate_string(s, max_length=100):
    """
    Truncate a string to a maximum length, adding ellipsis if truncated.

    :param s: The string to truncate
    :param max_length: Maximum length of the string
    :return: The truncated string
    """
    if len(str(s)) > max_length:
        return str(s)[:max_length] + "..."
    else:
        return s


def batched(iterable: Iterator[T], n: int) -> Iterator[tuple[T, ...]]:
    """
    Batch data into Tuples of size n.

    In [21]: list(batched('ABCDEFG', 3))
    Out[21]: [('A', 'B', 'C'), ('D', 'E', 'F'), ('G',)]

    similar to itertools.batched, for versions of python < 3.12
    https://docs.python.org/3/library/itertools.html#itertools.batched

    :param iterable: The iterable to batch
    :type iterable: Iterator[T]
    :param n: Batch size
    :type n: int
    :return: Iterator of batches
    :rtype: Iterator[tuple[T, ...]]
    :raises ValueError: If n is less than 1
    """
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


def strip_app_name_prefix(tool_name: str, app_name: str) -> str:
    """
    Strip the app name prefix from a tool name if it matches.

    :param tool_name: The tool name to strip the prefix from
    :type tool_name: str
    :param app_name: The app name to strip
    :type app_name: str
    :return: The tool name without the app name prefix
    :rtype: str
    """
    parts = tool_name.split("__", maxsplit=1)
    # Only remove the prefix if it matches the current app name
    if len(parts) > 1 and parts[0] == app_name:
        return parts[1]
    return tool_name


def uuid_hex(rng: random.Random) -> str:
    """
    Generate a deterministic UUID hex string using the provided random number generator.

    :param rng: Random number generator instance to use directly
    :return: UUID hex string (32 characters)
    """
    return "".join(rng.choice("0123456789abcdef") for _ in range(32))
