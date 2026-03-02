# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import copy
import json
import re
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from typing import Any


def serialize_field(value: Any) -> Any:
    """
    Serialize a field value to a JSON-compatible format.

    This function handles dataclasses, Enums, lists, and dictionaries by converting
    them to their serializable equivalents.

    :param value: The value to serialize
    :type value: Any
    :return: The serialized value
    :rtype: Any
    """
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    elif isinstance(value, Enum):
        return value.value
    elif isinstance(value, list):
        return [serialize_field(item) for item in value]
    elif isinstance(value, dict):
        return {k: serialize_field(v) for k, v in value.items()}
    else:
        return value


def make_serializable(value: Any) -> Any:
    """
    Convert data to a format compatible with any method of serialization.

    This function handles various data types including dataclasses, Enums, lists,
    dictionaries, and primitive types. It ensures that the data is converted into
    a format that can be serialized into JSON or Pickle. For unsupported types, it removes
    non-deterministic memory addresses from the string representation.

    :param value: The data to be serialized.
    :type value: Any
    :return: The data in a serialization-compatible format.
    :rtype: Any
    """
    if is_dataclass(value):
        return {
            f.name: make_serializable(getattr(value, f.name)) for f in fields(value)
        }
    elif isinstance(value, Enum):
        return value.value
    elif isinstance(value, list):
        return [make_serializable(item) for item in value]
    elif isinstance(value, dict):
        return {str(k): make_serializable(v) for k, v in value.items()}
    elif type(value) in (str, int, float, bool) or value is None:
        return value
    else:
        # A hack that removes non-deterministic memory addresses (e.g. 0x7f7b3a5b4b70) from the string.
        # An address can change between simulations and therefore influence testing (expected vs actual comparison).
        stringified: str = re.sub(r" at 0x[a-f0-9]{8,16}", "", str(value))
        return stringified


class EnumEncoder(json.JSONEncoder):
    """
    JSON encoder that handles Enum values by using their value attribute.

    This encoder extends the standard JSON encoder to properly serialize Enum instances.

    :example:
        json.dumps(my_data, cls=EnumEncoder)
    """

    def default(self, o):
        if isinstance(o, Enum):
            return o.value
        return json.JSONEncoder.default(self, o)


class SkippableDeepCopy:
    """
    Base class to be able to skip some fields during deepcopy or pickle
    if they are non serializable.

    Subclasses can define _skip_deepcopy_fields and _skip_pickle_fields lists
    to specify which fields should be skipped during deepcopy and pickle operations.
    """

    _skip_deepcopy_fields = []
    _skip_pickle_fields = []

    def __deepcopy__(self, memo):
        # Create a new instance of the class
        cls = self.__class__
        new_instance = cls.__new__(cls)
        memo[id(self)] = new_instance

        for k, v in self.__dict__.items():
            if k not in self._skip_deepcopy_fields:
                setattr(new_instance, k, copy.deepcopy(v, memo))
            else:
                setattr(new_instance, k, None)  # Or some other default value

        return new_instance

    def __getstate__(self):
        # Create a copy of the instance's state and remove the fields to be skipped
        state = self.__dict__.copy()
        for field in self._skip_pickle_fields:
            if field in state:
                del state[field]
        return state

    def __setstate__(self, state):
        # Restore the instance's state
        self.__dict__.update(state)
        # Optionally, set the skipped fields to a default value
        for field in self._skip_pickle_fields:
            setattr(self, field, None)
