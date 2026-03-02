'''
Generate **two parts** in the Python file at `state_dataclass_{server_name}.py`:

**CRITICAL: State Management Rules**
- Basic attributes (from init_state array): Direct class attributes (self.gender, self.solar_datetime)
- Complex data (from dataclass_spec array): Also become direct class attributes but initialized via dataclass
- DO NOT use `self._state` dictionary for storing any state
- Update attributes directly: `self.gender = 1`, `self.bazi_data = {{"new": "data"}}`
- Access attributes directly: `self.gender`, `self.bazi_data`
- Use `field(default_factory=dict)` for empty mutable defaults, `field(default_factory=lambda: {{"key": "value"}})` for non-empty
- Separate basic attributes and complex data structures clearly
        
```python
# env_state_code.py

# NOTE: All third-party libraries referenced by this server MUST be imported at module load time
# (top-level) with version checks, so that missing/incorrect dependencies fail fast and deterministically.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

# Import the correct base class
from src.apps.app import App


# Generate one @dataclass for EACH entry in SPEC["dataclass_spec"].
# Fill class names, docstrings, and fields from SPEC. This section is a placeholder.

# === Dataclass: <ClassName_1> ===
@dataclass
class <ClassName_1>:
    """
    <Class_1_Description>
    """
    # Expand fields from SPEC["dataclass_spec"][i]["fields"].
    # Use py_type_hint when provided; otherwise infer from json_schema.
    # Examples (replace with actual fields):
    # coordinates: Tuple[float, float] = (0.0, 0.0)
    # timezone_info: Dict[str, str] = field(default_factory=dict)
    # solar_times: Dict[str, datetime] = field(default_factory=dict)
    ...
    
# === Dataclass: <ClassName_2> ===
# @dataclass
# class <ClassName_2>:
#     """
#     <Class_2_Description>
#     """
#     ...
# (Repeat as needed for all dataclasses in SPEC.)

# Optionally, if your dataclasses need their own (de)serialization helpers,
# you can add stubs like:
#   def get_state(self) -> Dict[str, Any]: ...
#   def load_state(self, data: Dict[str, Any]) -> None: ...


class <ServerClassName>Server(App):
    """
    Server class that holds runtime state as direct attributes.
    Required methods to implement:
      - __init__(spec): initialize ALL attributes declared in SPEC["init_state"].
      - get_state() -> Dict[str, Any] | None: return a serializable snapshot.
      - load_state(state_dict: Dict[str, Any]) -> None: restore from a snapshot.

    Implementation notes:
    - Do NOT flatten dataclass fields into top-level dict keys.
      Assign instances/lists/dicts directly to the attribute named by init_state.key
      (e.g., self.location = LocationData(...)).
    - This template intentionally contains no generic logic. Fill in minimal, direct
      assignments based on your SPEC (defaults/overrides), and mirror that in the
      (de)serialization methods below.
    """

    def __init__(self, spec: Dict[str, Any]) -> None:
        """
        Initialize ALL attributes defined by SPEC["init_state"].

        How to implement (examples — replace with your real attributes):
            # Example: simple literals
            self.user_email: str = "user@meta.com"
            self.view_limit: int = 5

            # Example: a single dataclass instance
            # self.inbox = EmailFolder(folder_name=EmailFolderName.INBOX)

            # Example: a list of dataclass instances
            # self.backup_locations = [
            #     LocationData(coordinates=(1.0, 2.0)),
            #     LocationData(coordinates=(3.0, 4.0)),
            # ]

            # Example: a dict of dataclass instances
            # self.named_locations = {
            #     "home": LocationData(coordinates=(0.0, 0.0)),
            #     "office": LocationData(coordinates=(48.8566, 2.3522)),
            # }
        """
        super().__init__()
        ...
    
    def get_state(self) -> Dict[str, Any] | None:
        """
        Return a serializable snapshot of current state.

        How to implement (example pattern):
            result = {
                "user_email": self.user_email,
                "view_limit": self.view_limit,
                # If you keep a dict of dataclasses:
                # "folders": {k.value: v.get_state() for k, v in self.folders.items()},
            }
            return result
        """
        ...
    
    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Restore state from a snapshot produced by get_state().

        How to implement (example pattern):
            self.user_email = state_dict["user_email"]
            self.view_limit = state_dict["view_limit"]

            # If you keep a dict of dataclasses:
            # self.folders = {}
            # for folder_key in state_dict["folders"]:
            #     email_folder = EmailFolder(EmailFolderName(folder_key))
            #     email_folder.load_state(state_dict["folders"][folder_key])
            #     self.folders[email_folder.folder_name] = email_folder
        """
        ...
        ```
'''
