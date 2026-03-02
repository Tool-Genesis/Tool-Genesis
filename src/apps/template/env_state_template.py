"""
**KEY REQUIREMENTS**
- For each item, include: `description` and necessary `example_values`.
- `value_mode` ∈ {literal, dataclass, list_of_dataclass, dict_of_dataclass}.
- literal: must provide `json_type` (string/number/integer/boolean).
- dataclass modes: provide `dataclass_ref`; do **not** set `json_type`.
- In `dataclass_spec.fields`, use `json_schema` for serialized shape; `py_type_hint` is only a code-generation hint—do **not** embed Python objects in JSON.
- Put convenience APIs under `dataclass_spec.helper_methods`: only `name/description/inputs/returns`; implementation is left to the agent.
- Use `related_fields` references as `Class.field` or a bare `init_state.key`.
- Use ISO-8601 with timezone for datetimes; represent tuples as fixed-length arrays via `prefixItems`.

**DESIGN RULES**
- **No duplication:** simple literals live in `init_state`; complex structures are defined once in `dataclass_spec` and referenced via `dataclass_ref`.
- For objects/lists/dicts of objects: define the class in `dataclass_spec`, then initialize with `init_overrides` / `init_list` / `init_dict`.
- Choose **either** `default` **or** `default_strategy`—never both.
- **Avoid None/empty defaults:** Provide realistic default values instead of None, empty strings, or empty collections. Use `default_strategy` only when a specific initialization pattern is required.
- Names must be unique in scope: `init_state.key`, `dataclass_spec.class_name`, and each class’s `fields.name`.
- `extra_files_spec` is for **unstructured** assets only (docs/images/examples); paths should be sandboxed or relative; do not place structured data here.
- When generating, output **one JSON block at a time** (template *or* example).

## Templatae for dataclass_spec
{
"$schema": "https://json-schema.org/draft/2020-12/schema",
"version": "1.0.0",
"env_name": "REPLACE_ME",
"description": "REPLACE_ME",

"init_state": [
    {
    "key": "REPLACE_ME",
    "description": "REPLACE_ME",

    "value_mode": "literal",
    "x_options": { "value_mode": ["literal", "dataclass", "list_of_dataclass", "dict_of_dataclass"] },

    "json_type": "string",
    "x_options_literal_types": ["string", "number", "integer", "boolean"],
    "py_type_hint": "REPLACE_ME",

    "dataclass_ref": null,
    "init_overrides": null,
    "init_list": null,
    "init_dict": null,
    "key_type": null,

    "default": null,
    "default_strategy": "REPLACE_ME",
    "x_options_default_strategy": ["class_default", "empty_list", "empty_dict"],

    "related_fields": [],
    "example_values": []
    }
],

"dataclass_spec": [
    {
    "class_name": "REPLACE_ME",
    "description": "REPLACE_ME",

    "fields": [
        {
        "name": "REPLACE_ME",
        "description": "REPLACE_ME",
        "json_schema": {},
        "py_type_hint": "REPLACE_ME",
        "default": null,
        "related_fields": [],
        "example_values": []
        }
    ],

    "helper_methods": [
        {
        "name": "REPLACE_ME",
        "description": "REPLACE_ME",
        "inputs": [
            { "name": "REPLACE_ME", "py_type_hint": "REPLACE_ME", "description": "REPLACE_ME" }
        ],
        "returns": { "py_type_hint": "REPLACE_ME", "description": "REPLACE_ME" }
        }
    ],

    "language_hints": {
        "python": {
        "required_imports": []
        }
    }
    }
],

"extra_files_spec": [
{
    "name": "REPLACE_ME",
    "description": "REPLACE_ME",
    "content_kind": "unstructured", 
    "mime_type": "",
    "path": "REPLACE_ME/REPLACE_ME.ext",
}
]
}

## Example
{
"$schema": "https://json-schema.org/draft/2020-12/schema",
"version": "1.0.0",
"env_name": "AstroBaziEnv",
"description": "MCP server env: Bazi calculation with astronomical inputs.",

"init_state": [
    {
    "key": "timezone",
    "description": "IANA timezone name.",
    "value_mode": "literal",
    "json_type": "string",
    "py_type_hint": "str",
    "default": "Asia/Singapore"
    },
    {
    "key": "location",
    "description": "Primary LocationData instance.",
    "value_mode": "dataclass",
    "dataclass_ref": "LocationData",
    "py_type_hint": "LocationData",
    "init_overrides": {
        "coordinates": [1.3521, 103.8198],
        "timezone_info": { "tzid": "Asia/Singapore", "region": "SG" }
    }
    },
    {
    "key": "backup_locations",
    "description": "Fallback list of locations.",
    "value_mode": "list_of_dataclass",
    "dataclass_ref": "LocationData",
    "py_type_hint": "List[LocationData]",
    "init_list": [
        { "coordinates": [31.2304, 121.4737], "timezone_info": { "tzid": "Asia/Shanghai" } },
        { "coordinates": [22.3193, 114.1694], "timezone_info": { "tzid": "Asia/Hong_Kong" } }
    ]
    },
    {
    "key": "named_locations",
    "description": "Dictionary of named locations.",
    "value_mode": "dict_of_dataclass",
    "dataclass_ref": "LocationData",
    "py_type_hint": "Dict[str, LocationData]",
    "key_type": "string",
    "init_dict": {
        "home":   { "coordinates": [0.0, 0.0], "timezone_info": { "tzid": "UTC" } },
        "office": { "coordinates": [48.8566, 2.3522], "timezone_info": { "tzid": "Europe/Paris" } }
    }
    }
],

"dataclass_spec": [
    {
    "class_name": "LocationData",
    "description": "Geographic location for astronomy/Bazi.",
    "fields": [
        {
        "name": "coordinates",
        "description": "Latitude, Longitude.",
        "json_schema": {
            "type": "array",
            "prefixItems": [{ "type": "number" }, { "type": "number" }],
            "minItems": 2,
            "maxItems": 2
        },
        "py_type_hint": "Tuple[float, float]",
        "default": [0.0, 0.0]
        },
        {
        "name": "timezone_info",
        "description": "Region and tz database id.",
        "json_schema": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
            "region": { "type": "string" },
            "tzid": { "type": "string" }
            },
            "required": ["tzid"]
        },
        "py_type_hint": "Dict[str, str]"
        }
    ],
    "helper_methods": [
        {
        "name": "validate_coordinates",
        "description": "Return True if lat/lon are in valid ranges.",
        "inputs": [
            { "name": "coords", "py_type_hint": "Tuple[float, float]", "description": "Latitude, Longitude." }
        ],
        "returns": { "py_type_hint": "bool", "description": "Validation result." }
        }
    ]
    },
    {
    "class_name": "BaziCalculationData",
    "description": "Bazi results and key solar timestamps.",
    "fields": [
        {
        "name": "solar_times",
        "description": "Mapping name -> ISO8601 timestamp.",
        "json_schema": {
            "type": "object",
            "additionalProperties": { "type": "string", "format": "date-time" }
        },
        "py_type_hint": "Dict[str, datetime]"
        },
        {
        "name": "bazi_data",
        "description": "Heavenly stems & earthly branches.",
        "json_schema": { "type": "object" },
        "py_type_hint": "Dict[str, Any]"
        }
    ],
    "helper_methods": [
        {
        "name": "derive_solar_times",
        "description": "Compute and return key solar timestamps for a date and location.",
        "inputs": [
            { "name": "location", "py_type_hint": "LocationData", "description": "Geographic context." },
            { "name": "date", "py_type_hint": "datetime", "description": "Target date." }
        ],
        "returns": { "py_type_hint": "Dict[str, datetime]", "description": "Mapping name -> timestamp." }
        }
    ]
    }
],

"extra_files_spec": [
    {
    "name": "user_guide",
    "description": "End-user quickstart notes.",
    "content_kind": "unstructured",
    "mime_type": "text/markdown",
    "path": "assets/docs/guide.md"
    },
    {
    "name": "architecture_diagram",
    "description": "High-level architecture diagram.",
    "content_kind": "unstructured",
    "mime_type": "image/png",
    "path": "assets/diagrams/overview.png"
    }
]
}
"""