# Standard library
import os
import json
import csv
from pathlib import Path
from typing import Any, List, Dict
import yaml
import toml
import xmltodict

# ── third-party round-trip helpers ────────────────────────────────────────────
from ruamel.yaml import YAML                         
import tomlkit                                      
from configobj import ConfigObj                     
from lxml import etree
try:
    from src.core.toolkits import ExcelToolkit
except ImportError:
    # Fallback for when running tests or when src module is not available
    ExcelToolkit = None
import pandas as pd

class StructuredFileHandler:
    """
    A handler for parsing various structured file formats:
    CSV, JSON, Excel (XLS/XLSX), XML, YAML, TOML, INI/CFG/CONF.
    """
    def __init__(self): 
        if ExcelToolkit is not None:
            self.excel_tool = ExcelToolkit()
        else:
            self.excel_tool = None
        
        # ruamel.yaml instance (round-trip safe)
        self._yaml = YAML()
        self._yaml.preserve_quotes = True
        self._yaml.indent(mapping=2, sequence=4, offset=2)

    def read(self, file_path: str) -> str:
        """
        Determine file extension and dispatch to the appropriate parser.
        
        Args:
            file_path (str): Path to the structured file.
        
        Returns:
            str: Parsed content as string in Markdown format.
        """
        try:
            ext = os.path.splitext(file_path.lower())[1]
            
            if ext in {".json", ".jsonld"}:
                with open(file_path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    return f"```yaml\n{yaml.dump(data, allow_unicode=True, indent=4)}\n```"
            
            if ext == ".jsonl":
                with open(file_path, "r", encoding="utf-8") as fp:
                    data = [json.loads(line) for line in fp if line.strip()]
                    return f"```yaml\n{yaml.dump(data, allow_unicode=True, indent=4)}\n```"
            
            if ext == '.csv':
                with open(file_path, 'r', encoding='utf-8') as fp:
                    reader = csv.reader(fp)
                    data = list(reader)
                return f"```yaml\n{yaml.dump(data, allow_unicode=True, indent=4)}\n```"
            
            if ext in {'.xls', '.xlsx'}:
                df = pd.read_excel(file_path)
                data = df.to_dict('records')
                return f"```yaml\n{yaml.dump(data, allow_unicode=True, indent=4)}\n```"
            
            if ext == ".xml":
                with open(file_path, 'r', encoding='utf-8') as fp:
                    data = xmltodict.parse(fp.read())
                return f"```yaml\n{yaml.dump(data, allow_unicode=True, indent=4)}\n```"
            
            if ext in {".yaml", ".yml"}:
                with open(file_path, "r", encoding="utf-8") as fp:
                    data = self._yaml.load(fp)
                    return f"```yaml\n{yaml.dump(data, allow_unicode=True, indent=4)}\n```"
            
            if ext == ".toml":
                text = Path(file_path).read_text(encoding="utf-8")
                data = tomlkit.parse(text)
                return f"```yaml\n{yaml.dump(data.unwrap(), allow_unicode=True, indent=4)}\n```"
            
            if ext in {".ini", ".cfg", ".conf"}:
                config = ConfigObj(file_path, encoding="utf-8")
                def config_to_dict(conf):
                    result = {}
                    for key, value in conf.items():
                        if isinstance(value, dict):
                            result[key] = config_to_dict(value)
                        else:
                            result[key] = value
                    return result
                data = config_to_dict(config)
                return f"```yaml\n{yaml.dump(data, allow_unicode=True, indent=4)}\n```"
            
            raise ValueError(f"Unsupported structured file format: {file_path}")
        except (FileNotFoundError, PermissionError, json.JSONDecodeError, yaml.YAMLError, toml.TomlDecodeError) as e:
            raise ValueError(f"Error reading file {file_path}: {str(e)}") from e

    def write(self, file_path: str, content: Any) -> bool:
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            ext = path.suffix.lower()

            if ext in {".json", ".jsonld"}:
                if not isinstance(content, (dict, list)):
                    raise ValueError("JSON write expects dict or list")
                with open(path, "w", encoding="utf-8") as fp:
                    json.dump(content, fp, ensure_ascii=False, indent=2)
                return True

            if ext == ".jsonl":
                lines = content if isinstance(content, list) else [content]
                with open(path, "w", encoding="utf-8") as fp:
                    for obj in lines:
                        fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
                return True

            if ext == ".csv":
                if not (isinstance(content, list) and all(isinstance(row, list) for row in content)):
                    raise ValueError("CSV write expects List[List[str]]")
                with open(path, "w", newline="", encoding="utf-8") as fp:
                    writer = csv.writer(fp, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                    writer.writerows(content)
                return True

            if ext in {".xls", ".xlsx"}:
                if not (isinstance(content, list) and all(isinstance(r, dict) for r in content)):
                    raise ValueError("Excel write expects List[Dict]")
                if self.excel_tool is not None:
                    self.excel_tool.write_excel(str(path), content)
                else:
                    # Fallback: use pandas to write Excel files
                    import pandas as pd
                    df = pd.DataFrame(content)
                    df.to_excel(str(path), index=False)
                return True

            if ext == ".xml":
                if not isinstance(content, dict):
                    raise ValueError("XML write expects dict")
                xml_str = xmltodict.unparse(content, pretty=True)
                path.write_text(xml_str, encoding="utf-8")
                return True

            if ext in {".yaml", ".yml"}:
                with path.open("w", encoding="utf-8") as fp:
                    self._yaml.dump(content, fp)
                return True

            if ext == ".toml":
                if isinstance(content, dict):
                    content = tomlkit.dumps(content)
                elif not isinstance(content, tomlkit.TOMLDocument):
                    raise ValueError("TOML write expects dict or tomlkit.TOMLDocument")
                path.write_text(content, encoding="utf-8")
                return True

            if ext in {".ini", ".cfg", ".conf"}:
                if not isinstance(content, dict):
                    raise ValueError("INI write expects dict")
                config = ConfigObj()
                for section, values in content.items():
                    config[section] = {}
                    if isinstance(values, dict):
                        for k, v in values.items():
                            config[section][k] = v
                config.filename = str(path)
                config.write()
                return True
        
            raise ValueError(f"Unsupported structured file format: {file_path}")
        except Exception as e:
            print(f"Write error: {e}")
            return False

    def edit(self, file_path, new_data):
        # Edit structured data in a file.
        pass
