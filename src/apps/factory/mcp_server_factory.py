# mcp_factory.py
from __future__ import annotations
import atexit, importlib.util, inspect, json, os, sys, tempfile, types
from dataclasses import is_dataclass, asdict
from typing import Any, Dict, Optional, Type

try:
    from mcp.server.fastmcp import FastMCP
except Exception:
    FastMCP = None


def _to_serializable(obj: Any) -> Any:
    """尽量把 dataclass/复杂对象变成可 JSON 化的数据（给 auto-save 用）"""
    if is_dataclass(obj):
        return _to_serializable(asdict(obj))
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    return obj


def _load_module_from_path(file_path: str, module_name: Optional[str] = None) -> types.ModuleType:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No such file: {file_path}")
    module_name = module_name or f"_dynmod_{os.path.basename(file_path).replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def _load_module_from_code(code_str: str, module_name: Optional[str] = None) -> types.ModuleType:
    module_name = module_name or "_dynmod_inline_code"
    mod = types.ModuleType(module_name)
    exec(compile(code_str, filename=f"<{module_name}>", mode="exec"), mod.__dict__)
    sys.modules[module_name] = mod
    return mod


def _find_app_class(mod: types.ModuleType, class_name: Optional[str]) -> Type:
    """
    自动发现：优先用 class_name；否则找：
      1) 同时具备 _initialize_mcp_server、load_state、get_state 方法且继承自 App 的类；
      2) 具备 _initialize_mcp_server 方法的类；
      3) 若存在 App 基类，则找其子类；
      4) 退化为首个满足(2)的类。
    """
    if class_name:
        cls = getattr(mod, class_name, None)
        if cls is None or not inspect.isclass(cls):
            raise AttributeError(f"Class '{class_name}' not found in module {mod.__name__}")
        return cls

    # 获取 App 基类
    AppBase = getattr(mod, "App", None)
    
    # 候选1：同时满足所有条件的类
    def _has_required_methods(cls) -> bool:
        """检查类是否具有所需的方法"""
        required_methods = ["_initialize_mcp_server", "load_state", "get_state"]
        return all(
            hasattr(cls, method) and callable(getattr(cls, method))
            for method in required_methods
        )
    
    def _inherits_from_app(cls) -> bool:
        """检查类是否继承自 App"""
        if AppBase is None or not inspect.isclass(AppBase):
            return False
        return issubclass(cls, AppBase) and cls is not AppBase
    
    # 优先查找同时满足所有条件的类
    perfect_candidates = []
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if _has_required_methods(obj) and _inherits_from_app(obj):
            perfect_candidates.append(obj)
    
    if perfect_candidates:
        return perfect_candidates[0]

    # 候选2：有 _initialize_mcp_server 方法的类
    mcp_candidates = []
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if hasattr(obj, "_initialize_mcp_server") and callable(getattr(obj, "_initialize_mcp_server")):
            mcp_candidates.append(obj)

    if mcp_candidates:
        return mcp_candidates[0]

    # 候选3：如果模块里有 App 基类，找它的子类
    if inspect.isclass(AppBase):
        subs = [c for _, c in inspect.getmembers(mod, inspect.isclass)
                if issubclass(c, AppBase) and c is not AppBase]
        if subs:
            return subs[0]

    raise RuntimeError("No suitable App class found (need a class with _initialize_mcp_server, load_state, get_state methods and inherit from App).")


class MCPServerFactory:
    """
    精简版 Factory（无 admin 注入）：
    - 支持：app_cls / file_path / code_str 三种来源
    - 自动：load_state（可选）、进程退出时 save_state（可选）
    - 只负责拿 FastMCP；stdio 运行交给你的 MCPServerManager 等外部组件
    """

    def __init__(
        self,
        app_cls: Optional[Type] = None,
        *,
        file_path: Optional[str] = None,
        class_name: Optional[str] = None,
    ) -> None:
        if not FastMCP:
            raise RuntimeError("FastMCP is not available")

        self._app_cls = app_cls
        self._file_path = file_path
        self._class_name = class_name

        if not (app_cls or file_path):
            raise ValueError("Provide one of: app_cls | file_path | code_str")

    def _resolve_app_class(self) -> Type:
        if self._app_cls is not None:
            return self._app_cls

        if self._file_path:
            mod = _load_module_from_path(self._file_path)
            return _find_app_class(mod, self._class_name)

        raise RuntimeError("Failed to resolve app class")

    def create(
        self,
        spec: Dict[str, Any],
        *,
        auto_load_path: Optional[str] = None,
        auto_save_path: Optional[str] = None,
        register_atexit_save: bool = True,
        server_meta: Optional[Dict[str, Any]] = None,
    ):
        """
        返回 FastMCP 实例（来自 app._initialize_mcp_server()）
        - 不注入任何 admin 工具
        - auto_load_path：存在则启动时自动加载 App 状态（JSON）
        - auto_save_path：进程退出时自动保存 App 状态（JSON）
        - server_meta：若 FastMCP 实现支持 set_server_meta，会尝试设置（可选）
        """
        app_cls = self._resolve_app_class()
        app = app_cls(spec)  # 你的 App __init__(spec) 负责初始化状态

        # 启动自动 load
        if auto_load_path and os.path.exists(auto_load_path):
            try:
                with open(auto_load_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                if hasattr(app, "load_state"):
                    app.load_state(state)
            except Exception as e:
                print(f"[Factory] auto-load failed: {e}")

        # 初始化 MCP
        if not hasattr(app, "_initialize_mcp_server"):
            raise AttributeError(f"{app_cls.__name__} has no _initialize_mcp_server()")
        mcp = app._initialize_mcp_server()
        if mcp is None:
            raise RuntimeError("app._initialize_mcp_server() returned None")

        # 绑定 env
        mcp._env = app  # 方便外部访问

        # 设置元信息（如果 FastMCP 实现支持）
        if server_meta and hasattr(mcp, "set_server_meta"):
            try:
                mcp.set_server_meta(server_meta)
            except Exception:
                pass

        # 退出自动 save
        if auto_save_path and register_atexit_save:
            def _save_on_exit():
                try:
                    state = {}
                    if hasattr(app, "get_state"):
                        state = app.get_state() or {}
                    state = _to_serializable(state)
                    os.makedirs(os.path.dirname(auto_save_path), exist_ok=True)
                    with open(auto_save_path, "w", encoding="utf-8") as f:
                        json.dump(state, f, ensure_ascii=False)
                except Exception as e:
                    print(f"[Factory] auto-save failed: {e}")
            atexit.register(_save_on_exit)

        return mcp
