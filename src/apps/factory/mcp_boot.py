# mcp_boot.py
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

# 添加项目根目录到 sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
try:
    _wd = os.environ.get("MCP_SANDBOX_WORKDIR")
    if _wd and os.path.isdir(_wd):
        os.chdir(_wd)
except Exception:
    pass

# 确保能 import 到你的 Factory
from src.apps.factory.mcp_server_factory import MCPServerFactory

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file-path", type=str, default=None, help="Python .py file containing the App class")
    p.add_argument("--class-name", type=str, default=None, help="Class name inside the module (optional)")
    p.add_argument("--code-path", type=str, default=None, help="Alternative to --file-path: path to a .py code file to exec")
    p.add_argument("--spec-path", type=str, default=None, help="JSON file to pass as spec to App.__init__")
    p.add_argument("--auto-load", type=str, default=None, help="Auto load state JSON on start (optional)")
    p.add_argument("--auto-save", type=str, default=None, help="Auto save state JSON on exit (optional)")
    args = p.parse_args()

    # 读取 spec
    spec = {}
    if args.spec_path:
        with open(args.spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)

    # 依据输入来源构造 Factory
    if args.file_path:
        factory = MCPServerFactory(app_cls=None, file_path=args.file_path, class_name=args.class_name)
    elif args.code_path:
        code_str = open(args.code_path, "r", encoding="utf-8").read()
        factory = MCPServerFactory(app_cls=None, code_str=code_str, class_name=args.class_name)
    else:
        print("Either --file-path or --code-path must be provided", file=sys.stderr)
        sys.exit(2)

    # 创建 FastMCP
    mcp = factory.create(
        spec,
        auto_load_path=args.auto_load,
        auto_save_path=args.auto_save,
        register_atexit_save=True,
        server_meta={"name": args.class_name or "MCP-Server", "runner": "mcp_boot"},
    )

    # 以 stdio 模式运行，供上层 Manager 通过 mcp.client.stdio 连接
    # 不要返回；直到进程退出
    mcp.run()

if __name__ == "__main__":
    main()
