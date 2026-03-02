from __future__ import annotations
"""
沙箱工具函数
"""
import os
import json
import logging
import requests
import tomli
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


def get_github_repo_files(github_url):
    """获取GitHub仓库文件列表"""
    # 解析 owner 和 repo 名
    parts = github_url.rstrip('/').split('/')
    owner, repo = parts[-2], parts[-1]

    url = f"https://api.github.com/repos/{owner}/{repo}/contents/"
    logger.debug(f"GitHub API URL: {url}")
    resp = requests.get(url)
    if resp.status_code != 200:
        logger.error(f"访问失败: {url}")
        logger.error(f"返回内容: {resp.text}")
        return []
    items = resp.json()
    files = []
    for item in items:
        if item["type"] == "file":
            files.append({
                "name": item["name"],
                "download_url": item["download_url"]
            })
    return files


def find_dependencies(data):
    """递归查找依赖项"""
    deps = []
    if isinstance(data, dict):
        for k, v in data.items():
            if k == "dependencies":
                deps.append(v)
            else:
                deps.extend(find_dependencies(v))
    elif isinstance(data, list):
        for item in data:
            deps.extend(find_dependencies(item))
    return deps


def get_requirements(tools_path, mcp_server):
    """从工具配置文件获取依赖需求"""
    logger.debug(f"Tools path: {tools_path}")
    logger.debug(f"MCP server: {mcp_server}")
    
    # 查找 github_url
    github_url = None
    for key, value in mcp_server.items():
        if key == "github_url":
            github_url = value
            logger.info(f"找到 github 地址：{github_url}")
            break
    if not github_url:
        logger.warning("未找到对应的 github_url！")
        return []
    
    # 获取文件列表
    files = get_github_repo_files(github_url)
    requirements = []
    
    for f in files:
        if f['name'].endswith('.toml'):
            logger.info(f"发现 toml 文件: {f['name']}")
            try:
                resp = requests.get(f['download_url'])
                if resp.status_code == 200:
                    # 保存到临时文件
                    tmp_path = f"/tmp/{f['name']}"
                    with open(tmp_path, 'wb') as tmp_file:
                        tmp_file.write(resp.content)
                    
                    # 解析 toml 文件
                    with open(tmp_path, 'rb') as tmp_file:
                        data = tomli.load(tmp_file)
                    
                    # 查找依赖
                    deps = find_dependencies(data)
                    requirements.extend(deps)
                    
                    # 删除临时文件
                    os.remove(tmp_path)
                    logger.debug(f"已删除临时文件: {tmp_path}")
                else:
                    logger.error(f"下载失败: {f['download_url']}")
            except Exception as e:
                logger.error(f"处理 toml 文件时出错: {e}")
        elif f['name'] == 'requirements.txt':
            logger.info(f"发现 requirements.txt 文件: {f['name']}")
            try:
                resp = requests.get(f['download_url'])
                if resp.status_code == 200:
                    content = resp.text
                    lines = content.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            requirements.append(line)
                    logger.info(f"requirements.txt 依赖: {requirements}")
                else:
                    logger.error(f"下载失败: {f['download_url']}")
            except Exception as e:
                logger.error(f"处理 requirements.txt 时出错: {e}")

    return requirements
