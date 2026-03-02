# Environment Generation Module

环境生成模块用于从 MCP (Model Context Protocol) 服务器工具定义自动生成强化学习环境。该模块支持多种生成策略，从简单的单轮生成到复杂的多智能体协作生成。

## 📁 目录结构

```
env_generation/
├── README.md                    # 本文档
├── __init__.py                  # 模块初始化
├── agentic_framework.py         # 智能体框架核心
├── agentic_multi_turn.py        # 多轮智能体生成
├── agentic_single_turn.py       # 单轮智能体生成
├── single_turn.py               # 简单单轮生成
├── naive_multi_turn.py          # 朴素多轮生成
├── simulation_env_planner.py    # 环境规划器
├── env_eval.py                  # 环境评估器
├── agentic_utils/               # 智能体工具
│   ├── __init__.py
│   └── simulation_tool_registry.py
├── prompt/                      # 提示模板
│   ├── __init__.py
│   ├── base_prompt.py
│   ├── base_template.py
│   └── mcp_prompt.py
├── test_utils/                  # 测试工具
│   ├── __init__.py
│   ├── test_fix_env_code.py
│   ├── unified_agentic_env_test.py
│   ├── unified_base_env_test.py
│   └── unified_single_env_test.py
└── utils/                       # 通用工具
    ├── __init__.py
    ├── create_external_files.py
    ├── extract_tool_defs.py
    ├── load_data.py
    └── save_environment.py
```

## 🚀 核心功能

### 1. 环境生成策略

#### 🔹 简单单轮生成 (`single_turn.py`)
- **用途**: 基础的环境代码生成
- **特点**: 直接从 MCP 工具定义生成环境类
- **适用场景**: 简单工具集，快速原型开发

```python
from src.env_generation.single_turn import generate_environment_from_mcp

# 生成环境
env_code = generate_environment_from_mcp("server_name")
```

#### 🔹 智能体单轮生成 (`agentic_single_turn.py`)
- **用途**: 使用单个智能体进行深度研究和代码生成
- **特点**: 结合研究和代码生成，生成更完善的环境
- **适用场景**: 需要深入理解工具语义的复杂环境

```python
from src.env_generation.agentic_single_turn import generate_environment_from_mcp

# 使用智能体生成环境
env_code = generate_environment_from_mcp(
    server_name="server_name",
    model="gpt-4.1-mini",
    output_dir="temp/agentic/envs"
)
```

#### 🔹 智能体多轮生成 (`agentic_multi_turn.py`)
- **用途**: 多智能体协作生成高质量环境
- **特点**: 研究智能体 + 代码生成智能体协作
- **适用场景**: 复杂业务逻辑，需要高质量实现

```python
from src.env_generation.agentic_multi_turn import generate_environment_from_mcp

# 多智能体协作生成
env_code, env_name = generate_environment_from_mcp(
    server_name="server_name",
    model="gpt-4.1-mini",
    output_dir="temp/agentic/envs"
)
```

### 2. 智能体框架 (`agentic_framework.py`)

提供智能体初始化和环境生成的核心框架：

```python
from src.env_generation.agentic_framework import initialize_agent, generate_mcp_environment

# 初始化智能体
agents = initialize_agent(
    results_base_dir="temp/agentic",
    max_tokens=8192,
    enable_research=True,
    enable_codegen=True
)

# 生成环境
env_code = generate_mcp_environment(
    env_name="my_env",
    agents=agents,
    mcp_prompt=prompt,
    save_json_report=True
)
```

### 3. 环境评估 (`env_eval.py`)

全面的环境质量评估系统：

- **可执行性**: 语法和导入检查
- **模式保真度**: 函数签名匹配度
- **功能性**: LLM 审计每个工具实现
- **语义保真度**: 模式与代码的语义相似度
- **真实性**: 模拟调用的真实性评估

```python
from src.env_generation.env_eval import evaluate_environment

# 评估环境质量
score, details = evaluate_environment(
    code_path="path/to/env.py",
    mcp_data=mcp_tool_definitions
)
```

## 🛠️ 工具模块

### 📁 `utils/` - 通用工具
- `load_data.py`: 数据加载工具
- `extract_tool_defs.py`: 工具定义提取
- `save_environment.py`: 环境保存工具
- `create_external_files.py`: 外部文件创建

### 📁 `prompt/` - 提示模板
- `base_prompt.py`: 基础提示类
- `base_template.py`: 模板管理
- `mcp_prompt.py`: MCP 专用提示

### 📁 `test_utils/` - 测试工具
- `unified_agentic_env_test.py`: 智能体环境测试
- `unified_base_env_test.py`: 基础环境测试
- `test_fix_env_code.py`: 代码修复测试

### 📁 `agentic_utils/` - 智能体工具
- `simulation_tool_registry.py`: 模拟工具注册表

## 📊 使用示例

### 基础使用

```python
# 1. 简单生成
from src.env_generation.single_turn import generate_environment_from_mcp
env_code = generate_environment_from_mcp("filesystem")

# 2. 智能体生成
from src.env_generation.agentic_single_turn import generate_environment_from_mcp
env_code = generate_environment_from_mcp("filesystem", model="gpt-4.1-mini")

# 3. 多智能体协作
from src.env_generation.agentic_multi_turn import generate_environment_from_mcp
env_code, env_name = generate_environment_from_mcp("filesystem")
```

### 高级配置

```python
from src.env_generation.agentic_framework import initialize_agent
from src.env_generation.prompt import MCPPrompt

# 自定义智能体配置
agents = initialize_agent(
    results_base_dir="custom/output",
    max_tokens=16384,
    enable_research=True,
    enable_codegen=True
)

# 自定义提示
prompt = MCPPrompt(
    server_name="my_server",
    mcp_data=tool_definitions,
    base_env_template_path="custom/template.py"
)
```

## 🔧 配置选项

### 模型配置
- 支持 OpenAI GPT 系列模型
- 可配置 `max_tokens`、温度等参数
- 支持自定义模型平台

### 输出配置
- 可配置输出目录
- 支持保存中间结果和报告
- 可选择是否保存 JSON 格式报告

### 智能体配置
- 可启用/禁用研究智能体
- 可启用/禁用代码生成智能体
- 支持自定义智能体参数

## 🎯 最佳实践

1. **选择合适的生成策略**:
   - 简单工具 → `single_turn.py`
   - 复杂逻辑 → `agentic_single_turn.py`
   - 高质量需求 → `agentic_multi_turn.py`

2. **环境评估**:
   - 生成后使用 `env_eval.py` 评估质量
   - 根据评估结果调整生成策略

3. **模板定制**:
   - 根据需要定制基础环境模板
   - 使用自定义提示模板优化生成效果

4. **测试验证**:
   - 使用 `test_utils/` 中的工具验证生成的环境
   - 确保环境符合 UnifiedBaseEnv 规范

## 🔗 相关模块

- `src.core.agents`: 智能体核心框架
- `src.core.models`: 模型管理
- `src.core.prompts`: 提示模板系统
- `src.utils.unified_base_env`: 统一基础环境类

## 📝 开发指南

### 添加新的生成策略

1. 创建新的生成器文件
2. 实现 `generate_environment_from_mcp` 函数
3. 添加相应的测试用例
4. 更新文档

### 扩展评估指标

1. 在 `env_eval.py` 中添加新的评估函数
2. 更新权重配置
3. 添加相应的测试用例

---

*该模块是 AgentGen-v2 项目的核心组件，用于自动化生成高质量的强化学习环境。*