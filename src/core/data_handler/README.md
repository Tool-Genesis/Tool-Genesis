# Core Data Handler

这个模块提供了核心的状态管理功能，允许保存和加载智能体状态、对话历史和会话数据。它是基于 `src/simulation/data_handler` 的功能进行改进和适配的。

## 功能特性

- **状态导出 (StateExporter)**: 将智能体状态、对话和会话数据导出为JSON格式
- **状态导入 (StateImporter)**: 从JSON格式恢复智能体状态、对话和会话数据
- **备份管理**: 创建和管理状态的时间戳备份
- **版本控制**: 支持状态格式的版本管理
- **数据验证**: 验证状态文件的完整性和格式

## 主要组件

### 数据模型 (models.py)

定义了以下核心数据结构：

- `AgentState`: 智能体状态
- `ConversationState`: 对话状态  
- `SessionState`: 会话状态
- `StateMetadata`: 状态元数据
- `ExportedXXX`: 导出格式的数据模型

### 状态导出器 (exporter.py)

`StateExporter` 类提供以下功能：

- 导出智能体状态到JSON
- 导出对话状态到JSON
- 导出完整会话状态到JSON
- 创建时间戳备份
- 支持自定义输出路径和格式

### 状态导入器 (importer.py)

`StateImporter` 类提供以下功能：

- 从JSON导入智能体状态
- 从JSON导入对话状态
- 从JSON导入会话状态
- 列出和管理备份文件
- 验证状态文件格式

## 使用示例

### 基本使用

```python
from src.core.data_handler import StateExporter, StateImporter, AgentState

# 创建导出器和导入器
exporter = StateExporter()
importer = StateImporter()

# 创建一个示例智能体状态
agent_state = AgentState(
    agent_id="agent_001",
    agent_type="chat_agent",
    configuration={"model": "gpt-4", "temperature": 0.7},
    memory_blocks=[],
    tool_calls_history=[],
    current_context={"session_id": "session_001"}
)

# 导出状态到文件
success = exporter.export_to_file(
    state=agent_state,
    output_path="states/agent_001.json"
)

# 从文件导入状态
restored_state, metadata = importer.import_from_file("states/agent_001.json")
```

### 创建备份

```python
# 创建备份
backup_path = exporter.create_backup(
    state=agent_state,
    backup_dir="backups",
    prefix="agent_backup"
)

# 列出可用备份
backups = importer.list_backups(backup_dir="backups", state_type="agent")

# 从备份恢复
if backups:
    restored_state, metadata = importer.restore_from_backup(backups[0]["file_path"])
```

### 会话状态管理

```python
from src.core.data_handler import SessionState, ConversationState

# 创建会话状态
session_state = SessionState(
    session_id="session_001",
    agents=[agent_state],
    conversations=[
        ConversationState(
            conversation_id="conv_001",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            participants=["user", "agent_001"]
        )
    ],
    global_context={"theme": "dark", "language": "zh-CN"}
)

# 导出完整会话
json_str = exporter.export_session_state_to_json(
    session_state=session_state,
    output_path="sessions/session_001.json"
)
```

### 验证状态文件

```python
# 验证状态文件
validation_result = importer.validate_state_file("states/agent_001.json")

if validation_result["valid"]:
    print(f"文件有效，类型: {validation_result['type']}")
else:
    print(f"文件无效: {validation_result['error']}")
```

## 与原始 simulation data_handler 的区别

1. **专注于核心状态**: 专门处理智能体、对话和会话状态，而不是复杂的仿真场景
2. **简化的数据模型**: 使用更简洁的数据结构，适合核心功能使用
3. **增强的备份功能**: 提供更完善的备份和恢复机制
4. **更好的错误处理**: 改进了错误处理和验证逻辑
5. **灵活的导入导出**: 支持多种数据格式和自定义选项

## 注意事项

- 所有状态文件使用UTF-8编码
- 支持的版本: `core_state_v1`
- 备份文件按创建时间自动命名
- 大型状态对象可能需要较长的序列化时间
- 建议定期清理旧的备份文件

## 扩展性

该模块设计为可扩展的，可以轻松添加：

- 新的状态类型
- 自定义序列化格式
- 压缩和加密功能
- 远程存储支持
- 增量备份功能