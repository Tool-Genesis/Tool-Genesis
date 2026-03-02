# TODO: Tool-Genesis 代码整理与公开发布

## 一、实验修复（Journal Extension）

### 1. 实验 B：多轮进化 — 实现真实交互循环
- **文件**: `scripts/experiment_journal/evolution/run_evolution.py`
- **问题**: 当前 round 1+ 生成代码后没有真正评估，feedback 是占位符文本（`[Pending evaluation for round {r}]`），导致进化循环是假的
- **修复**:
  - 每轮生成代码后，实际启动 MCP server 进行评估（至少：语法检查 + UT 执行 + L4 proxy）
  - 用真实评估结果生成 feedback，再输入下一轮
  - 参考 `src/env_evalution/` 下的 L2、L4 评估管线
- **优先级**: P0

### 2. 分析 C：错误分类 — 补充缺失错误类型
- **文件**: `scripts/experiment_journal/analysis_c_error_taxonomy.py`
- **问题**: `classify_l2_schema()` 中定义了 `arg_missing`、`arg_extra`、`tool_extra` 但从未被实际发射
  - `tool_extra`：预测工具未匹配到任何 GT 工具 — 需要从 `tool_matches` 中识别未匹配的 pred tools
  - `arg_missing`：GT 参数在预测工具中缺失 — 需要从 `arg_matches` 中识别未匹配的 gt args
  - `arg_extra`：预测工具有多余参数 — 需要从 `arg_matches` 中识别未匹配的 pred args
- **同时修复**: launch failure 子分类始终默认为 `launch_fail_runtime`，因为 `log_lines` 始终传 `None`；需要尝试从 l1_debug.json 中读取日志
- **优先级**: P1

### 3. 实验 A：工具复用性 — 更换指标
- **文件**: `scripts/experiment_journal/eval_reusability.py`
- **问题**: `_sr_from_details()` 使用 `solved` 布尔值，几乎全为 True，导致 train/test gap 接近 0，指标无区分度
- **修复**:
  - 改用 trajectory 级别的 `soft_avg` / `hard_rate` 数值指标（已确认数据中存在，如 0.408、0.556 等）
  - 同时计算 Normalized Reusability
- **优先级**: P0

### 4. 重新运行并生成报告
- 修复以上三项后，重新运行所有分析脚本
- 生成最终 report（含图表）
- **依赖**: 上述 1、2、3 全部完成

---

## 二、代码整理（公开发布前）

### 5. 清理项目根目录
- [ ] 删除临时/个人文件：`temp.txt`、`test.py`、`sync_env_code_from_codegen.py`、`debug_env/`
- [ ] 删除生成的 PDF 散落在根目录：`correlation_heatmap_icml.pdf`、`failure_shift_bar.pdf`、`heatmap_transposed_bold.pdf`、`json_schema_*.pdf`、`scaling_multiplier_effect.pdf`
- [ ] 删除 `cache/`、`__pycache__/` 目录
- [ ] 确认 `json_schema.csv`、`json_schema.json` 是否需要保留（若是中间产物则删除）

### 6. 整理 .gitignore
- [ ] 确保 `.env`、`temp/`、`cache/`、`__pycache__/`、`*.pdf`（根目录）、`.claude/` 等已在 .gitignore 中
- [ ] 添加 `data/` 下的大文件策略（或用 Git LFS）

### 7. 统一依赖管理
- [ ] 合并 `requirements.txt`、`src/apps/requirements.txt`、`src/utils/requirements.txt` 为一个或明确分层
- [ ] 确保 `matplotlib`、`seaborn` 等绘图依赖在 requirements.txt 中

### 8. 整理 scripts/ 目录结构
- [ ] `scripts/experiment_journal/` — 期刊实验脚本，确认文档完整
- [ ] `scripts/run_benchmark/` — 基准评测脚本
- [ ] `scripts/build_benchmark/` — 数据构建脚本
- [ ] `scripts/plot/` — 绘图脚本（与 `scripts/experiment_journal/plot/` 的关系需厘清）

### 9. 整理 src/ 核心代码
- [ ] `src/core/sandbox/` — 沙箱模块，确认之前的安全修复已合并
- [ ] `src/env_evalution/` — 注意目录名拼写是 `evalution`（typo），考虑是否修正为 `evaluation`
- [ ] `src/env_generation/` — 环境生成模块
- [ ] `src/utils/llm.py` — LLM 调用工具，确认多平台支持（openai / bailian / openrouter）文档清晰
- [ ] `src/apps/` — 应用模块

### 10. 数据文件整理
- [ ] `data/tool_genesis_v3.json` — 主数据集（86 servers），确认是否随仓库发布
- [ ] `data/task_split.json` — train/test 划分
- [ ] `data/analysis_*.json`、`data/experiment_*.json` — 实验结果文件，考虑是否放入 `results/` 或 `.gitignore`
- [ ] `data/sample_1.json` — 示例数据，确认用途

### 11. README 完善
- [ ] 补充项目整体架构说明（评估四层级：L1/L2/L3/L4）
- [ ] 补充期刊实验说明（实验 A~G）
- [ ] 补充数据集格式说明
- [ ] 补充论文引用信息

### 12. Notebook 整理
- [ ] `build_benchmark.ipynb` — 确认是否保留，清理输出
- [ ] `run_benchmark.ipynb` — 同上

---

## 三、已完成的实验脚本

| 脚本 | 功能 | 状态 |
|------|------|------|
| `scripts/experiment_journal/split_tasks.py` | 数据 70/30 train/test 划分 | 已完成 |
| `scripts/experiment_journal/analysis_c_error_taxonomy.py` | 错误分类分析 | 需修复 (#2) |
| `scripts/experiment_journal/analysis_d_completeness.py` | 工具集完备性分析 | 已完成 |
| `scripts/experiment_journal/eval_reusability.py` | 工具复用性评估 | 需修复 (#3) |
| `scripts/experiment_journal/evolution/run_evolution.py` | 多轮进化主循环 | 需修复 (#1) |
| `scripts/experiment_journal/evolution/feedback_collector.py` | 反馈收集器 | 已完成 |
| `scripts/experiment_journal/evolution/prompt_template.py` | 进化 prompt 模板 | 已完成 |
| `scripts/experiment_journal/evolution/eval_evolution.py` | 进化评估（轻量） | 已完成 |
| `scripts/experiment_journal/ablation_oracle.py` | 消融 E：Oracle vs Cascaded | 已完成（未运行生成） |
| `scripts/experiment_journal/plot/plot_error_taxonomy.py` | 错误分类绘图 | 已完成 |
| `scripts/experiment_journal/plot/plot_completeness.py` | 完备性绘图 | 已完成 |
| `scripts/experiment_journal/plot/plot_reusability.py` | 复用性绘图 | 已完成 |
| `scripts/experiment_journal/plot/plot_evolution_curve.py` | 进化曲线绘图 | 已完成（待数据） |
