<h1 align="center">
  ⚙️ <b>Tool-Genesis：面向自进化语言智能体的任务驱动工具创建基准</b>
</h1>

<div align="center">

[![项目主页](https://img.shields.io/badge/Project-Website-1f6feb?style=flat-square)](https://tool-genesis.github.io)
[![数据集](https://img.shields.io/badge/Hugging%20Face-Dataset-ffbf00?style=flat-square)](https://huggingface.co/datasets/tool-genesis/Tool-Genesis-Benchmark)
[![模型](https://img.shields.io/badge/Hugging%20Face-Model-f59f00?style=flat-square)](https://huggingface.co/tool-genesis/Tool-Genesis-Qwen3-8B-SFT)
[![Stars][star-image]][star-url]
[![License][package-license-image]][package-license-url]

</div>

<div align="center" style="background-color: #eef7ff; padding: 18px; border-radius: 14px; border: 2px solid #4c8bf5; margin: 22px 0;">

<p style="font-size: 1.08em; margin: 8px 0; line-height: 1.55;">
  <b>Tool-Genesis</b> 是一个诊断型基准，用于评估语言智能体能否从抽象任务需求中<b>创建可复用工具</b>，
  而不仅仅是调用预定义 API。<br/>
  它覆盖了从接口推断、可执行实现到验证与下游任务效用的完整流程。
</p>

<p style="font-size: 0.98em; margin: 8px 0 0; line-height: 1.5;">
  <b>核心结论：</b>
  即使是强模型，在 one-shot 工具构建中也容易出现接口或逻辑小错误；
  这些错误会在流水线中被放大，导致下游指标明显下降。
</p>

<h3 style="color:#2f6fd6; margin: 14px 0 8px 0;">
  总体流程
</h3>

<p style="margin: 0;">
  <img src="https://tool-genesis.github.io/static/images/method.png" alt="Tool-Genesis overall pipeline" style="max-width: 900px; width: 100%; height: auto;" />
</p>

<div style="margin-top: 14px;">
  <a href="README.md" style="background-color: #424242; color: white; padding: 9px 16px; text-decoration: none; border-radius: 6px; font-weight: bold; margin: 0 6px; display: inline-block;">
    🇺🇸 English
  </a>
  <a href="https://tool-genesis.github.io" style="background-color: #2f6fd6; color: white; padding: 9px 16px; text-decoration: none; border-radius: 6px; font-weight: bold; margin: 0 6px; display: inline-block;">
    🌐 项目主页
  </a>
  <a href="#快速开始" style="background-color: #1e88e5; color: white; padding: 9px 16px; text-decoration: none; border-radius: 6px; font-weight: bold; margin: 0 6px; display: inline-block;">
    ⚡ 快速开始
  </a>
  <a href="#数据覆盖" style="background-color: #455a64; color: white; padding: 9px 16px; text-decoration: none; border-radius: 6px; font-weight: bold; margin: 0 6px; display: inline-block;">
    📊 数据与结果
  </a>
</div>

</div>

## 目录

- [项目结构](#项目结构)
- [安装](#安装)
  - [方式 1：使用 uv（推荐）](#方式-1使用-uv推荐)
  - [方式 2：使用 venv 加 pip](#方式-2使用-venv-加-pip)
  - [方式 3：使用 conda](#方式-3使用-conda)
- [环境变量](#环境变量)
  - [方式 1：使用 .env 文件（推荐）](#方式-1使用-env-文件推荐)
  - [方式 2：在终端设置环境变量](#方式-2在终端设置环境变量)
- [快速开始](#快速开始)
  - [1. 配置模型](#1-配置模型)
  - [2. 生成 MCP 服务器](#2-生成-mcp-服务器)
  - [3. 运行评测](#3-运行评测)
  - [4. 汇总结果](#4-汇总结果)
- [基准概览](#基准概览)
- [评测层级](#评测层级)
- [数据覆盖](#数据覆盖)
- [主要结果](#主要结果)
- [生成策略](#生成策略)
- [模型与后端支持](#模型与后端支持)
- [参与贡献](#参与贡献)
- [引用](#引用)
- [许可证](#许可证)
- [Star 历史](#star-历史)

## 项目结构

```text
Tool-Genesis/
├── src/
│   ├── core/                 # Agent 框架、沙箱、工具集、模型接口
│   ├── apps/                 # MCP server 工厂、文件系统应用、测试客户端
│   ├── env_generation/       # 工具生成策略
│   ├── env_evalution/        # 四级评测流程
│   └── utils/                # 通用工具函数
├── scripts/
│   ├── run_benchmark/        # 生成与评测脚本
│   ├── build_benchmark/      # 数据集构建流程
│   ├── experiment_journal/   # 期刊扩展实验
│   └── plot/                 # 绘图脚本
├── data/
│   └── tool_genesis_v3.json  # 主数据集
├── assets/                   # README 图片资源
├── requirements.txt
├── .env.template
└── TODO.md
```

## 安装

### 方式 1：使用 uv（推荐）

```bash
# 1) 创建虚拟环境
uv venv .venv --python 3.10

# 2) 激活
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

# 3) 安装依赖
uv pip install -r requirements.txt
```

### 方式 2：使用 venv 加 pip

```bash
# 1) 创建虚拟环境
python3.10 -m venv .venv

# 2) 激活
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

# 3) 安装依赖
pip install -r requirements.txt
```

### 方式 3：使用 conda

```bash
# 1) 创建环境
conda create -n toolgenesis python=3.10 -y
conda activate toolgenesis

# 2) 安装依赖
pip install -r requirements.txt
```

## 环境变量

项目通过环境变量读取各类模型后端与密钥配置。

### 方式 1：使用 .env 文件（推荐）

1. 复制模板：

```bash
cp .env.template .env
```

2. 在 `.env` 中填写最小可用配置（通常至少需要 OpenAI 兼容密钥和 base URL）：

```env
# 最小配置
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1

# 可选后端
OPENROUTER_API_KEY=...
DEEPSEEK_API_KEY=...
BAILIAN_API_KEY=...
GEMINI_API_KEY=...
```

### 方式 2：在终端设置环境变量

- macOS/Linux (Bash/Zsh)

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

- Windows (PowerShell)

```powershell
$env:OPENAI_API_KEY="your-key"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
```

## 快速开始

最小复现实验流程：生成工具服务器 -> 执行 L1-L4 评测 -> 汇总指标。

### 1. 配置模型

编辑 `scripts/run_benchmark/generate_mcp.sh`，设置模型列表：

```bash
PLATFORM_MODELS=(
  "OPENAI:openai/gpt-4.1-mini,openai/gpt-4.1"
  "OPENAI:anthropic/claude-sonnet-4"
)
```

### 2. 生成 MCP 服务器

```bash
bash scripts/run_benchmark/generate_mcp.sh
```

### 3. 运行评测

```bash
bash scripts/run_benchmark/run_evaluation.sh
```

### 4. 汇总结果

```bash
python scripts/run_benchmark/summarize_results.py --path temp/eval_results_v3
```

## 基准概览

| 统计项 | 数值 |
|---|---:|
| MCP servers | 86 |
| Total tools | 508 |
| Domain classes | 24 |
| Label classes | 18 |
| Unit tests | 9441 |
| Total tasks | 2150 |
| Average task length | 53 |
| Average step length | 6 |
| Average tool-using length | 3 |

## 评测层级

| 层级 | 评测内容 | 指标 |
|---|---|---|
| L1 | 协议与启动正确性 | Compliance, Exec. |
| L2 | 工具语义正确性 | Schema-F1, UT_soft |
| L3 | 负例与边界鲁棒性 | UT_hard |
| L4 | 下游任务效用 | Success Rate |

## 数据覆盖

请将数据概览图放到：

- `assets/figure_dataset_overview.png`

![Tool-Genesis dataset overview](assets/figure_dataset_overview.png)

## 主要结果

请将主结果图/表放到：

- `assets/table_main_results.png`

![Tool-Genesis main results](assets/table_main_results.png)

## 生成策略

| 策略 | 说明 |
|---|---|
| Direct | 单次生成 schema 与代码 |
| Coder-Agent | 迭代式工具辅助生成与修复 |

## 模型与后端支持

框架通过统一 LLM 层支持多种后端：

- OpenAI（及其兼容端点）
- Bailian
- OpenRouter
- DeepSeek
- vLLM（自部署）

## 参与贡献

欢迎通过 Issue / Pull Request 提交改进，包括基准扩展、评测修复和文档完善。

## 引用

如果本项目对你的研究有帮助，请引用：

```bibtex
@misc{tool_genesis_2025,
  title={Tool-Genesis: A Task-Driven Tool Creation Benchmark for Self-Evolving Language Agent},
  author={Xia, Bowei and Hu, Mengkang and Wang, Shijian and Jin, Jiarui and Jiao, Wenxiang and Lu, Yuan and Li, Kexin and Luo, Ping},
  year={2025},
  note={Project page: https://tool-genesis.github.io}
}
```

## 许可证

详见 [LICENSE](LICENSE)。

## Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=Tool-Genesis/Tool-Genesis&type=Date)](https://star-history.com/#Tool-Genesis/Tool-Genesis&Date)

[star-image]: https://img.shields.io/github/stars/Tool-Genesis/Tool-Genesis?label=stars&logo=github&color=brightgreen
[star-url]: https://github.com/Tool-Genesis/Tool-Genesis/stargazers
[package-license-image]: https://img.shields.io/badge/License-Apache_2.0-blue.svg
[package-license-url]: https://github.com/Tool-Genesis/Tool-Genesis/blob/main/LICENSE
