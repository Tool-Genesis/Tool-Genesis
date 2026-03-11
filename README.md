<h1 align="center">
  ⚙️ <b>Tool-Genesis: A Task-Driven Tool Creation Benchmark for Self-Evolving Language Agent</b>
</h1>

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Website-1f6feb?style=flat-square)](https://tool-genesis.github.io)
[![ArXiv](https://img.shields.io/badge/arXiv-2603.05578-b31b1b?style=flat-square)](https://arxiv.org/abs/2603.05578)
[![PDF](https://img.shields.io/badge/PDF-Download-ec407a?style=flat-square)](https://arxiv.org/pdf/2603.05578)
[![Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-ffbf00?style=flat-square)](https://huggingface.co/datasets/tool-genesis/Tool-Genesis-Benchmark)
[![Model](https://img.shields.io/badge/Hugging%20Face-Model-f59f00?style=flat-square)](https://huggingface.co/tool-genesis/Tool-Genesis-Qwen3-8B-SFT)
[![Stars][star-image]][star-url]
[![License][package-license-image]][package-license-url]

</div>

<div align="center" style="background-color: #eef7ff; padding: 18px; border-radius: 14px; border: 2px solid #4c8bf5; margin: 22px 0;">

<p style="font-size: 1.08em; margin: 8px 0; line-height: 1.55;">
  <b>Tool-Genesis</b> is a diagnostic benchmark for evaluating whether language agents can <b>create reusable tools</b>
  from abstract task requirements, instead of only calling pre-defined APIs.<br/>
  It measures the full pipeline from interface inference and executable implementation to validation and downstream utility.
</p>

<p style="font-size: 0.98em; margin: 8px 0 0; line-height: 1.5;">
  <b>Key points:</b>
  one-shot tool construction remains difficult even for strong models;
  small interface or logic errors in early stages are amplified and cause large downstream performance drops.
</p>

<h3 style="color:#2f6fd6; margin: 14px 0 8px 0;">
  Overall Pipeline
</h3>

<p style="margin: 0;">
  <img src="https://tool-genesis.github.io/static/images/method.png" alt="Overall pipeline of Tool-Genesis" style="max-width: 900px; width: 100%; height: auto;" />
</p>

<div style="margin-top: 14px;">
  <a href="README_zh.md" style="background-color: #424242; color: white; padding: 9px 16px; text-decoration: none; border-radius: 6px; font-weight: bold; margin: 0 6px; display: inline-block;">
    🇨🇳 中文说明
  </a>
  <a href="https://tool-genesis.github.io" style="background-color: #2f6fd6; color: white; padding: 9px 16px; text-decoration: none; border-radius: 6px; font-weight: bold; margin: 0 6px; display: inline-block;">
    🌐 Project Page
  </a>
  <a href="#quick-start" style="background-color: #1e88e5; color: white; padding: 9px 16px; text-decoration: none; border-radius: 6px; font-weight: bold; margin: 0 6px; display: inline-block;">
    ⚡ Quick Start
  </a>
  <a href="#data-coverage" style="background-color: #455a64; color: white; padding: 9px 16px; text-decoration: none; border-radius: 6px; font-weight: bold; margin: 0 6px; display: inline-block;">
    📊 Data And Results
  </a>
</div>

</div>

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Option 1: Use uv (recommended)](#option-1-use-uv-recommended)
  - [Option 2: Use venv plus pip](#option-2-use-venv-plus-pip)
  - [Option 3: Use conda](#option-3-use-conda)
- [Environment Variables](#environment-variables)
  - [Option 1: Use a .env file (recommended)](#option-1-use-a-env-file-recommended)
  - [Option 2: Set variables in terminal](#option-2-set-variables-in-terminal)
- [Quick Start](#quick-start)
  - [1. Configure models](#1-configure-models)
  - [2. Generate MCP servers](#2-generate-mcp-servers)
  - [3. Run evaluation](#3-run-evaluation)
  - [4. Summarize results](#4-summarize-results)
- [Benchmark At A Glance](#benchmark-at-a-glance)
- [Evaluation Levels](#evaluation-levels)
- [Data Coverage](#data-coverage)
- [Main Results](#main-results)
- [Generation Strategies](#generation-strategies)
- [Model And Backends Support](#model-and-backends-support)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Star History](#star-history)

## Project Structure

```text
Tool-Genesis/
├── src/
│   ├── core/                 # Agent framework, sandbox, toolkits, model interfaces
│   ├── apps/                 # MCP server factory, file system apps, test client
│   ├── env_generation/       # Tool generation strategies
│   ├── env_evalution/        # Four-level evaluation pipeline
│   └── utils/                # Shared utilities
├── scripts/
│   ├── run_benchmark/        # Generation and evaluation scripts
│   ├── build_benchmark/      # Dataset construction pipeline
│   ├── experiment_journal/   # Journal extension experiments
│   └── plot/                 # Plotting scripts
├── data/
│   └── tool_genesis_v3.json  # Main benchmark dataset
├── assets/                   # Figures used by README (add your images here)
├── requirements.txt
├── .env.template
└── TODO.md
```

## Installation

### Option 1: Use uv (recommended)

```bash
# 1) Create a virtual environment
uv venv .venv --python 3.10

# 2) Activate
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

# 3) Install dependencies
uv pip install -r requirements.txt
```

### Option 2: Use venv plus pip

```bash
# 1) Create a virtual environment
python3.10 -m venv .venv

# 2) Activate
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install -r requirements.txt
```

### Option 3: Use conda

```bash
# 1) Create environment
conda create -n toolgenesis python=3.10 -y
conda activate toolgenesis

# 2) Install dependencies
pip install -r requirements.txt
```

## Environment Variables

This project reads provider credentials and endpoints from environment variables.

### Option 1: Use a .env file (recommended)

1. Copy the template:

```bash
cp .env.template .env
```

2. Fill required keys in `.env` (minimum run usually needs OpenAI-compatible key plus base URL):

```env
# Minimal
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1

# Optional providers
OPENROUTER_API_KEY=...
DEEPSEEK_API_KEY=...
BAILIAN_API_KEY=...
GEMINI_API_KEY=...
```

### Option 2: Set variables in terminal

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

## Quick Start

Minimal reproduction flow: generate tool servers -> evaluate L1-L4 -> summarize metrics.

### 1. Configure models

Edit `scripts/run_benchmark/generate_mcp.sh` and set your model list:

```bash
PLATFORM_MODELS=(
  "OPENAI:openai/gpt-4.1-mini,openai/gpt-4.1"
  "OPENAI:anthropic/claude-sonnet-4"
)
```

### 2. Generate MCP servers

```bash
bash scripts/run_benchmark/generate_mcp.sh
```

### 3. Run evaluation

```bash
bash scripts/run_benchmark/run_evaluation.sh
```

### 4. Summarize results

```bash
python scripts/run_benchmark/summarize_results.py --path temp/eval_results_v3
```

## Benchmark At A Glance

| Statistic | Number |
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

## Evaluation Levels

| Level | What it tests | Metrics |
|---|---|---|
| L1 | Protocol and launch correctness | Compliance, Exec. |
| L2 | Semantic correctness of generated tools | Schema-F1, UT_soft |
| L3 | Boundary and negative-case robustness | UT_hard |
| L4 | Downstream task utility | Success Rate |

## Data Coverage

![Tool-Genesis dataset overview](assets/figure_dataset_overview.png)

*Figure: Overview statistics of Tool-Genesis. Left: domain coverage of MCP servers across 24 domain classes. Right: dataset scale and task/trajectory statistics.*

## Main Results

![Tool-Genesis main results](assets/table_main_results.png)

*Table: Main results under two evaluation paradigms (Direct and Code-Agent). Metrics include L1 compliance/execution, L2 schema fidelity, L3 unit-test correctness, and L4 downstream utility.*

## Generation Strategies

| Strategy | Description |
|---|---|
| Direct | Single-pass generation of schema plus code |
| Coder-Agent | Iterative generation with tool-assisted coding and repair |

## Model And Backends Support

The framework supports multiple backends through a unified LLM layer, including:

- OpenAI (and compatible endpoints)
- Bailian
- OpenRouter
- DeepSeek
- vLLM (self-hosted)

## Contributing

Issues and pull requests are welcome for benchmark improvements, evaluation fixes, and documentation updates.

## Citation

If this project is useful in your research, please cite:

```bibtex
@misc{tool_genesis_2025,
  title={Tool-Genesis: A Task-Driven Tool Creation Benchmark for Self-Evolving Language Agent},
  author={Xia, Bowei and Hu, Mengkang and Wang, Shijian and Jin, Jiarui and Jiao, Wenxiang and Lu, Yuan and Li, Kexin and Luo, Ping},
  year={2025},
  note={Project page: https://tool-genesis.github.io}
}
```

Replace with the official publication entry when available.

## License

See [LICENSE](LICENSE) for details.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Tool-Genesis/Tool-Genesis&type=Date)](https://star-history.com/#Tool-Genesis/Tool-Genesis&Date)

[star-image]: https://img.shields.io/github/stars/Tool-Genesis/Tool-Genesis?label=stars&logo=github&color=brightgreen
[star-url]: https://github.com/Tool-Genesis/Tool-Genesis/stargazers
[package-license-image]: https://img.shields.io/badge/License-Apache_2.0-blue.svg
[package-license-url]: https://github.com/Tool-Genesis/Tool-Genesis/blob/main/LICENSE
