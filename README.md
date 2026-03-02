# Tool-Genesis

**Can LLM Agents Autonomously Create Tools? A Benchmark for Tool Genesis**

Tool-Genesis is a benchmark framework for evaluating the ability of LLM agents to autonomously generate MCP (Model Context Protocol) tool servers from natural language descriptions.

## Overview

Given a scenario description, the agent must generate a complete, runnable MCP server — including tool schemas and implementation code. The generated server is then evaluated across four levels:

| Level | What it tests | Metrics |
|-------|--------------|---------|
| **L1: Protocol Compliance** | JSON format validity, server launch success | Compliance rate, Launch rate |
| **L2: Semantic Correctness** | Tool schema matching, unit test pass rate | Schema F1, UT pass rate |
| **L3: Capability Boundary** | No unauthorized capabilities, safe extra tools | Boundary violation rate |
| **L4: Task-Level** | Proxy agent solves downstream tasks with generated tools | Success rate (soft/hard) |

The benchmark covers **86 MCP servers** across diverse domains (finance, education, healthcare, etc.) with **1,720 tasks**.

## Project Structure

```
Tool-Genesis/
├── src/
│   ├── core/              # Agent framework, sandbox, toolkits, models
│   ├── apps/              # MCP server factory, file system, test client
│   ├── env_generation/    # Tool generation strategies (direct, coder_agent)
│   ├── env_evalution/     # Four-level evaluation pipeline (L1-L4)
│   └── utils/             # LLM API wrapper, token counting
├── scripts/
│   ├── run_benchmark/     # Generation & evaluation scripts
│   ├── build_benchmark/   # Dataset construction pipeline
│   ├── experiment_journal/# Journal extension experiments (A-G)
│   └── plot/              # Visualization scripts
├── data/
│   └── tool_genesis_v3.json  # Main benchmark dataset
├── requirements.txt
├── .env.template
└── TODO.md
```

## Quick Start

### 1. Environment Setup

```bash
conda create -n toolgenesis python=3.10 -y
conda activate toolgenesis
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.template .env
# Edit .env — at minimum set OPENAI_API_KEY and OPENAI_BASE_URL
```

### 3. Configure Models

Edit `scripts/run_benchmark/generate_mcp.sh` and set the `PLATFORM_MODELS` array with your available models:

```bash
PLATFORM_MODELS=(
  "OPENAI:openai/gpt-4.1-mini,openai/gpt-4.1"
  "OPENAI:anthropic/claude-sonnet-4"
)
```

### 4. Generate Environments

```bash
bash scripts/run_benchmark/generate_mcp.sh
```

### 5. Run Evaluation

```bash
bash scripts/run_benchmark/run_evaluation.sh
```

### 6. Summarize Results

```bash
python scripts/run_benchmark/summarize_results.py --path temp/eval_results_v3
```

## Generation Strategies

| Strategy | Description |
|----------|-------------|
| **Direct** | Single LLM call generates schema + code |
| **Coder-Agent** | Multi-turn agent with sandbox access, iteratively writes and tests code |

## LLM Platform Support

The framework supports multiple LLM providers via a unified `call_llm()` interface:

- **OpenAI** (and OpenAI-compatible endpoints)
- **Bailian** (Alibaba Cloud)
- **OpenRouter**
- **DeepSeek**
- **vLLM** (self-hosted)

Configure in `.env` and pass `platform` parameter to `call_llm()`.

## Journal Extension Experiments

Additional experiments for the journal version (see `scripts/experiment_journal/` and `TODO.md`):

| Experiment | Description | Status |
|-----------|-------------|--------|
| **A: Tool Reusability** | Can generated tools generalize to unseen tasks? | Needs fix |
| **B: Multi-Round Evolution** | Can agents iteratively improve tools via feedback? | Needs fix |
| **C: Error Taxonomy** | What types of failures occur across L1-L4? | Needs fix |
| **D: Toolset Completeness** | Are all required tools generated? | Done |
| **E: Oracle Ablation** | How much error comes from schema vs code? | Framework ready |

See `TODO.md` for detailed status and pending work.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
