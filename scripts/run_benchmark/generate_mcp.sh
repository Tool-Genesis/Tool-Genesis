#!/usr/bin/env bash
set -euo pipefail

# PLATFORM_MODELS=(
#   "BAILIAN:qwen3-4b,qwen3-8b,qwen3-14b,qwen3-30b-a3b-instruct-2507,qwen3-32b,qwen3-235b-a22b-instruct-2507",
#   "OPENAI:deepseek/deepseek-v3.2",
#   "OPENAI:moonshotai/kimi-k2",
#   "OPENAI:openai/gpt-4.1-mini,openai/gpt-4.1,openai/gpt-o3,openai/gpt-5.1,openai/gpt-5.2",
#   "OPENAI:anthropic/claude-sonnet-4",
#   "OPENAI:google/gemini-3-flash-preview"
# )
PLATFORM_MODELS=(
  # "BAILIAN:qwen3-4b,qwen3-8b,qwen3-14b,qwen3-30b-a3b-instruct-2507,qwen3-32b,qwen3-235b-a22b-instruct-2507",
  # "OPENAI:deepseek/deepseek-v3.2",
  # "OPENAI:moonshotai/kimi-k2",
  "OPENAI:openai/gpt-5.1",
  # "OPENAI:anthropic/claude-sonnet-4",
  # "OPENAI:google/gemini-3-flash-preview"
)

STRATEGIES=(
  "coder_agent",
  "direct"
)

DATA_PATH="${DATA_PATH:-data/tool_genesis_v3.json}"
OUT_ROOT="${OUT_ROOT:-temp/run_benchmark_v3}"
WORKERS="${WORKERS:-2}"
MAX_JOBS="${MAX_JOBS:-2}"
LIMIT="${LIMIT:-}"

LOG_DIR="${LOG_DIR:-${OUT_ROOT}/logs/${STRATEGIES}}"
PID_FILE="${LOG_DIR}/pids"

ACTION="${1:-start}"

mkdir -p "${LOG_DIR}"

pids=()
names=()

stop_jobs() {
  if [ ! -f "${PID_FILE}" ]; then
    echo "No pid file found at ${PID_FILE}, nothing to stop"
    return 0
  fi

  while IFS= read -r pid; do
    if [ -z "${pid}" ]; then
      continue
    fi
    if kill -0 "${pid}" 2>/dev/null; then
      echo "Stopping pid=${pid} (and its process group)"
      kill -TERM -"${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    fi
  done < "${PID_FILE}"

  rm -f "${PID_FILE}"
}

is_positive_int() {
  [[ "${1:-}" =~ ^[0-9]+$ ]] && [ "${1}" -ge 1 ]
}

if ! is_positive_int "${WORKERS}"; then
  echo "WORKERS must be a positive integer, got: ${WORKERS}" >&2
  exit 2
fi

if ! is_positive_int "${MAX_JOBS}"; then
  echo "MAX_JOBS must be a positive integer, got: ${MAX_JOBS}" >&2
  exit 2
fi

case "${ACTION}" in
  start)
    ;;
  stop)
    stop_jobs
    exit 0
    ;;
  *)
    echo "Usage: $0 [start|stop]" >&2
    exit 2
    ;;
esac

> "${PID_FILE}"

running_count() {
  local count=0
  local pid
  for pid in "${pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      count=$((count + 1))
    fi
  done
  printf "%s" "${count}"
}

sanitize_component() {
  printf "%s" "${1:-}" | sed -E \
    -e 's#[/\\]#_#g' \
    -e 's/[[:space:]]+/_/g' \
    -e 's/[^A-Za-z0-9_.-]+/_/g' \
    -e 's/_+/_/g' \
    -e 's/^_+//g' \
    -e 's/_+$//g'
}

wait_for_slot() {
  while [ "$(running_count)" -ge "${MAX_JOBS}" ]; do
    sleep 1
  done
}

for entry in "${PLATFORM_MODELS[@]}"; do
  platform="${entry%%:*}"
  models_csv="${entry#*:}"

  IFS=',' read -r -a MODELS <<< "${models_csv}"

  for model in "${MODELS[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
      wait_for_slot
      ts="$(date +%Y%m%d_%H%M%S)"
      safe_platform="$(sanitize_component "${platform}")"
      safe_strategy="$(sanitize_component "${strategy}")"
      safe_model="$(sanitize_component "${model}")"
      log_path="${LOG_DIR}/${safe_platform}__${safe_strategy}__${safe_model}__${ts}.log"
      name="${platform}__${strategy}__${model}"
      (
        cmd=(python scripts/run_benchmark/generate_mcp_from_task.py
          --data-path "${DATA_PATH}"
          --out-root "${OUT_ROOT}"
          --model "${model}"
          --workers "${WORKERS}"
          --strategy "${strategy}"
        )
        if [ -n "${LIMIT}" ]; then
          cmd+=(--limit "${LIMIT}")
        fi
        if [ -n "${platform}" ]; then
          cmd+=(--platform "${platform}")
        fi
        "${cmd[@]}"
      ) >"${log_path}" 2>&1 &
      pid=$!
      pids+=("${pid}")
      names+=("${name}")
      printf "🚀 Started %-60s pid=%s log=\"\"\n" "${name}" "${pid}"
      printf "   📄 log=%s\n" "${log_path}"
      echo "${pid}" >> "${PID_FILE}"
      sleep 0.2
    done
  done
done

failed=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  name="${names[$i]}"
  if wait "${pid}"; then
    printf "✅ Done    %-60s pid=%s\n" "${name}" "${pid}"
  else
    printf "❌ Failed  %-60s pid=%s\n" "${name}" "${pid}"
    failed=1
  fi
done

exit "${failed}"
