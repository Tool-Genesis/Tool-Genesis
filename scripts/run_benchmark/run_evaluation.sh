#!/usr/bin/env bash
set -euo pipefail

# PLATFORM_MODELS=(
#   "BAILIAN:qwen3-4b,qwen3-8b,qwen3-14b,qwen3-30b-a3b-instruct-2507,qwen3-32b,qwen3-235b-a22b-instruct-2507",
#   "OPENROUTER:deepseek/deepseek-v3.2",
#   "OPENROUTER:moonshotai/kimi-k2",
#   "OPENROUTER:openai/gpt-4.1-mini",
#   "OPENROUTER:anthropic/claude-3.5-haiku,anthropic/claude-haiku-4.5",
#   "OPENROUTER:google/gemini-2.5-flash"
# )

# STRATEGIES=(
#   "coder_agent"
#   # "direct"
# )

# PLATFORM_MODELS=(
#   # "BAILIAN:qwen3-4b,qwen3-8b,qwen3-14b,qwen3-30b-a3b-instruct-2507,qwen3-32b,qwen3-235b-a22b-instruct-2507",
#   # "BAILIAN:deepseek-v3.2",
#   # "OPENROUTER:moonshotai/kimi-k2",
#   "OPENROUTER:openai/gpt-4.1-mini,openai/gpt-4.1,openai/gpt-5-mini",
#   "OPENROUTER:anthropic/claude-3.5-haiku,anthropic/claude-haiku-4.5",
#   "OPENROUTER:google/gemini-2.5-flash,google/gemini-3-flash-preview"
# )
PLATFORM_MODELS=(
  "BAILIAN:qwen3-32b,qwen3-30b-a3b-instruct-2507,qwen3-235b-a22b-instruct-2507",
  "OPENAI:openai/gpt-4.1-mini,openai/gpt-4.1"
  # "OPENAI:openai/gpt-o3,openai/gpt-5.1,openai/gpt-5.2",
)

STRATEGIES=(
  "coder_agent"
  "direct"
)

PRED_ROOT="${PRED_ROOT:-temp/run_benchmark_v3}"
OUT_ROOT="${OUT_ROOT:-temp/eval_results_v3}"
WORKERS="${WORKERS:-2}"
MAX_JOBS="${MAX_JOBS:-3}"
LIMIT="${LIMIT:-}"
ATTEMPTS="${ATTEMPTS:-1}"
RESET="${RESET:-}"
SKIP_L1="${SKIP_L1:-}"
SKIP_L2="${SKIP_L2:-}"
SKIP_L3="${SKIP_L3:-}"

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

if ! is_positive_int "${WORKERS}" ]; then
  echo "WORKERS must be a positive integer, got: ${WORKERS}" >&2
  exit 2
fi

if ! is_positive_int "${MAX_JOBS}" ]; then
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
        cmd=(python scripts/run_benchmark/run_evaluation.py
          --pred-path "${PRED_ROOT}"
          --out-root "${OUT_ROOT}"
          --model "${model}"
          --workers "${WORKERS}"
          --strategy "${strategy}"
          --attempts "${ATTEMPTS}"
        )
        if [ -n "${LIMIT}" ]; then
          cmd+=(--limit "${LIMIT}")
        fi
        if [ "${RESET}" = "1" ] || [ "${RESET}" = "true" ]; then
          cmd+=(--reset)
        fi
        if [ "${SKIP_L1}" = "1" ] || [ "${SKIP_L1}" = "true" ]; then
          cmd+=(--skip-l1)
        fi
        if [ "${SKIP_L2}" = "1" ] || [ "${SKIP_L2}" = "true" ]; then
          cmd+=(--skip-l2)
        fi
        if [ "${SKIP_L3}" = "1" ] || [ "${SKIP_L3}" = "true" ]; then
          cmd+=(--skip-l3)
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
