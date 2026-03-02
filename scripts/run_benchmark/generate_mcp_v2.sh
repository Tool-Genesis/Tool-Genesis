#!/usr/bin/env bash
set -euo pipefail

MODEL_API_KEYS=(
  "openai/gpt-5.1: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|https://openrouter.ai/api/v1|2024-12-01-preview"
  "deepseek/deepseek-v3.2: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|https://openrouter.ai/api/v1|2024-12-01-preview"
  "moonshotai/kimi-k2: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|https://openrouter.ai/api/v1|2024-12-01-preview"
  # "openai/gpt-5.1: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|https://openrouter.ai/api/v1|2024-12-01-preview"
  # "openai/gpt-5.2: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|https://openrouter.ai/api/v1|2024-12-01-preview"
  # "anthropic/claude-sonnet-4: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|https://openrouter.ai/api/v1|2024-12-01-preview"
  # "google/gemini-3-flash-preview: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|https://openrouter.ai/api/v1|2024-12-01-preview"
)

STRATEGIES=(
  "direct"
  "coder_agent"
)

PLATFORM="${PLATFORM:-OPENAI}"
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

for entry in "${MODEL_API_KEYS[@]:-}"; do
  raw_model="${entry%%:*}"
  rest="${entry#*:}"

  if [ -z "${raw_model}" ]; then
    continue
  fi

  api_key=""
  base_url=""
  extra=""
  IFS='|' read -r api_key base_url extra <<< "${rest}"

  model="${raw_model}"
  if [[ "${model}" != */* ]]; then
    model="${model}"
  fi

  for strategy in "${STRATEGIES[@]}"; do
    wait_for_slot
    ts="$(date +%Y%m%d_%H%M%S)"
    safe_platform="$(sanitize_component "${PLATFORM}")"
    safe_strategy="$(sanitize_component "${strategy}")"
    safe_model="$(sanitize_component "${model}")"
    log_path="${LOG_DIR}/${safe_platform}__${safe_strategy}__${safe_model}__${ts}.log"
    name="${PLATFORM}__${strategy}__${model}"
    (
      if [ -n "${api_key}" ]; then
        export OPENAI_API_KEY="${api_key}"
      fi
      if [ -n "${base_url}" ]; then
        export OPENAI_BASE_URL="${base_url}"
      fi
      if [ -n "${extra}" ]; then
        export OPENAI_API_VERSION="${extra}"
      fi
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
      if [ -n "${PLATFORM}" ]; then
        cmd+=(--platform "${PLATFORM}")
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
