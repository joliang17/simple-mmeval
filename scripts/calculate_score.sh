#!/usr/bin/env bash

set -euo pipefail

# Score every real run under work_dirs and refresh eval_summary.csv.
#
# Common overrides:
#   SCORE_OUTPUT_NAME=score_rerun.json bash scripts/calculate_score.sh
#   ONLY_RUNS="qwen25vl3b_baseline/base" bash scripts/calculate_score.sh
#   DRY_RUN=true bash scripts/calculate_score.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="./:${PYTHONPATH:-}"

WORK_DIRS="${WORK_DIRS:-${ROOT_DIR}/work_dirs}"
SUMMARY_CSV="${SUMMARY_CSV:-${ROOT_DIR}/eval_summary.csv}"
SCORE_RESULT_GLOB="${SCORE_RESULT_GLOB:-*/result.json}"
SCORE_OUTPUT_NAME="${SCORE_OUTPUT_NAME:-score.json}"
PARALLEL_PER_TASK="${PARALLEL_PER_TASK:-8}"
MATCHING_ORDER="${MATCHING_ORDER:-exact,template}"
SCORE_RESUME="${SCORE_RESUME:-true}"
SCORE_PROGRESS_BAR="${SCORE_PROGRESS_BAR:-false}"
SCORE_SAVE_FREQ="${SCORE_SAVE_FREQ:-20}"
INCLUDE_EXAMPLES="${INCLUDE_EXAMPLES:-false}"
DRY_RUN="${DRY_RUN:-false}"
ONLY_RUNS="${ONLY_RUNS:-}"

if [[ ! -d "${WORK_DIRS}" ]]; then
  echo "ERROR: WORK_DIRS does not exist: ${WORK_DIRS}" >&2
  exit 1
fi

is_selected_run() {
  local run_name="$1"

  if [[ -z "${ONLY_RUNS}" ]]; then
    return 0
  fi

  local selected
  for selected in ${ONLY_RUNS}; do
    if [[ "${run_name}" == "${selected}" ]]; then
      return 0
    fi
  done

  return 1
}

discover_run_dirs() {
  local find_args=("${WORK_DIRS}")

  if [[ "${INCLUDE_EXAMPLES}" != "true" ]]; then
    find_args+=("(" "-path" "${WORK_DIRS}/examples" "-prune" ")" "-o")
  fi

  find "${find_args[@]}" -type f -name result.json -print \
    | while IFS= read -r result_path; do
        dirname "$(dirname "${result_path}")"
      done \
    | sort -u \
    | while IFS= read -r run_dir; do
        local run_name
        run_name="$(realpath --relative-to="${WORK_DIRS}" "${run_dir}")"
        if is_selected_run "${run_name}"; then
          printf '%s\n' "${run_dir}"
        fi
      done
}

mapfile -t RUN_DIRS < <(discover_run_dirs)

if [[ "${#RUN_DIRS[@]}" -eq 0 ]]; then
  echo "ERROR: no run directories with result.json found under ${WORK_DIRS}" >&2
  if [[ -n "${ONLY_RUNS}" ]]; then
    echo "       ONLY_RUNS=${ONLY_RUNS}" >&2
  fi
  exit 1
fi

echo "Scoring config:"
echo "  ROOT_DIR=${ROOT_DIR}"
echo "  WORK_DIRS=${WORK_DIRS}"
echo "  SUMMARY_CSV=${SUMMARY_CSV}"
echo "  SCORE_RESULT_GLOB=${SCORE_RESULT_GLOB}"
echo "  SCORE_OUTPUT_NAME=${SCORE_OUTPUT_NAME}"
echo "  PARALLEL_PER_TASK=${PARALLEL_PER_TASK}"
echo "  MATCHING_ORDER=${MATCHING_ORDER}"
echo "  SCORE_RESUME=${SCORE_RESUME}"
echo "  SCORE_PROGRESS_BAR=${SCORE_PROGRESS_BAR}"
echo "  SCORE_SAVE_FREQ=${SCORE_SAVE_FREQ}"
echo "  INCLUDE_EXAMPLES=${INCLUDE_EXAMPLES}"
echo "  ONLY_RUNS=${ONLY_RUNS:-<all real runs>}"
echo "  DRY_RUN=${DRY_RUN}"
echo

echo "Discovered ${#RUN_DIRS[@]} run directory(s):"
for run_dir in "${RUN_DIRS[@]}"; do
  echo "  - $(realpath --relative-to="${WORK_DIRS}" "${run_dir}")"
done
echo

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" != "true" ]]; then
    "$@"
  fi
}

for run_dir in "${RUN_DIRS[@]}"; do
  cmd=(
    python mmeval/score.py
    --out-dir "${run_dir}"
    --score-result-glob "${SCORE_RESULT_GLOB}"
    --score-output-name "${SCORE_OUTPUT_NAME}"
    --parallel-per-task "${PARALLEL_PER_TASK}"
    --matching-order "${MATCHING_ORDER}"
    --score-save-freq "${SCORE_SAVE_FREQ}"
  )

  if [[ "${SCORE_RESUME}" == "false" ]]; then
    cmd+=(--no-score-resume)
  fi

  if [[ "${SCORE_PROGRESS_BAR}" == "false" ]]; then
    cmd+=(--no-score-progress-bar)
  fi

  run_cmd "${cmd[@]}"
done

summary_cmd=(
  python scripts/update_eval_summary.py
  --root "${WORK_DIRS}"
  --out "${SUMMARY_CSV}"
  --score-output-name "${SCORE_OUTPUT_NAME}"
  --strict-scores
)

if [[ "${INCLUDE_EXAMPLES}" == "true" ]]; then
  summary_cmd+=(--include-examples)
fi

if [[ -n "${ONLY_RUNS}" ]]; then
  summary_cmd+=(--only)
  for selected in ${ONLY_RUNS}; do
    summary_cmd+=("${selected}")
  done
fi

run_cmd "${summary_cmd[@]}"
