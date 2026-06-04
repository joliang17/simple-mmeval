#!/usr/bin/env bash

set -u

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <parallel_datasets>"
  echo "Example: $0 4"
  exit 2
fi

PARALLEL_DATASETS="$1"
if ! [[ "${PARALLEL_DATASETS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] parallel_datasets must be a positive integer, got: ${PARALLEL_DATASETS}"
  exit 2
fi

# Run Qwen on VLMEvalKit datasets that are compatible with simple-mmeval TSVDataset flow.
# You can override these with environment variables before running:
#   MODEL_NAME_OR_PATH, OUT_DIR, GPU_PER_PARALLEL, PARALLEL_PER_TASK
export PYTHONPATH=./:${PYTHONPATH:-}
export DATASET_DIR=${DATASET_DIR:-./datasets}

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/tiger/yijiangli/project/OPSD/work_dirs/grpo_vlm/OpenMMReasoner-RL-74K_multimodal-open-r1-8k-verified/multimodal-open-r1-8k-verified/qwen25vl3b_2epochs_lr1e5_bs4_acc2_numgen4-merged}"
OUT_DIR="${OUT_DIR:-work_dirs/qwen25vl3b_2epochs_lr1e5_bs4_acc2_numgen4-merged}"
GPU_PER_PARALLEL="${GPU_PER_PARALLEL:-1}"
PARALLEL_PER_TASK="${PARALLEL_PER_TASK:-8}"

DATASETS=(
  # Standard ImageMCQDataset
  # "3DSRBench"
  # "A-Bench_TEST"
  # "A-Bench_VAL"
  # "A-OKVQA"
  # "AI2D_TEST"
  # "AI2D_TEST_NO_MASK"
  # "AesBench_TEST"
  # "AesBench_VAL"
  "BLINK"
  # "CMMU_MCQ"
  # "HRBench4K"
  # "HRBench8K"
  # "MedXpertQA_MM_test"
  # "MicroVQA"
  # "MLLMGuard_DS"
  # "MMBench_dev_ar"
  # "MMBench_dev_cn"
  "MMBench_dev_en"
  # "MMBench_dev_en_test"
  # "MMBench_dev_pt"
  # "MMBench_dev_ru"
  # "MMBench_dev_tr"
  # "MMMB_ar"
  # "MMMB_cn"
  # "MMMB_en"
  # "MMMB_pt"
  # "MMMB_ru"
  # "MMMB_tr"
  # "MMSci_DEV_MCQ"
  "MMStar"
  # "MMT-Bench_ALL"
  # "MMT-Bench_VAL"
  # "MMVP"
  # "PathMMU_TEST"
  # "PathMMU_VAL"
  # "Q-Bench1_TEST"
  # "Q-Bench1_VAL"
  # "R-Bench-Dis"
  # "R-Bench-Ref"
  "RealWorldQA"
  "SEEDBench2"
  # "SEEDBench2_Plus"
  "SEEDBench_IMG"
  # "ScienceQA_TEST"
  "ScienceQA_VAL"
  # "TaskMeAnything_v1_imageqa_random"
  # "VisOnlyQA-VLMEvalKit"
  "VStarBench"
  # "WorldMedQA-V"

  # Multipart / Concat (supported in simple-mmeval evalkit loader)
  # "MicroBench"
  # "OmniMedVQA"
  # "MMMB"
  # "MTL_MMBench_DEV"
  # Standard ImageYORNDataset (POPE only)
  # "POPE"

  # VCRDataset
  # "VCR_EN_EASY_ALL"
  # "VCR_EN_HARD_ALL"
  # "VCR_ZH_EASY_ALL"
  # "VCR_ZH_HARD_ALL"
)

echo "[INFO] model: ${MODEL_NAME_OR_PATH}"
echo "[INFO] out_dir: ${OUT_DIR}"
echo "[INFO] gpu_per_parallel: ${GPU_PER_PARALLEL}"
echo "[INFO] parallel_per_task: ${PARALLEL_PER_TASK}"
echo "[INFO] parallel_datasets: ${PARALLEL_DATASETS}"
echo "[INFO] dataset_dir: ${DATASET_DIR}"
echo "[INFO] total datasets: ${#DATASETS[@]}"

mkdir -p "${OUT_DIR}"

idx=0
total=${#DATASETS[@]}
status_dir="${OUT_DIR}/.dataset_status.$$"
mkdir -p "${status_dir}"

for dataset in "${DATASETS[@]}"; do
  idx=$((idx + 1))
  (
    start_ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo "[${start_ts}] [${idx}/${total}] START evalkit@${dataset}"

    python mmeval/run.py \
      --model_name_or_path "${MODEL_NAME_OR_PATH}" \
      --dataset "evalkit@${dataset}" \
      --out_dir "${OUT_DIR}/${dataset}" \
      --no_conda \
      --gpu_per_parallel "${GPU_PER_PARALLEL}" \
      --parallel_per_task "${PARALLEL_PER_TASK}"
    exit_code=$?

    end_ts="$(date '+%Y-%m-%d %H:%M:%S')"
    if [ "${exit_code}" -eq 0 ]; then
      echo "[${end_ts}] [${idx}/${total}] DONE evalkit@${dataset}"
      echo 0 > "${status_dir}/${dataset}.status"
    else
      echo "[${end_ts}] [${idx}/${total}] FAIL evalkit@${dataset} (exit=${exit_code})"
      echo "${exit_code}" > "${status_dir}/${dataset}.status"
    fi
  ) &

  while [ "$(jobs -rp | wc -l)" -ge "${PARALLEL_DATASETS}" ]; do
    wait -n
  done
done

wait

failed=()
for dataset in "${DATASETS[@]}"; do
  status_file="${status_dir}/${dataset}.status"
  if [ ! -f "${status_file}" ]; then
    failed+=("${dataset}")
    continue
  fi
  status="$(< "${status_file}")"
  if [ "${status}" != "0" ]; then
    failed+=("${dataset}")
  fi
done

rm -rf "${status_dir}"

echo "============================================================"
echo "[SUMMARY] total=${total}, failed=${#failed[@]}"
if [ "${#failed[@]}" -gt 0 ]; then
  echo "[SUMMARY] failed datasets:"
  for dataset in "${failed[@]}"; do
    echo "  - ${dataset}"
  done
  exit 1
fi

echo "[SUMMARY] all datasets finished successfully."
