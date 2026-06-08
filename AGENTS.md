# Repository Guidelines

## Project Structure & Module Organization

Simple-MMEval is a Python framework for multimodal model evaluation. Core code lives in `mmeval/`: `run.py` launches jobs, `registry.py` maps model series to environments, `infer/` contains model runners, `data/` contains dataset backends, `scoring/` contains scoring/report logic, and `utils/` holds shared helpers. Scripts are in `scripts/`; fixtures are in `tests/samples/` and `tests/media/`. Put generated outputs under `work_dirs/`.

## Build, Test, and Development Commands

Set the repository on `PYTHONPATH` before local runs:

```bash
export PYTHONPATH=./:$PYTHONPATH
```

Create model-specific environments from `env_files/`, for example:

```bash
conda create -p envs/qwenvl python=3.10 -y
conda activate envs/qwenvl
pip install -r env_files/qwenvl_requirements.txt
```

Run a local sample evaluation directly:

```bash
python mmeval/run.py --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset local@json --infile tests/samples/multi_image_video_interleave.json \
  --img_dir tests/media/448 --out_dir work_dirs/local_test \
  --gpu_per_parallel 1 --parallel_per_task 1
```

Use `scripts/update_eval_summary.py` to refresh summaries.

## Starting Evaluations with Scripts

Use `scripts/eval_model.sh` for inference-only VLMEvalKit runs. Pass a model path or HuggingFace ID, a quoted space-separated dataset list, and optionally an output root:

```bash
scripts/eval_model.sh Qwen/Qwen3-VL-4B-Instruct "MMStar BLINK"
scripts/eval_model.sh /path/to/checkpoint-200-merged "MMBench_dev_en MathVista_MINI" work_dirs/ckpt200
```

Use overrides such as `GPU_PER_PARALLEL=1`, `PARALLEL_PER_TASK=8`, `SAMPLE_NUM=20`, `MAX_NEW_TOKENS=1024`, `RUN_SETUP=false`, and `RUN_GPU_SM=false`. Outputs go to `work_dirs/<model>/<dataset>/` unless `OUT_ROOT` is provided.

Use `scripts/eval_and_score_any_model.sh` to run inference and LLM-judge scoring together:

```bash
scripts/eval_and_score_any_model.sh qwen3vl_4b "MMStar BLINK" work_dirs/qwen3vl4b_eval exact,template,llm
```

Use `scripts/score_any_model_judge.sh work_dirs/<run> "MMStar BLINK"` to score an existing run. LLM scoring requires judge credentials in `KEY_CONF` or the relevant API environment variables.

For checkpoint sweeps, prepare the checkpoint list before calling the entry script; do not embed `$(find ...)` in the launcher command. This keeps quoting stable and makes failures easier to debug:

```bash
cd /mnt/bn/algo-masp-nas-arnold2/yijunliang/project
bash trail_setup/trail_setup.sh
bash trail_setup/run_gpu_sm.sh

cd opsd/simple-mmeval
CHECKPOINT_ROOT=/mnt/bn/algo-masp-nas-arnold2/yijunliang/project/opsd/unsup-opsd/work_dirs/opsd_vlm/qwen3vl4b_1epochs_lr1e5_bs4_fixteach_aug_open-mm-reasoner-74k@virl39k_8gpu
mapfile -t CHECKPOINTS < <(find "$CHECKPOINT_ROOT" -maxdepth 2 -type d -name 'checkpoint-*' | sort -V)

DATASETS="MMStar BLINK MMBench_dev_en MathVista_MINI" \
MATCHING_ORDER=llm \
SCORE_OUTPUT_NAME=score_llm_only.json \
scripts/merge_eval_score_models.sh "${CHECKPOINTS[@]}"
```

## Coding Style & Naming Conventions

Follow existing Python style: 4-space indentation, snake_case names, and compact model runners named after their series, such as `mmeval/infer/qwen3_vl.py`. New model files should define `TaskRunner(Task)` and keep model-specific parsing/generation local. Register new models in both mappings in `mmeval/registry.py`.

## Testing Guidelines

There is no central pytest configuration. Validate changes with `tests/samples/`, choosing the smallest fixture that covers the modality. For model additions, run at least `single_image_start.json` and `multi_image_video_interleave.json` when supported. Confirm `work_dirs/<test_name>/result.json` contains assistant responses.

## Commit & Pull Request Guidelines

Recent commits use short subjects such as `update result csv` and `Support DATASETS env var in eval_qwen25vl3b.sh`. Prefer clear imperative subjects, for example `Add qwen3 vl scoring script`. Pull requests should follow `.github/pull_request_template.md`: describe the change, mark its type, include the test command, and mention README or registry updates.

## Security & Configuration Tips

Do not commit API keys, local `.env` files, or private model paths. Keep per-model dependencies in `env_files/<series>_requirements.txt`; document manual installs such as `flash-attn` when needed.
