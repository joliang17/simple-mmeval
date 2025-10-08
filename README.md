# Simple-MMEval
- [Overview](#overview)
- [Execution Flow](#execution-flow)
- [Supported Models](#supported-models)
- [Supported Datasets](#supported-datasets)
  - [Custom Dataset Support](#custom-dataset-support)
  - [VLMEvalKit Integration](#vlmevalkit-integration)
  - [LMMS-eval Integration (TODO)](#lmms-eval-integration-todo)

## Overview

Simple-MMEval is a simple, modular multimodal evaluation framework designed around these principles:
- **Simple**: Adding models and datasets requires only a Python file with a few lines of code; Run inference with just a few command lines
- **Self-contained**: Each model serie has its own complete inference implementation in a single file, and we provide a [Docker image](https://hub.docker.com/repository/docker/icywang444/simple-mmeval/image-management) for each serie of model.
- **Scalable**: Automatic parallel processing across multiple GPUs; Efficient memory management (lazy loading) for large datasets
- **Modular**: Evaluation process is broken into separate steps connected by a Python connector, enabling easy expansion of datasets and models
- **2 inference options**:
  - **Text Generation**: For open-ended questions where models generate free-form responses
  - **Logit Scoring**: For multiple-choice questions using conditional probability scoring - computes logit scores of completing the prompt with each choice option conditioned on the prompt, an MLLM adaptation of [Minicon](https://github.com/kanishkamisra/minicons/blob/master/minicons/scorer.py)'s LLM logit evaluator
    
## Execution Flow

With `scripts/test_bed/qwen2d5-score.sh` as an example
### 1. Script Execution 

```bash
export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/image-qca.json \
    --dataset local@json \
    --out_dir test_bed/test_qwen2d5_score \
    --img_dir test_bed \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --score_target
```
| Parameter | Purpose | Example Value |
|-----------|---------|---------------|
| `--infile` | Input dataset file path | `test_bed/image-qca.json` - JSON file containing evaluation samples |
| `--dataset` | Dataset type identifier | `local@json` - Uses local JSON dataset loader, TODO: lmms, evalkit, customized |
| `--out_dir` | Output directory for results | `test_bed/test_qwen2d5_score` - Where final results are saved |
| `--img_dir` | Base directory for media files | `test_bed` - Root path for resolving relative image/video paths |
| `--model_name_or_path` | Model identifier or path | `Qwen/Qwen2.5-VL-3B-Instruct` - HuggingFace model name |
| `--gpu_per_parallel` | GPUs allocated per parallel task | `1` - Each worker uses 1 GPU |
| `--parallel_per_task` | Number of parallel workers | `1` - Run single worker (no parallelization) |
| `--score_target` | Enable scoring mode | Flag - Score multiple choice options instead of generating text |


### 2. Main Runner (`mmeval/run.py`)

The runner orchestrates the entire evaluation process:

#### Model Registration & Environment Selection
```python
series = get_series(model_name_or_path.split("/")[-1])
infer_file = series_infer_env_mapping[series]["infer_file"]
infer_env = series_infer_env_mapping[series]["env"]
```

Registry pattern (`mmeval/registery.py`) maps model names to their inference implementations and environments

#### Resource Management & Parallel Scheduling
The runner implements sophisticated GPU scheduling:
- Calculates available GPUs and resource requirements
- Spawns multiple parallel processes using conda environments
- Monitors completion and resource cleanup
- Merges results from all parallel workers

Parallel execution with automatic resource management eliminates manual coordination and maximizes GPU utilization.

### 3. Model-Specific Inference (`mmeval/infer/qwenvl2d5.py`)

Each model has its own inference implementation inheriting from the base `Task` class:

```python
class TaskRunner(Task):
    def load_model(self, args):
        # Model-specific loading logic
        
    def run_sample(self, sample: dict):
        # Model-specific inference logic for each sample
```
The framework supports two inference modes:
- **Generation mode**: Model generates free-form text responses
- **Scoring mode** (`--score_target`): Model scores multiple choice options

```python
if not self.args.score_target:
    ori_sample["response"] = self._generate_response(text, image_inputs, video_inputs)
else:
    ori_sample.update(self._score_choices(text, image_inputs, video_inputs, sample))
```

### 4. Dataset Loading (`mmeval/data/`)

The framework provides a pluggable dataset system:

#### Dataset Registry
```python
def load_dataset(args):
    if args.dataset in VLMEVALKIT_DATASETS:
        return VLMEvalKitDataset(args)
    elif args.dataset == "local@json":
        return LocalJSONDataset(args)
```

TODO: local, lmms, evalkit

### 5. Response Handling (`mmeval/utils/res_handler.py`)

The response handler manages incremental saving and caching:

- Maintains cache of completed samples
- Saves progress periodically to prevent data loss
- Resumes from cache on restart
- Validates completion across all workers

Separate worker outputs are merged into a single result file, abstracting parallelization from the user.

### Scalability
- Automatic parallel processing across multiple GPUs
- Worker-based architecture scales with available resources
- Efficient memory management for large datasets

## Supported Models

Currently supported model series:


| Model Series | Models | Usage Example |
|--------------|--------|---------------|
| **Qwen2.5-VL** | Qwen2.5-VL-3B-Instruct, Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-32B-Instruct, Qwen2.5-VL-72B-Instruct | [`scripts/test_bed/qwen2d5.sh`](scripts/test_bed/qwen2d5.sh) |
| **VideoLLaMA2** | VideoLLaMA2-7B, VideoLLaMA2-13B | [`scripts/test_bed/videollama2.sh`](scripts/test_bed/videollama2.sh) |
| **BLIP2-FLAN-T5** | Salesforce/blip2-flan-t5-xl, Salesforce/blip2-flan-t5-xxl | [`scripts/test_bed/blip2-flan-t5-xl.sh`](scripts/test_bed/blip2-flan-t5-xl.sh) |
| **LLaVA-OneVision-1.5** | lmms-lab/LLaVA-OneVision-1.5-8B-Instruct | [`scripts/test_bed/llava-ov-1d5.sh`](scripts/test_bed/llava-ov-1d5.sh) |
| **Cambrian** | cambrian-8b, cambrian-13b, cambrian-34b, cambrian-phi3-3b | [`scripts/test_bed/cambrian-multi-image-video-interleave.sh`](scripts/test_bed/cambrian-multi-image-video-interleave.sh) |
| **Gemma3** | gemma-3-4b-it, gemma-3-12b-it, gemma-3-27b-it | [`scripts/test_bed/gemma3-multi-image-interleave.sh`](scripts/test_bed/gemma3-multi-image-interleave.sh) |
| **GLM-4V** | glm-4v-9b | [`scripts/test_bed/glm-4v-single-image-start.sh`](scripts/test_bed/glm-4v-single-image-start.sh) |
| **InternVL-Chat** | InternVL-Chat-V1-1, InternVL-Chat-V1-2, InternVL-Chat-V1-2-Plus | [`scripts/test_bed/internvl-chat-multi-image-video-interleave.sh`](scripts/test_bed/internvl-chat-multi-image-video-interleave.sh) |
| **InternVL-Chat-1.5** | Mini-InternVL-Chat-2B-V1-5, Mini-InternVL-Chat-4B-V1-5, InternVL-Chat-V1-5 | [`scripts/test_bed/internvl-chat1d5-multi-image-video-interleave.sh`](scripts/test_bed/internvl-chat1d5-multi-image-video-interleave.sh) |
| **InternVL2** | InternVL2-1B, InternVL2-2B, InternVL2-4B, InternVL2-8B, InternVL2-26B, InternVL2-40B, InternVL2-Llama3-76B | [`scripts/test_bed/internvl2-multi-image-video-interleave.sh`](scripts/test_bed/internvl2-multi-image-video-interleave.sh) |
| **InternVL2.5** | InternVL2_5-1B, InternVL2_5-2B, InternVL2_5-4B, InternVL2_5-8B, InternVL2_5-26B, InternVL2_5-38B, InternVL2_5-78B, InternVL2_5-1B-MPO, InternVL2_5-2B-MPO, InternVL2_5-4B-MPO, InternVL2_5-8B-MPO, InternVL2_5-26B-MPO, InternVL2_5-38B-MPO, InternVL2_5-78B-MPO | [`scripts/test_bed/internvl2d5-multi-image-video-interleave.sh`](scripts/test_bed/internvl2d5-multi-image-video-interleave.sh) |
| **InternVL3** | InternVL3-1B, InternVL3-2B, InternVL3-8B, InternVL3-9B, InternVL3-14B, InternVL3-38B, InternVL3-78B, InternVL3-1B-Instruct, InternVL3-2B-Instruct, InternVL3-8B-Instruct, InternVL3-9B-Instruct, InternVL3-14B-Instruct, InternVL3-38B-Instruct, InternVL3-78B-Instruct, InternVL3-1B-Pretrained, InternVL3-2B-Pretrained, InternVL3-8B-Pretrained, InternVL3-9B-Pretrained, InternVL3-14B-Pretrained, InternVL3-38B-Pretrained, InternVL3-78B-Pretrained | [`scripts/test_bed/internvl3-multi-image-video-interleave.sh`](scripts/test_bed/internvl3-multi-image-video-interleave.sh) |
| **Janus-Pro** | Janus-Pro-1B, Janus-Pro-7B | [`scripts/test_bed/janus-pro-multi-image-video-interleave.sh`](scripts/test_bed/janus-pro-multi-image-video-interleave.sh) |
| **LLaVA-OneVision** | LLaVA-OneVision-Qwen2-0.5B, LLaVA-OneVision-Qwen2-7B, LLaVA-OneVision-Qwen2-72B | [`scripts/test_bed/llava_ov.sh`](scripts/test_bed/llava_ov.sh) |
| **Mantis** | Mantis-8B-clip-llama3, Mantis-8B-siglip-llama3 | [`scripts/test_bed/mantis-multi-image-interleave.sh`](scripts/test_bed/mantis-multi-image-interleave.sh) |
| **Mantis-Idefics2** | Mantis-8B-Idefics2 | [`scripts/test_bed/mantis-idefics2-multi-image-interleave.sh`](scripts/test_bed/mantis-idefics2-multi-image-interleave.sh) |
| **Moondream1** | moondream1 | [`scripts/test_bed/moondream1.sh`](scripts/test_bed/moondream1.sh) |
| **Moondream2** | moondream2 | [`scripts/test_bed/moondream2.sh`](scripts/test_bed/moondream2.sh) |
| **Ovis1.5** | Ovis1.5-Llama3-8B, Ovis1.5-Gemma2-9B | [`scripts/test_bed/ovis1d5-single-image-start.sh`](scripts/test_bed/ovis1d5-single-image-start.sh) |
| **Parrot** | Parrot-7B, Parrot-14B | [`scripts/test_bed/parrot.sh`](scripts/test_bed/parrot.sh) |


## Supported Datasets


| Dataset                                                                  | Split/Version | Loader Type  |
| ------------------------------------------------------------------------ | ------------- | ------------ |
| [MMBench-en](https://huggingface.co/datasets/mm-eval/MMBench-en) | dev, test     | `mmeval_hf@MMBench-en` |
| [MMBench-cc](https://huggingface.co/datasets/mm-eval/MMBench-cc) | test     | `mmeval_hf@mm-eval/MMBench-cc` |
| [MMMU](https://huggingface.co/datasets/mm-eval/MMMU) | dev, test, validation | `mmeval_hf@mm-eval/MMMU` |



### Custom Dataset Support
- **Local JSON**: Simple JSON format for custom evaluation datasets
- **Custom Loaders**: Easy integration of new dataset formats through the modular dataset system

### Dataset Format Example
```json
[
  {
    "id": "sample_1", 
    "prompt": "What is shown in <image>?",
    "media": ["image1.jpg"],
    "choices": ["A cat", "A dog", "A bird"]
  }
]
```

### VLMEvalKit Integration
Direct support for 100+ datasets from VLMEvalKit, including:

| Category | Datasets |
|----------|----------|
| **General VQA** | MME, MMBench, MMMU, MMVet, POPE, LLaVABench |
| **OCR & Document** | OCRBench, DocVQA, InfoVQA, ChartQA, TextVQA |
| **Scientific** | ScienceQA, MathVista, AI2D, MMSci |
| **Multimodal Reasoning** | BLINK, LogicVista, MMT-Bench, MMVP |
| **Multilingual** | MMMB (Arabic, Chinese, English, Portuguese, Russian, Turkish) |
| **Specialized** | PathVQA (Medical), Q-Bench (Quality Assessment), RealWorldQA |

Usage example: TODO


### LMMS-eval Integration (TODO)
Direct support for XX datasets from VLMEvalKit, including:
