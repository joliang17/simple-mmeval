import argparse
import inspect
import os
import sys
import warnings
from dataclasses import dataclass, field, fields
from typing import Dict, Optional, Sequence, get_args

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)

    # parameters for model
    low_cpu_mem_usage: Optional[bool] = field(default=None, metadata={"help": "Tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model."})
    attn_implementation: Optional[str] = field(default=None, metadata={"help": "The attention implementation to use in the model (if relevant)."})

    # parameters for model inference
    dtype: Optional[str] = field(default=None, metadata={"help": "Override the default torch.dtype and load the model under a specific dtype."})
    device_map: Optional[str] = field(default=None, metadata={"help": "A map that specifies where each submodule should go."})
    
    # parameters that control the length of the output
    max_length: Optional[int] = field(default=None, metadata={"help": "The maximum length the generated tokens can have."})
    max_new_tokens: Optional[int] = field(default=None, metadata={"help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."})
    min_length: Optional[int] = field(default=None, metadata={"help": "The minimum length of the sequence to be generated."})
    min_new_tokens: Optional[int] = field(default=None, metadata={"help": "The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt."})
    early_stopping: Optional[bool] = field(default=None, metadata={"help": "Controls the stopping condition for beam-based methods, like beam-search."})
    max_time: Optional[float] = field(default=None, metadata={"help": "The maximum amount of time you allow the computation to run for in seconds."})
    
    # parameters that control the generation strategy used
    do_sample: Optional[bool] = field(default=None, metadata={"help": "Whether or not to use sampling ; use greedy decoding otherwise."})
    num_beams: Optional[int] = field(default=None, metadata={"help": "Number of beams for beam search. 1 means no beam search."})

    # parameters that control the cache
    use_cache: Optional[bool] = field(default=None, metadata={"help": "Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding."})
    cache_implementation: Optional[str] = field(default=None, metadata={"help": "Name of the cache class that will be instantiated in generate."})

    # parameters for manipulation of the model output logits
    temperature: Optional[float] = field(default=None, metadata={"help": "The value used to module the next token probabilities."})
    top_k: Optional[int] = field(default=None, metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k-filtering."})
    top_p: Optional[float] = field(default=None, metadata={"help": "If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation."})
    min_p: Optional[float] = field(default=None, metadata={"help": "Minimum token probability, which will be scaled by the probability of the most likely token."})
    diversity_penalty: Optional[float] = field(default=None, metadata={"help": "This value is subtracted from a beam’s score if it generates a token same as any beam from other group at a particular time."})
    repetition_penalty: Optional[float] = field(default=None, metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty."})
    length_penalty: Optional[float] = field(default=None, metadata={"help": "Exponential penalty to the length that is used with beam-based generation."})

@dataclass
class DataArguments:
    dataset: Optional[str] = field(default=None,
                           metadata={"help": "name of the dataset."})
    split: Optional[str] = field(default=None,
                           metadata={"help": "split of the dataset for huggingface."})
    infile: Optional[str]= field(default=None,
                           metadata={"help": "input file."})
    img_dir: Optional[str] = field(default=None,
                           metadata={"help": "image directory."})
    template: Optional[str] = field(default=None,
                           metadata={"help": "template file path or template string."})
    circular: bool = field(default=False, 
                           metadata={"help": "whether to prepare data for circular evaluation."})
    resize: Optional[int] = field(default=None,
                           metadata={"help": "resize images to this pixel value."})

@dataclass
class InferenceArguments:
    save_freq: int = field(default=3, metadata={"help": "save frequency for cache."})
    max_retry: int = field(default=1, metadata={"help": "maximum number of retries for entire dataset."}) 
    max_retry_sample: int = field(default=1, metadata={"help": "maximum number of retries for one inference sample."}) 
    out_dir: Optional[str] = field(default=None,
                           metadata={"help": "output directory."})
    score_target: bool = field(default=False, metadata={"help": "whether to output scores for each choice."})
    resume: bool = field(default=True, metadata={"help": "whether to resume from cache."})
    
@dataclass
class ExperimentArguments:
    gpu_per_parallel: int = field(default=1, metadata={"help": "number of gpus per task"})
    parallel_per_task: int = field(default=4, metadata={"help": "number of parallel tasks."}) 
    rank: int = field(default=-1, metadata={"help": "rank for parallel inference"})
    no_conda: bool = field(default=False, metadata={"help": "use current python env instead of conda"})


@dataclass
class ScoreRuntimeArguments:
    out_dir: Optional[str] = field(default=None, metadata={"help": "output directory containing result.json files"})
    parallel_per_task: int = field(default=4, metadata={"help": "number of sample workers inside each result.json"})


@dataclass
class ScoreArguments:
    score_scan_recursive: bool = field(default=True, metadata={"help": "recursively scan out_dir for result.json"})
    score_result_glob: str = field(default="**/result.json", metadata={"help": "glob pattern for result files in score mode"})
    score_output_name: str = field(default="score.json", metadata={"help": "output score json file name"})
    score_progress_bar: bool = field(default=True, metadata={"help": "show progress bar while scoring samples"})
    score_resume: bool = field(default=True, metadata={"help": "resume scoring from score tmp/final files when available"})
    score_save_freq: int = field(default=20, metadata={"help": "flush frequency for score resume tmp file"})
    score_dump_failures: bool = field(default=False, metadata={"help": "dump failed/unmatched samples into score_failures.json"})
    score_debug: bool = field(default=False, metadata={"help": "write matcher trace into score outputs"})
    matching_order: str = field(default="exact,template", metadata={"help": "matcher chain order, comma-separated"})
    matching_stop_on_first: bool = field(default=True, metadata={"help": "stop matcher chain once one matcher succeeds"})

    score_gt_field: str = field(default="answer", metadata={"help": "field name for ground-truth in result sample"})
    score_pred_field: str = field(default="messages[-1].response", metadata={"help": "field path for prediction in result sample"})
    score_type_field: Optional[str] = field(default=None, metadata={"help": "optional field name for question type"})
    score_force_question_type: str = field(default="auto", metadata={"help": "force question type: auto|mcq|open"})

    judge_provider: Optional[str] = field(default=None, metadata={"help": "llm judge provider: openai|azure_openai"})
    judge_model: Optional[str] = field(default=None, metadata={"help": "llm judge model/deployment name"})
    judge_max_retry: int = field(default=2, metadata={"help": "llm judge max retry count"})
    judge_temperature: float = field(default=0.0, metadata={"help": "llm judge generation temperature"})
    judge_concurrency: int = field(default=1, metadata={"help": "reserved for llm judge concurrency control"})
    judge_include_reason: bool = field(default=False, metadata={"help": "include llm judge reason in output"})


ARGUMENT_DATACLASSES = (ModelArguments, DataArguments, InferenceArguments, ExperimentArguments, ScoreArguments)
INFER_ARGUMENT_DATACLASSES = (ModelArguments, DataArguments, InferenceArguments, ExperimentArguments)
SCORE_ARGUMENT_DATACLASSES = (ScoreRuntimeArguments, ScoreArguments)


BOOL_DEFAULTS = {
    f.name: f.default
    for dc in ARGUMENT_DATACLASSES
    for f in fields(dc)
    if (
        f.type is bool
        or (
            bool in get_args(f.type)
            and type(None) in get_args(f.type)
        )
    )
}

def _add_dataclass_arguments(parser, dataclasses):
    for dc in dataclasses:
        for f in fields(dc):
            tp = f.type
            type_args = getattr(tp, '__args__', None)
            if type_args and type(None) in type_args:
                tp = next(a for a in type_args if a is not type(None))
            help_text = f.metadata.get("help", "")
            cli_name = f.name.replace("_", "-")
            if tp is bool:
                if f.default is True:
                    parser.add_argument(f"--no-{cli_name}", f"--no_{f.name}",
                                        dest=f.name, action="store_false",
                                        default=True, help=help_text)
                elif f.default is False:
                    parser.add_argument(f"--{cli_name}", f"--{f.name}",
                                        dest=f.name, action="store_true",
                                        default=False, help=help_text)
                else:
                    parser.add_argument(f"--{cli_name}", f"--{f.name}",
                                        dest=f.name, action="store_true",
                                        default=None, help=help_text)
                    parser.add_argument(f"--no-{cli_name}", f"--no_{f.name}",
                                        dest=f.name, action="store_false")
            else:
                parser.add_argument(f"--{cli_name}", f"--{f.name}",
                                    dest=f.name, type=tp,
                                    default=f.default, help=help_text)


def parse_args(mode: str = "auto"):
    mode = (mode or "auto").strip().lower()
    if mode == "auto":
        script_name = os.path.basename(sys.argv[0])
        mode = "score" if script_name == "score.py" else "infer"

    if mode == "infer":
        dataclasses = INFER_ARGUMENT_DATACLASSES
    elif mode == "score":
        dataclasses = SCORE_ARGUMENT_DATACLASSES
    else:
        raise ValueError(f"Unsupported parse mode: {mode}. Expected infer|score|auto.")

    parser = argparse.ArgumentParser()
    _add_dataclass_arguments(parser, dataclasses)
    return parser.parse_args()
def parse_model_kwargs(args, default_kwargs=None):
    default_kwargs = default_kwargs or {}
    model_kwargs = {}

    for key in (
        "low_cpu_mem_usage", "attn_implementation", "device_map"
    ):
        value = getattr(args, key, None)
        if value is None:
            value = default_kwargs.get(key)
        if value is not None:
            model_kwargs[key] = value

    return model_kwargs

def parse_gen_kwargs(args, default_kwargs=None):
    default_kwargs = default_kwargs or {}
    gen_kwargs = {}

    for key in (
        "max_length", "max_new_tokens",
        "min_length", "min_new_tokens",
        "early_stopping", "max_time",
        "do_sample", "num_beams",
        "use_cache", "cache_implementation",
        "temperature", "top_k", "top_p", "min_p",
        "diversity_penalty", "repetition_penalty", "length_penalty",
    ):
        value = getattr(args, key, None)
        if value is None:
            value = default_kwargs.get(key)
        if value is not None:
            gen_kwargs[key] = value

    return gen_kwargs


# Default parameter name mapping (Transformers -> OpenAI-compatible API)
GEN_KWARGS_MAPPING = {
    "max_new_tokens": "max_completion_tokens",
}


def filter_gen_kwargs(gen_kwargs, api_method, mapping=None):
    """Filter and map gen_kwargs to match target API method signature."""
    if mapping is None:
        mapping = GEN_KWARGS_MAPPING

    mapped_all = {mapping.get(key, key): value for key, value in gen_kwargs.items()}
    try:
        sig = inspect.signature(api_method)
    except (TypeError, ValueError):
        warnings.warn("Could not inspect target API signature, all generation kwargs will be passed through.")
        return mapped_all

    # api_method accepts **kwargs, so all mapped parameters are supported
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return mapped_all

    supported = set(sig.parameters.keys())

    filtered_kwargs = {}
    unsupported_kwargs = []

    for key, value in gen_kwargs.items():
        mapped_key = mapping.get(key, key)
        if mapped_key in supported:
            filtered_kwargs[mapped_key] = value
        else:
            unsupported_kwargs.append(key)

    if unsupported_kwargs:
        warnings.warn(f"Unsupported generation parameters will be ignored: {unsupported_kwargs}")

    return filtered_kwargs

