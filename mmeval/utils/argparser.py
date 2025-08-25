from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, TYPE_CHECKING
import transformers


if TYPE_CHECKING:
    import transformers

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)

    # parameters for model
    low_cpu_mem_usage: bool = field(default=None, metadata={"help": "Tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model."})
    attn_implementation: str = field(default=None, metadata={"help": "The attention implementation to use in the model (if relevant)."})

    # parameters for model inference
    dtype: str = field(default=None, metadata={"help": "Override the default torch.dtype and load the model under a specific dtype."})
    device_map: str = field(default=None, metadata={"help": "A map that specifies where each submodule should go."})
    
    # parameters that control the length of the output
    max_length: int = field(default=None, metadata={"help": "The maximum length the generated tokens can have."})
    max_new_tokens: int = field(default=None, metadata={"help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."})
    min_length: int = field(default=None, metadata={"help": "The minimum length of the sequence to be generated."})
    min_new_tokens: int = field(default=None, metadata={"help": "The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt."})
    early_stopping: bool = field(default=None, metadata={"help": "Controls the stopping condition for beam-based methods, like beam-search."})
    max_time: float = field(default=None, metadata={"help": "The maximum amount of time you allow the computation to run for in seconds."})
    
    # parameters that control the generation strategy used
    do_sample: bool = field(default=None, metadata={"help": "Whether or not to use sampling ; use greedy decoding otherwise."})
    num_beams: int = field(default=None, metadata={"help": "Number of beams for beam search. 1 means no beam search."})

    # parameters that control the cache
    use_cache: bool = field(default=None, metadata={"help": "Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding."})
    cache_implementation: str = field(default=None, metadata={"help": "Name of the cache class that will be instantiated in generate."})

    # parameters for manipulation of the model output logits
    temperature: float = field(default=None, metadata={"help": "The value used to module the next token probabilities."})
    top_k: int = field(default=None, metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k-filtering."})
    top_p: float = field(default=None, metadata={"help": "If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation."})
    min_p: float = field(default=None, metadata={"help": "Minimum token probability, which will be scaled by the probability of the most likely token."})
    diversity_penalty: float = field(default=None, metadata={"help": "This value is subtracted from a beam’s score if it generates a token same as any beam from other group at a particular time."})
    repetition_penalty: float = field(default=None, metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty."})
    length_penalty: float = field(default=None, metadata={"help": "Exponential penalty to the length that is used with beam-based generation."})

@dataclass
class DataArguments:
    dataset: str = field(default=None,
                           metadata={"help": "name of the dataset."})
    split: str = field(default="None",
                           metadata={"help": "split of the dataset for huggingface."})
    infile: Optional[str]= field(default=None,
                           metadata={"help": "input file."})
    img_dir: Optional[str] = field(default=None,
                           metadata={"help": "image directory."})
    circular: bool = field(default=False, 
                           metadata={"help": "whether to prepare data for circular evaluation."})
    resize: int = field(default=None,
                           metadata={"help": "resize images to this pixel value."})

@dataclass
class InferenceArguments:
    save_freq: int = field(default=3, metadata={"help": "save frequency for cache."})
    max_retry: int = field(default=1, metadata={"help": "maximum number of retries for entire dataset."}) 
    max_retry_sample: int = field(default=1, metadata={"help": "maximum number of retries for one inference sample."}) 
    out_dir: str = field(default=None,
                           metadata={"help": "output directory."})
    score_target: bool = field(default=False, metadata={"help": "whether to output scores for each choice."})
    resume: bool = field(default=True, metadata={"help": "whether to resume from cache."})
    
@dataclass
class ExperimentArguments:
    gpu_per_parallel: int = field(default=1, metadata={"help": "number of gpus per task"})
    parallel_per_task: int = field(default=4, metadata={"help": "number of parallel tasks."}) 
    rank: int = field(default=-1, metadata={"help": "rank for parallel inference"}) 


def parse_args():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, InferenceArguments, ExperimentArguments))
    args = parser.parse_args()
    return args

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

