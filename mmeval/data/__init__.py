from .local import LocalJSONDataset
from .evalkit import load_evalkit_dataset
from .mmeval_hf import MMEvalHFDataset


def load_dataset(args):
    """Load dataset based on the dataset specification.
    
    Supported formats:
    - local@json: Load from local JSON file
    - evalkit@dataset_name: Load VLMEvalKit dataset
    - evalkit@url: Load from remote TSV URL
    - dataset_name: Legacy support for VLMEvalKit datasets
    """
    if args.dataset == "local@json":
        return LocalJSONDataset(args)
    elif args.dataset.endswith(".tsv"):
        return load_evalkit_dataset(args)
    elif args.dataset.startswith("evalkit@"):
        return load_evalkit_dataset(args)
    elif args.dataset.startswith("mmeval_hf@"):
        return MMEvalHFDataset(args)
    else:
        raise ValueError(f"Unsupported dataset specification: {args.dataset}") 