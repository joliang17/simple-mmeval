import os
import pandas as pd
from typing import Any
from mmeval.data.tsv import TSVDataset
from mmeval.data.utils import download_tsv

# VLMEvalKit supported datasets
VLMEVALKIT_DATASET_LIST = [
   '3DSRBench',
   'A-Bench_TEST',
   'A-Bench_VAL',
   'A-OKVQA',
   'A4Bench',
   'AI2D_TEST',
   'AI2D_TEST_NO_MASK',
   'AMBER',
   'AesBench_TEST',
   'AesBench_VAL',
   'BLINK',
   'CMMU_MCQ',
   'CRPE_EXIST',
   'CharXiv_descriptive_val',
   'CharXiv_reasoning_val',
   'ChartQA_TEST',
   'Creation_MMBench',
   'GOBench',
   'GQA_TestDev_Balanced',
   'HRBench4K',
   'HRBench8K',
   'InfoVQA_TEST',
   'InfoVQA_VAL',
   'LEGO',
   'LLaVABench',
   'LogicVista',
   'LogicVista',
   'MIA-Bench',
   'MLLMGuard_DS',
   'MM-IFEval',
   'MM-Math',
   'MMBench_dev_ar',
   'MMBench_dev_cn',
   'MMBench_dev_en',
   'MMBench_dev_pt',
   'MMBench_dev_ru',
   'MMBench_dev_tr',
   'MMMB',
   'MMMB_ar',
   'MMMB_cn',
   'MMMB_en',
   'MMMB_pt',
   'MMMB_ru',
   'MMMB_tr',
   'MMSci_DEV_Captioning_image_only',
   'MMSci_DEV_MCQ',
   'MMStar',
   'MMT-Bench_ALL',
   'MMT-Bench_VAL',
   'MMVP',
   'MMVet',
   'MMVet_Hard',
   'MTL_MMBench_DEV',
   'MUIRBench',
   'MathVerse_MINI',
   'MathVerse_MINI_Text_Dominant',
   'MathVerse_MINI_Text_Lite',
   'MathVerse_MINI_Vision_Dominant',
   'MathVerse_MINI_Vision_Intensive',
   'MathVerse_MINI_Vision_Only',
   'MathVision',
   'MathVision_MINI',
   'MathVista_MINI',
   'MedXpertQA_MM_test',
   'MicroBench',
   'MicroVQA',
   'NaturalBenchDataset',
   'OCRBench',
   'OlympiadBench',
   'OmniMedVQA',
   'POPE',
   'PathMMU_TEST',
   'PathMMU_VAL',
   'PathVQA_TEST',
   'PathVQA_VAL',
   'Q-Bench1_TEST',
   'Q-Bench1_VAL',
   'R-Bench-Dis',
   'R-Bench-Ref',
   'RealWorldQA',
   'SEEDBench2',
   'SEEDBench2_Plus',
   'SEEDBench_IMG',
   'ScienceQA_TEST',
   'ScienceQA_VAL',
   'TableVQABench',
   'TaskMeAnything_v1_imageqa_random',
   'VCR_EN_EASY_ALL',
   'VCR_EN_HARD_ALL',
   'VCR_ZH_EASY_ALL',
   'VCR_ZH_HARD_ALL',
   'VL-RewardBench',
   'VStarBench',
   'VisOnlyQA-VLMEvalKit',
   'VizWiz',
   'WeMath',
   'WeMath_COT',
   'WildVision',
   'WorldMedQA-V',
   'atomic_dataset',
   'electro_dataset',
   'hle',
   'mechanics_dataset',
   'optics_dataset',
   'quantum_dataset',
   'statistics_dataset'
]

VLMEVALKIT_MULTIPART_DATASET_CONFIG = {
    "MicroBench": {"filename_pattern": "microbench_part_{}", "start_idx": 1, "end_idx": 14},
    "XLRS-Bench-lite": {"filename_pattern": "XLRS-Bench-lite_part{}", "start_idx": 0, "end_idx": 14}, 
    "OmniEarth-Bench": {"filename_pattern": "OmniEarth-Bench_MCQ_part{}", "start_idx": 0, "end_idx": 14},
    "OmniMedVQA": {"filename_pattern": "omnimedbench_part_{}", "start_idx": 1, "end_idx": 14}
}

VLMEVALKIT_CONCAT_DATASET_SETS = {
    'MMMB': ['MMMB_ar', 'MMMB_cn', 'MMMB_en', 'MMMB_pt', 'MMMB_ru', 'MMMB_tr'],
    'MTL_MMBench_DEV': [
        'MMBench_dev_ar', 'MMBench_dev_cn', 'MMBench_dev_en',
        'MMBench_dev_pt', 'MMBench_dev_ru', 'MMBench_dev_tr'
    ]
}

def load_single_dataset(dataset_name: str, dataset_dir: str) -> pd.DataFrame:
    """Load single dataset file for VLMEvalKit specific datasets.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame from the single dataset file
    """
    file_path = os.path.join(dataset_dir, f"{dataset_name}.tsv")

    if not os.path.exists(file_path):
        dataset_url = f"https://huggingface.co/datasets/mm-eval/VLMEvalKit/resolve/main/{dataset_name}.tsv"
        download_tsv(dataset_url, file_path)

def load_multipart_dataset(dataset_name: str, dataset_dir: str) -> pd.DataFrame:
    """Load multipart dataset files for VLMEvalKit specific datasets.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load
        
    Returns
    -------
    pd.DataFrame
        Loaded and merged DataFrame from all parts, saved as dataset_name.tsv
    """
    file_path = os.path.join(dataset_dir, f"{dataset_name}.tsv")

    if not os.path.exists(file_path):
        config = VLMEVALKIT_MULTIPART_DATASET_CONFIG[dataset_name]
        pattern = config["filename_pattern"]
        
        # Load and merge all parts into DataFrames
        dataframe = []
        for part_idx in range(config["start_idx"], config["end_idx"]+1):
            sub_file_path = os.path.join(dataset_dir, f"{pattern.format(part_idx)}.tsv")
            if not os.path.exists(sub_file_path):
                sub_dataset_url = f"https://huggingface.co/datasets/mm-eval/VLMEvalKit/resolve/main/{pattern.format(part_idx)}.tsv"
                download_tsv(sub_dataset_url, sub_file_path)
            sub_dataframe = pd.read_csv(sub_file_path, sep='\t')
            dataframe.append(sub_dataframe)
        
        # Concatenate all dataframes
        combined_df = pd.concat(dataframe, ignore_index=True)
        combined_df.to_csv(file_path, sep='\t', index=False, chunksize=100000)

def load_concat_dataset(dataset_name: str, dataset_dir: str) -> pd.DataFrame:
    """Load multiple datasets for VLMEvalKit composite datasets.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load
    
    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame from all parts, saved as dataset_name.tsv
    """
    file_path = os.path.join(dataset_dir, f"{dataset_name}.tsv")

    if not os.path.exists(file_path):
        dataset_list = VLMEVALKIT_CONCAT_DATASET_SETS[dataset_name]
        dataframes = []
        for sub_dataset_name in dataset_list:
            sub_file_path = os.path.join(dataset_dir, f"{sub_dataset_name}.tsv")
            if not os.path.exists(sub_file_path):
                sub_dataset_url = f"https://huggingface.co/datasets/mm-eval/VLMEvalKit/resolve/main/{sub_dataset_name}.tsv"
                download_tsv(sub_dataset_url, sub_file_path)
            sub_dataframe = pd.read_csv(sub_file_path, sep='\t')
            sub_dataframe['sub_dataset'] = [sub_dataset_name] * len(sub_dataframe)
            dataframes.append(sub_dataframe)

        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df.to_csv(file_path, sep='\t', index=False, chunksize=100000)

def load_evalkit_dataset(args) -> Any:
    """Load raw data from TSV files.
    
    Returns
    -------
    Any
        Pandas DataFrame containing the dataset
    """
    # Validate environment
    dataset_dir = os.getenv('DATASET_DIR') or "./dataset"

    if args.dataset.startswith("evalkit@"):
        dataset_name = args.dataset.split("@")[-1]
        args.dataset = dataset_name

        if dataset_name not in VLMEVALKIT_DATASET_LIST:
            raise ValueError(f"Dataset {dataset_name} is not supported.")
        
        if dataset_name in ["MicroBench", "XLRS-Bench-lite", "OmniEarth-Bench", "OmniMedVQA"]:
            load_multipart_dataset(dataset_name, dataset_dir)
        elif dataset_name in ["MMMB", "MTL_MMBench_DEV"]:
            load_concat_dataset(dataset_name, dataset_dir)
        else:
            load_single_dataset(dataset_name, dataset_dir)

    return TSVDataset(args)
