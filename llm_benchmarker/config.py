import os
import sys
import dataclasses
from typing import Union, Optional, Literal


from loguru import logger

from pathlib import Path

from llm_benchmarker.utils import load_from_github,\
        load_from_hf,\
        load_from_no_where,\
        load_file_from_web,\
        list_mkdires_if_not_exists


"""Directories in the project"""
DATASET_DIR_NAME = "data"
BASE_PATH = Path(__file__).parent  # path to the benchmarker directory.
DATA_DIR_PATH = BASE_PATH / DATASET_DIR_NAME  # path to the benchmarker/data directory.

"""Keys for Dataset providers. For now we support Huggingface and Github(one file for now)"""
BACKEND_GITHUB = "URL"
BACKEND_HUGGINGFACE = "HF"
BACKEND_CUSTOM_NO_DIRECT_DOWNLOAD = "BCNDD"
BACKEND_CUSTOM_DIRECT_DOWNLOAD_UNIFILE = "BCDDU"

"""Types of the loading dataset into benchmarker. These keys will be used in other parts of the project"""
LOAD_TYPE_LOCALLY = "LOC"
LOAD_TYPE_HUB = "HUB"

"""We use loguru for logging and logger.loguru.critical instead of Exceptions"""
# logger.add(lambda _: sys.exit(1), level="CRITICAL")
logger.remove()
logger.add(sys.stderr, level="INFO")

"""A configuration for How loading dataset locally or from hub per each backend"""
BACKEND_STR2FUNC = {
    BACKEND_HUGGINGFACE: {
        LOAD_TYPE_HUB: load_from_hf,
        LOAD_TYPE_LOCALLY: None,
    },

    BACKEND_GITHUB: {
        LOAD_TYPE_HUB: load_from_github,
        LOAD_TYPE_LOCALLY: None
        # For GitHub downloaded dataset, we may have to load from different ways. this is why we used event handler
    },

    BACKEND_CUSTOM_NO_DIRECT_DOWNLOAD: {
        LOAD_TYPE_HUB: load_from_no_where,
        LOAD_TYPE_LOCALLY: None
    },

    BACKEND_CUSTOM_DIRECT_DOWNLOAD_UNIFILE: {
        LOAD_TYPE_HUB: load_file_from_web,
        LOAD_TYPE_LOCALLY: None
    }
}


"""these are the name of bench modules in ```evals``` directory and the their name in data directory"""
BENCH_CATEGORY_MULTILING = "multiling"
BENCH_CATEGORY_SCIENCE = "science"
BENCH_CATEGORY_LANG = "lang"
BENCH_CATEGORY_AGENTIC = "agentic"

"""List of local path to each Dataset category. this is useful for specifying the absolute path of the datasets."""
BENCH_CATEGORY_DATASET_LOCAL_DIR = {
    BENCH_CATEGORY_MULTILING: DATA_DIR_PATH / BENCH_CATEGORY_MULTILING,
    BENCH_CATEGORY_SCIENCE: DATA_DIR_PATH / BENCH_CATEGORY_SCIENCE,
    BENCH_CATEGORY_LANG: DATA_DIR_PATH / BENCH_CATEGORY_LANG,
    BENCH_CATEGORY_AGENTIC: DATA_DIR_PATH / BENCH_CATEGORY_AGENTIC,
}


# for dir in BENCH_CATEGORY_DATASET_LOCAL_DIR.values():
#     if not os.path.exists(dir):
#         os.makedirs(dir)

list_mkdires_if_not_exists(BENCH_CATEGORY_DATASET_LOCAL_DIR.values())


"""These are the names of the Benchmarks. these are the shared key between dataset loaders, benchmarks and slot functions"""
BENCHMARK_NAME_PERSIAN_QA = "PersianQA"
BENCHMARK_NAME_MMLU = "MMLU"
BENCHMARK_NAME_MOLECULENET = "MoleculeNet"



"""datasets used for each benchmark. they are going to be downloaded in the first place"""
DATASETS_PER_BENCH = {
    BENCHMARK_NAME_PERSIAN_QA: {
        "backend": BACKEND_GITHUB,
        "category": BENCH_CATEGORY_MULTILING,
        "path": "https://github.com/sajjjadayobi/PersianQA/blob/main/dataset/pqa_test.json",
        "local_dir": BENCH_CATEGORY_DATASET_LOCAL_DIR.get(BENCH_CATEGORY_MULTILING) / Path("persianQA") /
                                 Path("pqa_test.json"),
        "download_kwargs": {}
    },

    BENCHMARK_NAME_MMLU: {
        "backend": BACKEND_HUGGINGFACE,
        "category": BENCH_CATEGORY_LANG,
        "path": "cais/mmlu",
        "local_dir": BENCH_CATEGORY_DATASET_LOCAL_DIR.get(BENCH_CATEGORY_LANG) / Path("MMLU"),
        "download_kwargs": {"name": "all"}
    },

    # BENCHMARK_NAME_MOLECULENET: {
    #     "backend": BACKEND_CUSTOM_DIRECT_DOWNLOAD_UNIFILE,
    #     "category": BENCH_CATEGORY_SCIENCE,
    #     "path": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
    #     "local_dir": path_concat(BENCH_CATEGORY_DATASET_LOCAL_DIR.get(BENCH_CATEGORY_SCIENCE), Path("DeepChem")),
    #     "download_kwargs": {}
    # },
}


"""Configs as classes"""
@dataclasses.dataclass(frozen=True)
class BenchmarkConfig:
    backend: str
    category: str
    path: str
    local_dir: Union[str, Path]


class BenchmarkConfigManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._configs = None
        return cls._instance

    @property
    def configs(self) -> dict[str, BenchmarkConfig]:
        if self._configs is None:
            self._configs = self._load_configs()
        return self._configs


    def _load_configs(self):
        configs = {}
        for name, raw_config in DATASETS_PER_BENCH.items():
            try:
                configs[name] = BenchmarkConfig(**raw_config)
            except Exception as e:
                logger.debug(f"Invalid config for {name}: {e}")
        return configs

    def get_config(self, benchmark_name: str) -> Optional[BenchmarkConfig]:
        return self.configs.get(benchmark_name)


"""Path's to directories contains modules with slot functions"""
SLOT_DIR_PATH = [
    DATA_DIR_PATH / Path("readers"),
]

GENERATOR_FUNC_KEY = "gen_func"
CHAT_TEMPLATE_FUNC = "prompt_formatter_func"


