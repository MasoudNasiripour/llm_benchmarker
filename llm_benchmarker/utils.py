import os.path
import pathlib
import inspect
import requests
import importlib.util

from typing import Union, Callable, Any, List

from loguru import logger

from datasets import load_dataset
from llm_benchmarker.events.handlers import EventHandler

RESPONSE_STATUS_CODE_SUCCEED = 200


def github_url_to_raw_github_url(github_url: Union[str, list]) -> list:
    """Convert github url into raw github url for downloading a file more clear."""
    outcomes = []

    if type(github_url) == str:
        github_url = [github_url, ]

    for url in github_url:

        if not url.startswith("https://github.com") and not url.startswith("https://raw.githubusercontent.com"):
            raise Exception(f"Input url must be from github not {url}")

        if url.startswith("https://github.com"):
            outcomes.append(url \
                            .replace("https://github.com", "https://raw.githubusercontent.com") \
                            .replace("blob", "refs/heads"))
        else:
            outcomes.append(url)

    return outcomes


def load_file_from_web(url: str, destination_path: str, **kwargs) -> bool:
    response = requests.get(url)
    if response.status_code == RESPONSE_STATUS_CODE_SUCCEED:
        with open(destination_path, "wb") as file:
            file.write(response.content)
            return True
    else:
        return False


def load_from_no_where(url: str, destination_path: str, *args, **kwargs) -> bool:
    return True


def load_from_github(url: str, destination_path: str, **kwargs) -> bool:
    """With this you can download a file from github. like a json file that contains your data"""
    url = github_url_to_raw_github_url(url)[0]
    return load_file_from_web(url, destination_path, **kwargs)


def load_from_hf(path: str, destination_path: str, **kwargs):
    """download dataset from Huggingface"""
    try:
        load_dataset(path, **kwargs).save_to_disk(destination_path)
        return True
    except Exception as e:
        logger.debug(e)
        return False


def backend2func(backend: str) -> dict[str, Callable]:
    from llm_benchmarker.config import BACKEND_STR2FUNC
    return BACKEND_STR2FUNC.get(backend)

def mkdires_if_not_exists(dir: Union[str, pathlib.Path]):
    local_path = pathlib.Path(dir) if type(dir) is str else pathlib.Path(dir)
    if not os.path.exists(local_path):
        if os.path.isdir(local_path):
            os.makedirs(local_path, exist_ok=True)
        else:
            os.makedirs(local_path.parent, exist_ok=True)

def list_mkdires_if_not_exists(dirs: List[Union[str, pathlib.Path]]):
    for dir in dirs:
        mkdires_if_not_exists(dir)


def get_benchmark_config(benchmark_name: str, as_class=False) -> Union[dict, object]:
    from llm_benchmarker.config import DATASETS_PER_BENCH, BenchmarkConfigManager
    if as_class:
        return BenchmarkConfigManager().get_config(benchmark_name)
    return DATASETS_PER_BENCH.get(benchmark_name)


def list_module_in_path(dir_path: Union[str, pathlib.Path]):
    results = []
    if os.path.isdir(dir_path):
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if file_name.endswith(".py"):
                file_name = file_name[:-3]  # remove .py suffix from file name
                results.append(
                    (file_path, file_name)
                )
    return results


def list_module_in_list_path(dir_pathes: Union[list[str], list[pathlib.Path]]):
    return [module for path in dir_pathes for module in list_module_in_path(path)]


def list_slots_in_module(file_name: str, file_path: Union[str, pathlib.Path]) -> list[Callable]:
    _slots = []
    spec = importlib.util.spec_from_file_location(file_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj):
            if hasattr(obj, f'_slot_decorated'):
                _slots.append([obj, obj.__dataset__])
    return _slots


def list_slots_in_list_pathes(dir_pathes: Union[list[str], list[pathlib.Path]]):
    modules = list_module_in_list_path(dir_pathes)
    return [(slot__dataset__, slot_fn) for file_path, file_name in modules for slot_fn, slot__dataset__ in
            list_slots_in_module(file_name, file_path)]


def load_slots(slot_pathes: Union[list[str], list[pathlib.Path]]):
    handler = EventHandler()
    logger.debug("Finding slot functions...")
    list_slots_in_list_pathes(slot_pathes)
    logger.debug("Slot's were found successfully")
