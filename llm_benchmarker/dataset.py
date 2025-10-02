import os
from pathlib import Path

from typing import Union, Type, Callable, Any, Tuple
from loguru import logger

from llm_benchmarker.events.decorators import signal
from llm_benchmarker.config import LOAD_TYPE_LOCALLY, LOAD_TYPE_HUB
from llm_benchmarker.utils import backend2func,\
      get_benchmark_config as bench2dataset,\
      mkdires_if_not_exists


class BenchDatasetLoader:
    """This will manage the dataset for a benchmark"""
    def __init__(self, dataset_hub_path, dataset_local_path, __dataset__, local_load_func):
        """Create a BenchDatasetLoader for managing a dataset with __dataset__ key.
        :param dataset_hub_path: string path to the dataset in the remote repo.
        :param dataset_local_path: string path to the dataset in the disk.
        :param __dataset__: shared key between this loader and benchmark.
        :param local_load_func: function that load dataset from disk. if this was ``None`` then loader will look
            for this in ``slot`` function"""
        self._hub_path = dataset_hub_path
        self._local_path = dataset_local_path
        self.__dataset__ = __dataset__
        self._local_load_func = local_load_func

    def load_from_disk(self):
        """load dataset from disk and return it"""
        @signal(self.__dataset__)
        def path():
            return self.get_local_path()
        if self._local_load_func is None:
            return path()
        else:
            return self._local_load_func(self.get_local_path())

    def get_local_path(self) -> str:
        """give the local path of the related dataset"""
        return self._local_path

    def get_hub_path(self) -> str:
        """give the hub path of the related dataset"""
        return self._hub_path


    @classmethod
    def _extract_info(
            cls,
            shared_key: str
    ) -> Union[Tuple[str|None, str|None, Callable[[str, str], bool]|None, dict[str, Any]], None]:
        """extract information of a based on benchmark name
        :param shared_key: the name of the benchmark.
        :returns: if all information was correct and complete then hub path of dataset, local path of dataset and it's
            load(local/remote) functions are returned as a tuple."""
        logger.debug("Extract dataset information's...")
        info = bench2dataset(shared_key)
        load_funcs = backend2func(info.get("backend"))
        downloader = load_funcs.get(LOAD_TYPE_HUB)
        downloader_kwargs = info.get("download_kwargs", None)
        hub_path = info.get("path")
        local_path = info.get("local_dir")
        mkdires_if_not_exists(local_path)
        if hub_path in ["", None] or local_path in ["", None]:
            raise Exception("Local path or Remote path to the dataset is not specified")
            # logger.debug("Local path or Remote path to the dataset is not specified")
            return None
        if downloader is None:
            logger.debug(f"Download function for downloading dataset from {hub_path} is not specified")
            return None
        return hub_path, local_path, load_funcs, downloader_kwargs

    @classmethod
    def load(cls, shared_key: str):
        """By this we create an BenchDatasetLoader instance but with some preprocess
        :param btype: a benchmark type
        :returns: an object of BenchDatasetLoader"""
        logger.debug(f"Loading the {shared_key} dataset into disk if it doesn't exists")
        check_dataset = cls._extract_info(shared_key)
        if check_dataset is None:
            logger.debug(f"Provided dataset info for {shared_key} is not correct")
            raise Exception(f"Provided dataset info for {shared_key} is not correct")
        dataset_path, destination_path, load_funcs, downloader_kwargs = check_dataset
        self = cls(dataset_path, destination_path, shared_key, load_funcs.get(LOAD_TYPE_LOCALLY))
        self.__dataset__ = shared_key
        if not os.path.exists(destination_path):
            if not load_funcs.get(LOAD_TYPE_HUB)(dataset_path, destination_path, **downloader_kwargs):
                logger.debug(f"Failed to download dataset from {dataset_path}")
                raise Exception(f"Failed to download dataset from {dataset_path}")
            else:
                logger.debug(f"{dataset_path} loaded in {destination_path}")
                return self
        logger.debug(f"{dataset_path} is already existed in {destination_path}")
        return self


class DatasetManager:

    def __init__(self, btypes: list):
        """Create a list of BenchDatasetLoader and manage it
        :param btypes: list of benchmark types. like [FarsiBench, ]"""
        self._loaders = {}
        for b in btypes:
            self._loaders[b.shared_key()] = BenchDatasetLoader.load(b.shared_key())

    def get_loaders(self) -> dict[str, BenchDatasetLoader]:
        """Get a dictionary with key benchmark name(__dataset__ shared key) as and BenchDatasetLoader as value"""
        return self._loaders

    def get_loader_by_bench(self, btype) -> BenchDatasetLoader:
        """Gives a BenchDatasetLoader from specified Benchmark type."""
        return self._loaders.get(btype.shared_key())
