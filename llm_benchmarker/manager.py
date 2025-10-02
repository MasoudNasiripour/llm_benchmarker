from loguru import logger
import inspect
import pandas as pd

from typing import Type, Callable


from llm_benchmarker.evals import BaseBench
from llm_benchmarker.dataset import DatasetManager
from llm_benchmarker.config import GENERATOR_FUNC_KEY, CHAT_TEMPLATE_FUNC, SLOT_DIR_PATH
from llm_benchmarker.pipelines import ModelPipeline, BenchmarkPipeline
from llm_benchmarker.utils import load_slots


class BenchManager:
    def __init__(self, benchmark_model_conf: dict[Type[BaseBench], dict[str, Callable]]):
        """Manage requested benchmarks
        :param benchmark_model_conf: a dictionary of benchmarks that keys are Benchmarks in the ``evals``
            and values are dictionary as well. the dictionary in the value must have two (key, value) pairs.
            the first key must be ``"gen_func"`` and it's value is a function that gets list of prompts as strings
            and return a list of strings. this function run your model against inputed prompts. in second key,value pair
            set key as ``"prompt_formatter_func"`` and it's value must be a function with two inputs, one is for
            ``system prompt`` as ``str`` and another one is a list of strings. this function gets a syste, prompt
            and a string list of prompts then formmated the prompts for your model and return the formatted as list
            of ``Any``.

        >>> from benchmarker.evals import FarsiBench
        >>> import benchmarker.manager.BenchManager as BenchManager
        >>> model_conf_per_bench = {
        >>> FarsiBench : {
        >>>     GENERATOR_FUNC_KEY: generation,
        >>>     CHAT_TEMPLATE_FUNC: message_format_func,
        >>>     },
        >>> }
        >>> benchmarks = BenchManager(model_conf_per_bench=benchmark_model_conf)
        >>> benchmarks.run()
        ... {'PersianQA': {
        ...     'f1_score': 0.12886794355824202,
        ...     'exact_match': 0.0,
        ...     'bleu': 0.0,
        ...     'precisions': [0.0, 0.0, 0.0, 0.0],
        ...     'brevity_penalty': 1.0,
        ...     'length_ratio': 223.6290322580645,
        ...     'translation_length': 207975,
        ...     'reference_length': 930,
        ...     'rouge1': np.float64(0.0),
        ...     'rouge2': np.float64(0.0),
        ...     'rougeL': np.float64(0.0),
        ...     'rougeLsum': np.float64(0.0)}
        ... }
        """
        self._btypes: list[Type[BaseBench]] = list(set(benchmark_model_conf.keys())) # btype stands for (B)enchmark (TYPE), is a list of requested benchmarks
        self._benchmark_model_conf = benchmark_model_conf # This is the inputted dictionary.
        self._check_model_benchmark_conf()
        logger.debug("Check datasets per benchmark")
        self.loader_manager = DatasetManager(self._btypes)
        logger.debug("Loading the benchmarks")
        load_slots(SLOT_DIR_PATH)
        self.benchmarks = {
           bt.shared_key(): bt() for bt in self._btypes
        }

    def _pipe_creator(self, ) -> dict[Type[BaseBench], BenchmarkPipeline]:
        """Create Benchmark pipeline for all requested benchmarks.
        :returns: a dictionary, keys are Benchmark Type and
            values are benchmark pipelines"""
        logger.debug("Create pipelines based on Model info's er benchmarks.")
        results = {}
        for bench_type, fns in self._benchmark_model_conf.items():
            results[bench_type] = BenchmarkPipeline(self.benchmarks[bench_type.shared_key()],
                                                    ModelPipeline(**fns),
                                                    self.loader_manager.get_loader_by_bench(bench_type))
        return results

    def _pipe_per_bench(self, benchmark: Type[BaseBench]) -> BenchmarkPipeline:
        """Create a benchmark pipeline for requested benchmark
        :param benchmark:``Type[BaseBench]``: The type of the benchmark. like FarsiBench
        :returns:``BenchmarkPipeline``: A benchmark pipeline will be returned
        >>> from benchmarker.evals import FarsiBench
        >>> self._pipe_per_bench(FarsiBench)"""
        return BenchmarkPipeline(self.benchmarks[benchmark.shared_key()],
                                 ModelPipeline(**self._benchmark_model_conf[benchmark]),
                                 self.loader_manager.get_loader_by_bench(benchmark.shared_key()))


    def _check_model_benchmark_conf(self, ):
        """Check the correctness of the configs that client sent. we check the ``"prompt_formatter_func"`` and
         ``"gen_func"`` functions for their inputs and their existence in input. If any information goes wrong then
         program will be stopped."""
        logger.debug("Checking the Model info provided for each benchmark")
        for b in self._btypes:
            if b not in self._benchmark_model_conf.keys():
                raise Exception(f"Provided Model info's are incomplete in {GENERATOR_FUNC_KEY} and {CHAT_TEMPLATE_FUNC} for {b.shared_key()}")
                # logger.debug(f"Provided Model info's are incomplete in {GENERATOR_FUNC_KEY} and {CHAT_TEMPLATE_FUNC} for {b.shared_key()}")
            if GENERATOR_FUNC_KEY not in self._benchmark_model_conf[b].keys() or CHAT_TEMPLATE_FUNC not in self._benchmark_model_conf[b].keys():
                raise Exception(f"Provided Model info's are incomplete in {GENERATOR_FUNC_KEY} and {CHAT_TEMPLATE_FUNC} for {b}")
                logger.debug(
                    f"Provided Model info's are incomplete in {GENERATOR_FUNC_KEY} and {CHAT_TEMPLATE_FUNC} for {b}")
            cht_fn = self._benchmark_model_conf[b].get(CHAT_TEMPLATE_FUNC, None)
            if cht_fn is None or not len(inspect.signature(cht_fn).parameters.keys()) == 2:
                raise Exception(f"Number of parameters in {CHAT_TEMPLATE_FUNC} wasn't enough for {b}")
                logger.debug(
                    f"Number of parameters in {CHAT_TEMPLATE_FUNC} wasn't enough for {b}")
            logger.debug("Correct model info.")

    def get_bench_obj_by_btype(self, btype: Type[BaseBench]) -> BaseBench:
        """Give the benchmark object by its type.
        :param btype: the type of the benchmark, like FarsiBench
        :returns: the object(instance) of the requested benchmark.
        >>> from benchmarker.evals import FarsiBench
        >>> get_bench_obj_by_btype(FarsiBench)"""
        return self.benchmarks.get(btype.shared_key())

    def run_bench_by_dataset(self, __dataset__,):
        ...

    def summary(self) -> pd.DataFrame:
        """A dataframe of the requested benchmark with their names and they're local and remote path.
        :returns: a pandas.DataFrame from this information"""
        return pd.DataFrame({
            "Benchmark Name": [bt.__name__ for bt in self._btypes],
            "Benchmark Dataset Name": [bt.shared_key() for bt in self._btypes],
            "Benchmark Dataset[remote]": [self
                                .loader_manager
                                .get_loader_by_bench(bt)
                                .get_hub_path() for bt in self._btypes],
            "Benchmark Dataset[local]": [self
                                  .loader_manager
                                  .get_loader_by_bench(bt)
                                  .get_local_path() for bt in self._btypes]
        })

    def run(self):
        """Run all requested benchmarks on your model.
        :returns: a dictionary with keys as the benchmark name and values as dictionary too.
            in dictionary value keys are metrics and values are the value of that metric for that benchmarks"""
        results = {}
        pipes_dict = self._pipe_creator()
        for btype, benchmark_pipe in pipes_dict.items():
            bout = benchmark_pipe()
            if bout is not None:
                results.update(bout)
        return results
