"""This Module provide us with some functionality for make benchmarks running easier with pipelines.
"""

from llm_benchmarker.evals import BaseBench
from llm_benchmarker.dataset import BenchDatasetLoader

from typing import Callable, Any


class ModelPipeline:
    """With this class a pipeline for the model will be created."""
    def __init__(self, gen_func: Callable[[Any], list[str]],
                 prompt_formatter_func: Callable[[str, list[str]], Any]):
        """A pipeline will be created base on inputted functions
        :param gen_func: The generated function of the model
        :param prompt_formatter_func: A function the formatted the input prompt"""
        self.gen_func: Callable[[Any], list[str]] = gen_func
        self.prompt_formatter: Callable[[str, list[str]], Any] = prompt_formatter_func

    def __call__(self, sys_prompt: str, prompts: list[str], batch_size) -> list[str]:
        """Runs prompt_formatter_func and gen_func in order to get output of the model with specified batch size.
        :param sys_prompt: This is a system prompt for the model.
        :param prompts: list pf prompts to be processed by the model.
        :param batch_size: Batch size for put data in it.
        :returns: a list model output w.r.t input prompts"""
        results = []
        ix = 0
        while True:
            start_ix = batch_size * ix
            end_ix = batch_size * (ix+1)
            if end_ix > len(prompts) and start_ix > len(prompts):
                break
            if end_ix > len(prompts) > start_ix:
                results += self._run(sys_prompt, prompts[start_ix:])
            else:
                results += self._run(sys_prompt, prompts[start_ix:end_ix])
            ix += 1
        return results

    def _run(self, sys_prompt: str, batch_prompt: list[str]) -> list[str]:
        """Run pipeline on a batch of prompts
        :param sys_prompt: This is a system prompt for the model.
        :param batch_prompt: list pf prompts to be processed by the model.
        :returns: a list model output w.r.t input prompts"""
        formatted_prompts = self.prompt_formatter(sys_prompt, batch_prompt)
        return self.gen_func(formatted_prompts)


class BenchmarkPipeline:
    """With this class a pipeline for the Benchmark will be created."""
    def __init__(self, bobj: BaseBench, model_pipeline: ModelPipeline, dataset_loader: BenchDatasetLoader):
        """Create a pipeline for a benchmark base on benchmark object(not type),
         the pipeline of the model and dataset loader
        :param bobj: stands for (B)enchmark (OBJ)ect, an object of the requested benchmark.
        :param model_pipeline: a model pipeline that made from ``ModelPipeline``.
        :param dataset_loader: an instance of dataset loader related to the benchmark type"""
        self.bobj = bobj
        self._model_pipeline = model_pipeline
        self._dataset_loader = dataset_loader

    def __call__(self, *args, **kwargs):
        """Runs the ```self._run``` and return its output."""
        return self._run()

    def _run(self):
        """this will run the benchmark pipeline with defined attribute. system prompt, prompts and targets loaded by
        dataset loader of the related benchmark then they passed into model pipeline and finally the results passed
        to the benchmark to compute the benchmarks
        :returns: a dictionary of the calculated metrics in benchmarks"""
        system_prompt, prompts, targets = self._dataset_loader.load_from_disk()[self.bobj.shared_key()]
        predictions = self._model_pipeline(system_prompt, prompts, 50)
        return self.bobj.compute(predictions, targets)
