from .base import BaseBench, BenchmarkResults

from llm_benchmarker.config import BENCHMARK_NAME_MMLU
from llm_benchmarker.evals.metrics import calc_accuracy


class MMLUBench(BaseBench):
    """This class used for implmenting the measuing Massive Multitask Language Understanding(MMMLU) Benchmark"""

    def __init__(self):
        super().__init__()
        self.benchmark_name = MMLUBench.shared_key()

    @classmethod
    def shared_key(cls, ) -> str:
        return BENCHMARK_NAME_MMLU

    def compute(self, predictions: list[str], targets: list[list[str]]):

        try:
            predictions, targets = self._validate_inputs(predictions, targets)
            predictions = list(predictions)
            targets = list(targets)
            targets
        except Exception as e:
            return BenchmarkResults(
                benchmark_name=self.benchmark_name,
                errors= {
                    "validation": str(e)
                }
            ).to_dict()
        
        result = BenchmarkResults(benchmark_name=self.benchmark_name,)

        metrics_to_calc = [
            ('accuracy', calc_accuracy),
        ]

        for name, fn in metrics_to_calc:
            metric_results, metrics_errors = self._safe_metric_calc(name, fn, predictions, targets)

            if metric_results:
                result.metrics.update(metric_results)
            if metrics_errors:
                result.errors.update({name: metrics_errors})
        return result.to_dict()
