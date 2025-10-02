from typing import Tuple, Dict
from loguru import logger
from .base import BaseBench, BenchmarkResults
from .metrics import f1_score_exact_match, calc_bleu, calc_rouge

from llm_benchmarker.config import BENCHMARK_NAME_PERSIAN_QA


class FarsiBench(BaseBench):
    """This class used for implementing the measuing Persian Language Benchmark"""

    def __init__(self):
        super().__init__()
        self.benchmark_name = FarsiBench.shared_key()

    @classmethod
    def shared_key(cls, ) -> str:
        return BENCHMARK_NAME_PERSIAN_QA

    def compute(self, predictions: list[str], targets: list[list[str]]) -> Dict:

        # validate the inputted data
        try:
            predictions, targets = self._validate_inputs(predictions, targets)
            predictions = list(predictions)
            targets = list(targets)
        except Exception as e:
            return BenchmarkResults(
                benchmark_name=self.benchmark_name,
                errors= {
                    "validation": str(e)
                }
            ).to_dict()

        result = BenchmarkResults(benchmark_name=self.benchmark_name,)

        metrics_to_calc = [
            ('f1', f1_score_exact_match),
            ('bleu', calc_bleu),
            ('rouge', calc_rouge)
        ]

        for name, fn in metrics_to_calc:
            metric_results, metrics_errors = self._safe_metric_calc(name, fn, predictions, targets)

            if metric_results:
                result.metrics.update(metric_results)
            if metrics_errors:
                result.errors.update({name: metrics_errors})
        return result.to_dict()
