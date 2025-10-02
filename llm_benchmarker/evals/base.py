"""Benchmark names that are shared between benchmark classes and their information in benchmarker/config.py
These values are the key communication between benchmarks and they're datasets"""
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Callable, List

from abc import ABC, abstractmethod

from llm_benchmarker.berrors import LengthMisMatchError, \
    InvalidPredictionsForBenchmarkError, MetricCalculationError
from loguru import logger


class BaseBench(ABC):
    """A base class for All benchmarks that we need to compute"""

    def __init__(self):
        ...

    @abstractmethod
    def compute(self, predictions: list[str], targets: list[list[str]]):
        """This function will calculate the benchmark itself."""
        pass


    @classmethod
    @abstractmethod
    def shared_key(cls, ) -> str:
        return ""


    def _validate_inputs(
            self,
            predictions: list[str],
            targets: list[list[str]]
    ) -> Tuple[list[str], list[list[str]]]:
        """Validate and clean the inputted data"""
        if not len(predictions) == len(targets):
            raise LengthMisMatchError(
                f"Length MisMatch"
            )

        if not predictions:
            raise InvalidPredictionsForBenchmarkError("Prediction list is empty.")

        valid_pairs = [
            (pred, targ)
            for pred, targ in zip(predictions, targets)
            if pred is not None and isinstance(pred, str) and pred.strip()
        ]

        num_filtered = len(predictions) - len(valid_pairs)
        if num_filtered > 0:
            logger.warning(f"Filtered {num_filtered} invalid predictions")

        if not valid_pairs:
            raise InvalidPredictionsForBenchmarkError("No Valid predictions after validating.")

        return zip(*valid_pairs)


    def _safe_metric_calc(
            self,
            metric_name: str,
            metric_fn: Callable,
            predictions: List[str],
            targets: List[List[str]]
    ):
        """Safely calculates the metric with error handling"""
        try:
            result = metric_fn(predictions, targets)
            return result, None
        except Exception as e:
            logger.error(f"{metric_name} calculation failed: {e}")
            return None, str(e)


    def __str__(self):
        return f"{self.__class__.__name__}"


@dataclass
class BenchmarkResults:
    """Structured benchmark results"""
    benchmark_name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    meta_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {self.benchmark_name: {
            **self.metrics,
            **{'error_' + k: v for k, v in self.errors.items()},
        }}


class AggregatedResults:
    """Aggregate results of type BenchmarkResults"""

    results: List[BenchmarkResults] = field(default_factory=list)

    def add(self, result: BenchmarkResults):
        """add new result to current list of results"""
        self.results.append(result)

    def to_dict(self):
        output = {}
        for bresult in self.results:
            output.update(bresult.to_dict())
        return output
