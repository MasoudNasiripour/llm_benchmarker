"""This module is about evaluating models for Science Knowledge"""

from .base import BaseBench

from llm_benchmarker.config import BENCHMARK_NAME_MOLECULENET

# class ChemistryBench(BaseBench):
#
#     def __init__(self):
#         super().__init__()
#
#     @classmethod
#     def shared_key(cls, ) -> str:
#         return BENCHMARK_NAME_MOLECULENET
#
#     def compute(self, predictions: list[str], targets: list[list[str]]):
#         return self.scores.as_dict()