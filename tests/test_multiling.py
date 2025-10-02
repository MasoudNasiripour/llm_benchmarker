import numpy as np
from unittest import TestCase

import sys

sys.path.append("/benchmarker")

from llm_benchmarker.evals.multiling import FarsiBench
from llm_benchmarker.dataset import DatasetManager

from llm_benchmarker.data.readers._multiling import _c2dict, _read_qa

class TestPersianQA(TestCase):
    def test_farsi_benchmark(self):
        manager = DatasetManager([FarsiBench])
        loader = manager.get_loader_by_bench(FarsiBench)
        targets = [answer["text"] if len(answer['text']) > 0 else [''] for answer in
                   _c2dict(_read_qa(loader.get_local_path()))['answers']]
        preds = [target[0] if len(target) > 0 else '' for target in targets]
        bench = FarsiBench()
        result = bench.compute(preds,
                               targets)
        true_evals = {
            'PersianQA': {
                'f1_score': 100.0,
                'exact_match': 100.0,
                'bleu': 1.0,
                'precisions': [1.0, 1.0, 1.0, 1.0],
                'brevity_penalty': 1.0,
                'length_ratio': 1.5680875141456054,
                'translation_length': 4157,
                'reference_length': 2651,
                'rouge1': np.float64(0.004608294930875576),
                'rouge2': np.float64(0.0),
                'rougeL': np.float64(0.004608294930875576),
                'rougeLsum': np.float64(0.004608294930875576)}}
        self.assertDictEqual(true_evals, result)