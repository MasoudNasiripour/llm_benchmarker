from unittest import TestCase

import sys

sys.path.append("/benchmarker")

from llm_benchmarker.evals.metrics import calc_bleu, calc_rouge
from llm_benchmarker.data.readers._multiling import _c2dict, _read_qa
from llm_benchmarker.evals.multiling import FarsiBench
from llm_benchmarker.dataset import DatasetManager


class TestMetrics(TestCase):

    def test_bleu_score(self):
        manager = DatasetManager([FarsiBench])
        loader = manager.get_loader_by_bench(FarsiBench)
        targets = [answer["text"] if len(answer['text']) > 0 else [''] for answer in _c2dict(_read_qa(loader.get_local_path()))['answers']]

        preds = [target[0] if len(target) > 0 else '' for target in targets]

        evals = calc_bleu(preds, targets)

        self.assertIsInstance(evals, dict)
        self.assertListEqual(evals["precisions"], [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(evals["bleu"], 1.0)


    def test_rouge_score(self):
        manager = DatasetManager([FarsiBench])
        loader = manager.get_loader_by_bench(FarsiBench)
        targets = [answer["text"] if len(answer['text']) > 0 else [''] for answer in
                   _c2dict(_read_qa(loader.get_local_path()))['answers']]
        preds = [target[0] if len(target) > 0 else '' for target in targets]
        evals = calc_rouge(preds, targets)
        true_evals = {
            'rouge1': 0.0032258064516129032,
            'rouge2': 0.0,
            'rougeL': 0.0032258064516129032,
            'rougeLsum': 0.0032258064516129032
        }
        self.assertIsInstance(evals, dict)
        self.assertDictEqual(evals, true_evals)
