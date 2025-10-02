from llm_benchmarker.config import BENCHMARK_NAME_MOLECULENET
from llm_benchmarker.events.decorators import slot

# import deepchem as dc
# from deepchem.feat import CircularFingerprint

# @slot(BENCHMARK_NAME_MOLECULENET)
# def deepchem_load_from_disk(dataset_path: str):
#     featurizer = CircularFingerprint(size=1024)
#     loader = dc.data.CSVLoader(
#         tasks=["measured log solubility in mols per litre"],
#         feature_field="smiles",
#         featurizer=featurizer
#     )
#
#     splitter = dc.splits.RandomSplitter()
#     dataset = loader.featurize(dataset_path)
#     _, _, test_dataset = splitter.train_valid_test_split(dataset)
#     print(test_dataset.X[:5])
#     print(test_dataset.y[:5])
