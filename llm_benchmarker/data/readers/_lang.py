from llm_benchmarker.events.decorators import slot
from llm_benchmarker.config import BENCHMARK_NAME_MMLU

from llm_benchmarker.data.readers.prompts import SYS_PROMPT_MMLU, USR_PROMPT_MMLU

from datasets import load_from_disk


def format_mmlu_prompt(sample):
    q = sample["question"]
    choices = sample["choices"]
    answer = sample["answer"]
    text = USR_PROMPT_MMLU.format(q, *choices)
    return text, answer


@slot(BENCHMARK_NAME_MMLU)
def MMLU_load_from_disk(dataset_path: str):
    results = load_from_disk(dataset_path)
    prompt_out = []
    answer_out = []
    for sample in results["test"]:
        prompt, answer = format_mmlu_prompt(sample)
        prompt_out.append(prompt)
        answer_out.append([answer])
    return SYS_PROMPT_MMLU, prompt_out, answer_out
