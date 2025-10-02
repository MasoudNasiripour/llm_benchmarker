import json
from pathlib import Path
from loguru import logger
from collections import OrderedDict

from llm_benchmarker.data.readers import prompts
from llm_benchmarker.events.decorators import slot
from llm_benchmarker.config import BENCHMARK_NAME_PERSIAN_QA


def _c2dict(ds):
    """ convert dataset to orderdict """
    return OrderedDict([('answers', [i['answers'] for i in ds]),
                        ('context', [i['context'] for i in ds]),
                        ('question', [i['question'] for i in ds])])


def _read_qa(path):
    """
    this read dataset from JSON files like SQuAD2.0
    you can use this function for loading train and test file (even SQuAD2.0)
    """
    ds = []
    with open(Path(path), encoding="utf-8") as f:
        squad = json.load(f)
    for example in squad["data"]:
        title = example.get("title", "").strip()
        for paragraph in example["paragraphs"]:
            for qa in paragraph["qas"]:
                answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                answers = [answer["text"].strip() for answer in qa["answers"]]
                ds.append({
                    "title": title,
                    "context": paragraph["context"].strip(),
                    "question": qa["question"].strip(),
                    "id": qa["id"],
                    "answers": {
                        "answer_start": answer_starts,
                        "text": answers}, })
    return ds


@slot(BENCHMARK_NAME_PERSIAN_QA)
def persian_qa_dataset_loader(dataset_path: str):
    logger.debug(f"Loading the {BENCHMARK_NAME_PERSIAN_QA} dataset...")
    dataset = _c2dict(_read_qa(dataset_path))
    questions, contexts, answers = dataset['question'], dataset['context'], dataset['answers']
    prompt_out = []
    answers_out = []
    for qst, ctx, ans in zip(questions, contexts, answers):
        if qst == '' or ctx == '' or ans == '':
            continue
        prompt_out.append(prompts.USR_PROMPT_PERSIAN_QA.format(ctx, qst))
        answers_out.append(ans)
    logger.debug(f"Dataset {BENCHMARK_NAME_PERSIAN_QA} is loaded.")
    return prompts.SYS_PROMPT_PERSIAN_QA, prompt_out, answers_out
