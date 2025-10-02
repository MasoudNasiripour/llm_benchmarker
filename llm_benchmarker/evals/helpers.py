from collections import OrderedDict
from pathlib import Path
import json

def c2dict(ds):
    """ convert dataset to orderdict """
    return OrderedDict([('answers', [i['answers'] for i in ds]), 
                        ('context', [i['context'] for i in ds]), 
                        ('question', [i['question'] for i in ds])])


def read_qa(path):
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
                      "text": answers},})
    return ds

from collections import Counter
import re

def cleaner(text):
    return re.sub('\u200c', " ", text).strip()

def f1_score(prediction, ground_truth):
    prediction_tokens = cleaner(prediction)
    ground_truth_tokens = cleaner(ground_truth)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return (cleaner(prediction) == cleaner(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    
    return max(scores_for_ground_truths)


def evaluate(gold_answers, predictions):
    f1 = exact_match = total = 0
    for ground_truths, prediction in zip(gold_answers, predictions):
        total += 1
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


def chat_v2(prompt="Explain quantum mechanics clearly and concisely.", context=""):
    messages = [
        {"role": "system", "content": "You are an Persian AI assistant. Your job is filling the Answer part of the given prompt. Reasoning: low."},
        {"role": "user", "content": f"براساس گزاره تهیه شده به سوال پاسخ کوتاه و دقیق بده.\nگزاره:{context}\nسوال:{prompt}\nپاسخ:"},
    ]
    
    outputs = pipe(
        messages,
        max_new_tokens=4096,
    )
    return outputs[0]["generated_text"][-1]

def generate(message, context):
    outputs = chat_v2(message, context)["content"]
    return {"text": outputs}