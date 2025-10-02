from collections import Counter
import re


def _cleaner(text):
    return re.sub('\u200c', " ", text).strip()


def _f1_score(prediction, ground_truth):
    prediction_tokens = _cleaner(prediction)
    ground_truth_tokens = _cleaner(ground_truth)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def _exact_match_score(prediction, ground_truth):
    return (_cleaner(prediction) == _cleaner(ground_truth))


def _metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def calc_bleu(predictions: list[str], targets: list[list[str]]) -> dict:
    from evaluate import load
    metric = load("bleu")
    return metric.compute(predictions=predictions, references=targets)


def calc_rouge(predictions: list[str], targets: list[list[str]]) -> dict:
    from evaluate import load
    metric = load("rouge")
    scores = metric.compute(predictions=predictions, references=targets)
    return scores


def f1_score_exact_match(predictions: list[str], targets: list[list[str]]) -> dict:
    f1 = exact_match = total = 0
    for ground_truths, prediction in zip(targets, predictions):
        total += 1
        exact_match += _metric_max_over_ground_truths(_exact_match_score, prediction, ground_truths)
        f1 += _metric_max_over_ground_truths(_f1_score, prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {"f1_score": f1, "exact_match": exact_match}


def calc_accuracy(predictions: list[str], targets: list[list[str]]) -> dict:
    correct = 0
    total = len(predictions)
    for ground_truth, prediction in zip(targets, predictions):
        pred_norm = str(prediction).strip().lower()
        ground_truth_norms = [str(g).strip().lower() for g in ground_truth]
        if pred_norm in ground_truth_norms:
            correct += 1
    return {"accuracy": correct / total if total > 0 else 0.0}
