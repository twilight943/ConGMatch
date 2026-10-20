import time

import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from candidate_graph import build_candidate_batch


def _synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_metrics(predictions, scores, labels):
    auroc = roc_auc_score(labels, scores)
    metric_scores = scores
    if auroc < 0.5:
        metric_scores = -scores
        auroc = roc_auc_score(labels, metric_scores)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, normalize="all").ravel()
    return {
        "auroc": float(auroc),
        "acc": float(accuracy_score(labels, predictions)),
        "prec": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "avg_prec": float(average_precision_score(labels, metric_scores)),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def _cache(args):
    if not args.candidate_graph_cache:
        return None
    if not hasattr(args, "_candidate_graph_cache"):
        args._candidate_graph_cache = {}
    return args._candidate_graph_cache


@torch.no_grad()
def evaluate(model, dataset, device, args):
    model.eval()
    scores = []
    predictions = []
    labels = []
    build_time = 0.0
    forward_time = 0.0
    pair_count = 0.0
    edge_count = 0.0
    sample_count = 0
    positive_empty = 0
    negative_empty = 0
    batch_size = args.candidate_eval_batch_size or args.batch_size
    if hasattr(dataset, "iter_batches"):
        batches = dataset.iter_batches(batch_size)
    else:
        batches = (dataset.get_batch(batch_size) for _ in range(args.eval_batches))
    for _, targets, positives, negatives in batches:
        batch_labels = torch.tensor(
            [1] * len(positives) + [0] * len(negatives), device=device
        )
        _synchronize(device)
        start = time.perf_counter()
        candidate_batch = build_candidate_batch(
            positives + negatives,
            targets + targets,
            device,
            maxloop=args.candidate_maxloop,
            max_pairs=args.association_max_pairs,
            max_edges=args.association_max_edges,
            cache=_cache(args),
            cache_max_entries=args.candidate_graph_cache_max_entries,
        )
        _synchronize(device)
        build_time += time.perf_counter() - start
        empty = candidate_batch.has_empty_candidate_set.bool().view(-1)
        positive_empty += int(empty[: len(positives)].sum().item())
        negative_empty += int(empty[len(positives) :].sum().item())
        pair_count += float(candidate_batch.association_pair_count.sum().item())
        edge_count += float(candidate_batch.association_edge_count.sum().item())
        sample_count += int(batch_labels.numel())
        _synchronize(device)
        start = time.perf_counter()
        batch_scores = torch.sigmoid(model.candidate_logits(candidate_batch)).view(-1)
        batch_predictions = (batch_scores >= args.prediction_threshold).long()
        _synchronize(device)
        forward_time += time.perf_counter() - start
        scores.append(batch_scores.cpu())
        predictions.append(batch_predictions.cpu())
        labels.append(batch_labels.cpu())
    scores = torch.cat(scores).numpy()
    predictions = torch.cat(predictions).numpy()
    labels = torch.cat(labels).numpy()
    result = evaluate_metrics(predictions, scores, labels)
    denominator = max(sample_count, 1)
    positive_count = max(int((labels == 1).sum()), 1)
    negative_count = max(int((labels == 0).sum()), 1)
    result.update(
        {
            "candidate_build_time": build_time,
            "model_forward_time": forward_time,
            "runtime": build_time + forward_time,
            "runtime_avg": (build_time + forward_time) / denominator,
            "association_pairs_avg": pair_count / denominator,
            "association_edges_avg": edge_count / denominator,
            "positive_empty_rate": positive_empty / positive_count,
            "negative_empty_rate": negative_empty / negative_count,
            "num_eval_samples": sample_count,
        }
    )
    return result
