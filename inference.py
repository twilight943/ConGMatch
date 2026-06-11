import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, precision_score, recall_score, roc_auc_score

from candidate_invariance import build_candidate_batch_from_pyg_pairs


def get_candidate_graph_cache(args):
    if not getattr(args, "candidate_graph_cache", False):
        return None
    cache = getattr(args, "_candidate_graph_cache", None)
    if cache is None:
        cache = {}
        setattr(args, "_candidate_graph_cache", cache)
    return cache


def candidate_prediction_signal(logits, args):
    if getattr(args, "candidate_loss", "mse") == "mse":
        return torch.sigmoid(logits)
    return logits


def evaluator(pred, raw_pred, labels):
    auroc = roc_auc_score(labels, raw_pred)
    if auroc < 0.5:
        raw_pred = -raw_pred
        auroc = roc_auc_score(labels, raw_pred)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1], normalize="all").ravel()
    return {
        "auroc": auroc,
        "acc": accuracy_score(labels, pred),
        "prec": precision_score(labels, pred, zero_division=0),
        "recall": recall_score(labels, pred, zero_division=0),
        "avg_prec": average_precision_score(labels, raw_pred),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


@torch.no_grad()
def get_preds(model, dataset, device, args):
    preds = []
    raw_preds = []
    all_labels = []
    runtime = 0.0
    eval_batch_size = args.candidate_eval_batch_size if args.candidate_eval_batch_size > 0 else args.batch_size
    batches = dataset.iter_batches(eval_batch_size) if hasattr(dataset, "iter_batches") else (
        dataset.get_batch(eval_batch_size) for _ in range(args.eval_batches)
    )
    for _, g1, g2, g3 in batches:
        batch_size = len(g1)
        labels = torch.tensor(([1] * batch_size + [0] * batch_size), dtype=torch.long, device=device)
        start_time = time.time()
        candidate_batch = build_candidate_batch_from_pyg_pairs(
            g2 + g3,
            g1 + g1,
            mode="association",
            maxloop=args.candidate_maxloop,
            device=device,
            max_pairs=args.association_max_pairs,
            max_edges=args.association_max_edges,
            cache=get_candidate_graph_cache(args),
            cache_max_entries=args.candidate_graph_cache_max_entries,
        )
        logits = model.candidate_logits(candidate_batch).view(-1)
        raw_pred = candidate_prediction_signal(logits, args).view(-1)
        pred = model.predictor(raw_pred.unsqueeze(-1)).argmax(dim=-1)
        runtime += time.time() - start_time
        preds.append(pred.view(-1))
        raw_preds.append(raw_pred.view(-1))
        all_labels.append(labels.view(-1))
    pred = torch.cat(preds, dim=-1).detach().cpu().numpy()
    raw_pred = torch.cat(raw_preds, dim=-1).detach().cpu().numpy()
    labels = torch.cat(all_labels, dim=-1).detach().cpu().numpy()
    return evaluator(pred, raw_pred, labels), runtime


@torch.no_grad()
def validation(model, valset, device, args):
    model.eval()
    return get_preds(model, valset, device, args)


@torch.no_grad()
def test(model, testset, device, args):
    model.eval()
    return get_preds(model, testset, device, args)
