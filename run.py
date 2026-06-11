import argparse
import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.nn import CrossEntropyLoss

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

from dataset import GRFTripletDataset, SCDataset, build_grf_label_to_index, load_tu_dataset
from inference import test, validation
from models import NC
from candidate_invariance import build_candidate_batch_from_pyg_pairs


torch.set_printoptions(precision=4)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
RESULT_KEYS = ["auroc", "acc", "prec", "recall", "avg_prec", "tn", "fp", "fn", "tp"]


def str2bool(value):
    if isinstance(value, bool):
        return value
    return value.lower() in {"true", "1", "yes", "y"}


def set_seeds_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def log_builder(args):
    current_time = time.strftime("%Y-%m-%dT%H-%M", time.localtime())
    dataset_label = os.path.basename(os.path.normpath(args.dataset_name)) or "dataset"
    logd = os.path.join(f"exp_{args.task}", f"{args.name}_transformer_mask_{dataset_label}_{current_time}")
    os.makedirs(logd, exist_ok=True)
    logger = SummaryWriter(log_dir=logd)
    config = {key: value for key, value in vars(args).items() if not key.startswith("_")}
    config["path"] = logd
    with open(os.path.join(logd, "config.yaml"), "w") as file:
        file.write(yaml.dump(config))
    return logger, logd


def log_train(logger, loss, acc, num_batches):
    logger.add_scalar("loss/train", loss.item(), num_batches)
    logger.add_scalar("acc/train", acc.item(), num_batches)


def result_to_vector(result):
    return [result[key] for key in RESULT_KEYS]


def summarize_result_vectors(vectors, prefix):
    values = np.array(vectors, dtype=float)
    means = np.mean(values, axis=0)
    stds = np.sqrt(np.var(values, axis=0))
    summary = {}
    for idx, key in enumerate(RESULT_KEYS):
        summary[f"{prefix}_{key}_mean"] = means[idx]
        summary[f"{prefix}_{key}_std"] = stds[idx]
    return summary


def log_test_result(logger, args, result, runtime):
    logger.add_scalar(f"{args.task}_runtime/test", runtime)
    for key, value in result.items():
        logger.add_scalar(f"{args.task}_{key}/test", value)


def split_train_test_pairs(raw_dataset, graph_dataset, train_ratio=0.8, seed=0):
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    n = len(raw_dataset)
    if n != len(graph_dataset):
        raise ValueError(f"raw_dataset and graph_dataset must have the same length, got {n} and {len(graph_dataset)}")
    if n < 2:
        raise ValueError(f"Need at least 2 samples to split train/test, got {n}")
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_size = int(n * train_ratio)
    train_size = max(1, min(train_size, n - 1))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    raw_train = [raw_dataset[i] for i in train_indices]
    graph_train = [graph_dataset[i] for i in train_indices]
    raw_test = [raw_dataset[i] for i in test_indices]
    graph_test = [graph_dataset[i] for i in test_indices]
    return raw_train, graph_train, raw_test, graph_test


def split_dataset_object(dataset, valid_ratio=0.2, seed=0):
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError(f"valid_ratio must be in (0, 1), got {valid_ratio}")
    n = len(dataset)
    if n < 2:
        raise ValueError(f"Need at least 2 samples to split train/valid, got {n}")
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    valid_size = int(n * valid_ratio)
    valid_size = max(1, min(valid_size, n - 1))
    valid_indices = indices[:valid_size]
    train_indices = indices[valid_size:]

    def build_subset(selected_indices):
        subset = copy.copy(dataset)
        for attr_name, value in vars(dataset).items():
            if isinstance(value, list) and len(value) == n:
                setattr(subset, attr_name, [value[i] for i in selected_indices])
            elif isinstance(value, tuple) and len(value) == n:
                setattr(subset, attr_name, tuple(value[i] for i in selected_indices))
            elif isinstance(value, np.ndarray) and len(value) == n:
                setattr(subset, attr_name, value[selected_indices])
        return subset

    return build_subset(train_indices), build_subset(valid_indices)


def resolve_run_seeds(args):
    if args.seeds:
        seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
        if not seeds:
            raise ValueError("--seeds was provided but no valid integer seeds were found.")
        args.runs = len(seeds)
        return seeds
    return [args.seed + run for run in range(args.runs)]


def resolve_grf_root(dataset_name):
    candidates = [dataset_name, os.path.join("dataset", dataset_name)]
    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, "queries")) and os.path.isdir(os.path.join(candidate, "targets")):
            return candidate
    return None


def load_standard_datasets(args, seed):
    grf_root = resolve_grf_root(args.dataset_name)
    if grf_root is not None:
        perm_root = os.path.join(grf_root, "perm")
        label_roots = [grf_root]
        if os.path.isdir(perm_root):
            label_roots.append(perm_root)
        label_to_index = build_grf_label_to_index(label_roots)
        full_trainset = GRFTripletDataset(
            grf_root,
            label_to_index=label_to_index,
            cache=args.grf_cache,
            preload=args.grf_preload,
        )
        trainset, validset = split_dataset_object(full_trainset, valid_ratio=args.grf_valid_ratio, seed=seed)
        evalset = (
            GRFTripletDataset(
                perm_root,
                label_to_index=label_to_index,
                cache=args.grf_cache,
                preload=args.grf_preload,
            )
            if os.path.isdir(perm_root)
            else validset
        )
        return trainset, validset, evalset
    num_node_labels, (graphs_all, _, _), (raw_all, _, _) = load_tu_dataset(args.dataset_name)
    args.input_dim = 1 if num_node_labels == 0 else num_node_labels
    raw_train, graphs_train, raw_test, graphs_test = split_train_test_pairs(
        raw_all,
        graphs_all,
        train_ratio=args.train_ratio,
        seed=seed,
    )
    trainset = SCDataset(raw_train, graphs_train, args.dataset_name)
    validset = SCDataset(raw_test, graphs_test, args.dataset_name)
    return trainset, validset, validset


def get_candidate_graph_cache(args):
    if not args.candidate_graph_cache:
        return None
    cache = getattr(args, "_candidate_graph_cache", None)
    if cache is None:
        cache = {}
        setattr(args, "_candidate_graph_cache", cache)
    return cache


def candidate_training_signal(logits, args):
    if args.candidate_loss == "mse":
        return torch.sigmoid(logits)
    return logits


def candidate_objective(logits, labels, batch_size, args):
    labels = labels.float()
    if args.candidate_loss == "mse":
        return F.mse_loss(torch.sigmoid(logits), labels)
    if args.candidate_loss == "bce":
        return F.binary_cross_entropy_with_logits(logits, labels)
    if args.candidate_loss == "bce_pairwise":
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        pos_logits = logits[:batch_size]
        neg_logits = logits[batch_size:batch_size + pos_logits.size(0)]
        pairwise = F.softplus(neg_logits - pos_logits + args.candidate_pairwise_margin).mean()
        return bce + args.candidate_pairwise_weight * pairwise
    raise ValueError("candidate_loss must be one of: mse, bce, bce_pairwise.")


def train(model, optimizer, pred_optimizer, dataset, logger, epoch, args):
    model.train()
    total_loss = 0.0
    labels = torch.tensor(([1] * args.batch_size + [0] * args.batch_size), dtype=torch.long, device=device)
    runtime = 0.0
    criterion = CrossEntropyLoss()
    for i in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)
        pred_optimizer.zero_grad(set_to_none=True)
        _, g1, g2, g3 = dataset.get_batch(args.batch_size)
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
        loss = candidate_objective(logits, labels, args.batch_size, args)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.candidate_model.parameters(), 1.0)
        optimizer.step()
        raw_pred = candidate_training_signal(logits.detach(), args).view(-1)
        pred = model.predictor(raw_pred.unsqueeze(-1))
        pred_loss = criterion(pred, labels)
        pred_loss.backward()
        pred_optimizer.step()
        acc = torch.mean((pred.argmax(dim=-1) == labels).float())
        step = i + args.iterations * epoch
        log_train(logger, loss, acc, step)
        runtime += time.time() - start_time
        print(
            "Iteration: {:02d}. time: {:.4f}. loss: {:.4f}. Acc: {:.4f}.".format(
                step,
                time.time() - start_time,
                loss.item(),
                acc.item(),
            ),
            end="                    \r",
        )
        total_loss += loss.detach()
    return total_loss / args.iterations, runtime


def run(args):
    print(f"executing on {device}")
    results_list = []
    run_seeds = resolve_run_seeds(args)
    for run_idx, seed in enumerate(run_seeds):
        set_seeds_all(seed)
        logger, logd = log_builder(args)
        trainset, validset, evalset = load_standard_datasets(args, seed)
        model = NC(args).to(device)
        optimizer = torch.optim.Adam(model.candidate_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        pred_optimizer = torch.optim.Adam(model.predictor.parameters(), lr=args.lr)
        best_epoch = -1
        best_auroc = float("-inf")
        best_result = None
        best_state_dict = None
        bad_epoch = 0
        total_runtime = 0.0
        start_run = time.time()
        for epoch in range(args.epochs):
            loss, runtime = train(model, optimizer, pred_optimizer, trainset, logger, epoch, args)
            total_runtime += runtime
            logger.add_scalar("total_loss/train", loss.item(), epoch)
            logger.add_scalar("runtime/train", runtime, epoch)
            if (epoch + 1) % args.eval_steps == 0 and (epoch + 1) >= args.eval_interval:
                result, runtime = validation(model, validset, device, args)
                total_runtime += runtime
                if result["auroc"] > best_auroc:
                    best_auroc = result["auroc"]
                    best_result = result
                    best_epoch = epoch
                    bad_epoch = 0
                    best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    if args.save_model:
                        torch.save(model.state_dict(), os.path.join(logd, "best.pt"))
                else:
                    bad_epoch += 1
                for key, value in result.items():
                    logger.add_scalar(f"{args.task}_{key}/val", value, epoch)
                logger.add_scalar(f"{args.task}_best_auroc/val", best_auroc, epoch)
                logger.add_scalar(f"{args.task}_best_epoch/val", best_epoch, epoch)
                logger.add_scalar(f"{args.task}_runtime/val", runtime, epoch)
                print(
                    "\nEpoch: {:02d}. Loss: {:.4f}. AUROC: {:.4f}. Best AUROC: {:.4f}. ACC: {:.4f}. Best epoch: {:02d}".format(
                        epoch,
                        loss.item(),
                        result["auroc"],
                        best_auroc,
                        result["acc"],
                        best_epoch,
                    )
                )
            if bad_epoch >= args.patience:
                break
            if args.max_runtime_hours > 0 and total_runtime >= args.max_runtime_hours * 3600:
                break
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        result, runtime = test(model, evalset, device, args)
        log_test_result(logger, args, result, runtime)
        result_vector = result_to_vector(result)
        results_list.append(result_vector)
        with open(os.path.join(logd, "result"), "a") as file:
            if best_result is not None:
                for key, value in best_result.items():
                    file.write(f"best_{key} {value}\n")
            file.write(f"best_epoch {best_epoch}\n")
            for key, value in result.items():
                file.write(f"{key} {value}\n")
            file.write(f"runtime {runtime}\n")
        torch.save(result_vector, os.path.join(logd, "result.pt"))
        print(
            "\nRun: {:02d}. Runtime: {:.4f}. Best AUROC: {:.4f}. Test AUROC: {:.4f}. Test ACC: {:.4f}. Inference time: {:.4f}.".format(
                run_idx,
                time.time() - start_run,
                best_auroc,
                result["auroc"],
                result["acc"],
                runtime,
            )
        )
        logger.close()
    final_result = summarize_result_vectors(results_list, args.task)
    print(final_result)


def build_parser():
    parser = argparse.ArgumentParser(description="LPIMatch")
    parser.add_argument("--dataset_name", type=str, default="aids")
    parser.add_argument("--seed", type=int, default=4538)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--eval_batches", type=int, default=32)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--grf_valid_ratio", type=float, default=0.2)
    parser.add_argument("--grf_cache", type=str2bool, default=True)
    parser.add_argument("--grf_preload", type=str2bool, default=False)
    parser.add_argument("--save_model", type=str2bool, default=True)
    parser.add_argument("--max_runtime_hours", type=float, default=0.0)
    parser.add_argument("--task", type=str, default="main")
    parser.add_argument("--name", type=str, default="lpimatch")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--label_pi_mode", type=str, default="off", choices=["off"])
    parser.add_argument("--candidate_pi_mode", type=str, default="association", choices=["association"])
    parser.add_argument("--association_backend", type=str, default="transformer_mask", choices=["transformer_mask"])
    parser.add_argument("--association_heads", type=int, default=4)
    parser.add_argument("--association_pool", type=str, default="max", choices=["max", "mean", "sum", "max_mean"])
    parser.add_argument("--association_max_pairs", type=int, default=0)
    parser.add_argument("--association_max_edges", type=int, default=0)
    parser.add_argument("--candidate_maxloop", type=int, default=10)
    parser.add_argument("--candidate_hidden_dim", type=int, default=64)
    parser.add_argument("--candidate_num_layers", type=int, default=3)
    parser.add_argument("--candidate_dropout", type=float, default=0.0)
    parser.add_argument("--candidate_loss", type=str, default="mse", choices=["mse", "bce", "bce_pairwise"])
    parser.add_argument("--candidate_pairwise_weight", type=float, default=0.5)
    parser.add_argument("--candidate_pairwise_margin", type=float, default=0.2)
    parser.add_argument("--candidate_eval_batch_size", type=int, default=1)
    parser.add_argument("--candidate_graph_cache", type=str2bool, default=True)
    parser.add_argument("--candidate_graph_cache_max_entries", type=int, default=0)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
