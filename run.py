import argparse
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from candidate_graph import build_candidate_batch
from dataset import load_datasets
from inference import evaluate
from model import CongMatch


RESULT_KEYS = (
    "auroc",
    "acc",
    "prec",
    "recall",
    "f1",
    "avg_prec",
    "tn",
    "fp",
    "fn",
    "tp",
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def candidate_cache(args):
    if not args.candidate_graph_cache:
        return None
    if not hasattr(args, "_candidate_graph_cache"):
        args._candidate_graph_cache = {}
    return args._candidate_graph_cache


def positive_mean(values, labels):
    positive = labels.view(-1) > 0.5
    if not positive.any():
        return values.new_zeros(())
    return values.view(-1)[positive].mean()


def train_epoch(model, optimizer, dataset, device, args, epoch):
    model.train()
    loss_sum = 0.0
    correct = 0
    example_count = 0
    runtime = 0.0
    for iteration in range(args.iterations):
        start = time.perf_counter()
        _, targets, positives, negatives = dataset.get_batch(args.batch_size)
        queries = positives + negatives
        paired_targets = targets + targets
        labels = torch.tensor(
            [1.0] * len(positives) + [0.0] * len(negatives), device=device
        )
        candidate_batch = build_candidate_batch(
            queries,
            paired_targets,
            device,
            maxloop=args.candidate_maxloop,
            max_pairs=args.association_max_pairs,
            max_edges=args.association_max_edges,
            cache=candidate_cache(args),
            cache_max_entries=args.candidate_graph_cache_max_entries,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model.candidate_logits(candidate_batch).view(-1)
        constraints = model.candidate_constraint_losses()
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss = loss + args.camh_lambda_cover * positive_mean(
            constraints["cover"], labels
        )
        loss = loss + args.camh_lambda_inj * positive_mean(
            constraints["inj"], labels
        )
        loss = loss + args.camh_lambda_edge * positive_mean(
            constraints["edge"], labels
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            predictions = (torch.sigmoid(logits) >= args.prediction_threshold).long()
            correct += int((predictions == labels.long()).sum().item())
            example_count += labels.numel()
        elapsed = time.perf_counter() - start
        runtime += elapsed
        loss_sum += float(loss.item())
        global_step = epoch * args.iterations + iteration
        print(
            f"epoch={epoch + 1} step={global_step + 1} "
            f"loss={loss.item():.6f} acc={correct / example_count:.4f} "
            f"time={elapsed:.4f}s",
            end="\r",
        )
    print()
    return loss_sum / args.iterations, correct / max(example_count, 1), runtime


def safe_name(value):
    parts = [part for part in Path(value).parts[-2:] if part not in {".", "..", "\\"}]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", "_".join(parts)).strip("_") or "dataset"


def make_output_directory(args, run_index, seed):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"congmatch_{safe_name(args.dataset_name)}_{timestamp}_run{run_index:02d}_seed{seed}"
    path = Path(args.output_dir) / name
    path.mkdir(parents=True, exist_ok=False)
    config = {key: value for key, value in vars(args).items() if not key.startswith("_")}
    (path / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def run_once(args, run_index, seed, device):
    set_seed(seed)
    if hasattr(args, "_candidate_graph_cache"):
        args._candidate_graph_cache.clear()
    output_directory = make_output_directory(args, run_index, seed)
    train_set, validation_set, test_set = load_datasets(args, seed)
    model = CongMatch(args).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_accuracy = float("-inf")
    best_epoch = -1
    best_state = None
    best_train_accuracy = 0.0
    stale_evaluations = 0
    total_runtime = 0.0
    for epoch in range(args.epochs):
        loss, train_accuracy, runtime = train_epoch(
            model, optimizer, train_set, device, args, epoch
        )
        total_runtime += runtime
        should_evaluate = (
            (epoch + 1) >= args.eval_interval
            and (epoch + 1) % args.eval_steps == 0
        )
        if should_evaluate:
            validation = evaluate(model, validation_set, device, args)
            total_runtime += validation["runtime"]
            print(
                f"epoch={epoch + 1} loss={loss:.6f} train_acc={train_accuracy:.4f} "
                f"val_acc={validation['acc']:.4f} val_auroc={validation['auroc']:.4f}"
            )
            if validation["acc"] > best_accuracy:
                best_accuracy = validation["acc"]
                best_epoch = epoch + 1
                best_train_accuracy = train_accuracy
                stale_evaluations = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                if args.save_model:
                    torch.save(best_state, output_directory / "best.pt")
            else:
                stale_evaluations += 1
        if args.patience > 0 and stale_evaluations >= args.patience:
            break
        if args.max_runtime_hours > 0 and total_runtime >= args.max_runtime_hours * 3600:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    validation = evaluate(model, validation_set, device, args)
    test = evaluate(model, test_set, device, args)
    result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "selected_train_acc": best_train_accuracy,
        "validation": validation,
        "test": test,
        "training_validation_runtime": total_runtime,
    }
    (output_directory / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"run={run_index:02d} best_epoch={best_epoch} "
        f"validation_acc={validation['acc']:.4f} test_acc={test['acc']:.4f} "
        f"test_auroc={test['auroc']:.4f} output={output_directory}"
    )
    return result


def run(args):
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    seeds = (
        [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
        if args.seeds
        else [args.seed + index for index in range(args.runs)]
    )
    results = [run_once(args, index, seed, device) for index, seed in enumerate(seeds)]
    test_vectors = np.array(
        [[result["test"][key] for key in RESULT_KEYS] for result in results], dtype=float
    )
    summary = {
        key: {
            "mean": float(test_vectors[:, index].mean()),
            "std": float(test_vectors[:, index].std()),
        }
        for index, key in enumerate(RESULT_KEYS)
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description="CongMatch")
    parser.add_argument("--dataset_name", default="cox2")
    parser.add_argument("--seed", type=int, default=4538)
    parser.add_argument("--seeds")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--device")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--eval_batches", type=int, default=32)
    parser.add_argument("--max_runtime_hours", type=float, default=0.0)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--grf_valid_ratio", type=float, default=0.2)
    parser.add_argument("--grf_cache", type=str2bool, default=True)
    parser.add_argument("--grf_preload", type=str2bool, default=False)
    parser.add_argument("--lr", type=float, default=0.00017)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--candidate_hidden_dim", type=int, default=32)
    parser.add_argument("--candidate_num_layers", type=int, default=2)
    parser.add_argument("--candidate_dropout", type=float, default=0.0)
    parser.add_argument("--association_heads", type=int, default=2)
    parser.add_argument("--candidate_maxloop", type=int, default=2)
    parser.add_argument("--association_max_pairs", type=int, default=0)
    parser.add_argument("--association_max_edges", type=int, default=0)
    parser.add_argument("--camh_gamma_c", type=float, default=0.984)
    parser.add_argument("--camh_lambda_cover", type=float, default=0.999)
    parser.add_argument("--camh_lambda_inj", type=float, default=0.975)
    parser.add_argument("--camh_lambda_edge", type=float, default=0.571)
    parser.add_argument("--candidate_eval_batch_size", type=int, default=0)
    parser.add_argument("--prediction_threshold", type=float, default=0.5)
    parser.add_argument("--candidate_graph_cache", type=str2bool, default=True)
    parser.add_argument("--candidate_graph_cache_max_entries", type=int, default=0)
    parser.add_argument("--output_dir", default="experiments")
    parser.add_argument("--save_model", type=str2bool, default=True)
    parser.add_argument(
        "--candidate_pi_mode", default="association_fused", choices=["association_fused"]
    )
    parser.add_argument(
        "--association_backend",
        default="transformer_dual_relation",
        choices=["transformer_dual_relation"],
    )
    parser.add_argument("--label_pi_mode", default="off", choices=["off"])
    parser.add_argument("--candidate_loss", default="bce", choices=["bce"])
    parser.add_argument("--camh_enable", type=str2bool, default=True)
    parser.add_argument("--camh_loss_type", default="camh_new", choices=["camh_new"])
    return parser


def validate_fixed_configuration(args):
    if not args.camh_enable:
        raise ValueError("This streamlined implementation only supports --camh_enable true")
    if args.batch_size <= 0 or args.iterations <= 0 or args.epochs <= 0:
        raise ValueError("batch_size, iterations and epochs must be positive")
    if args.candidate_eval_batch_size < 0:
        raise ValueError("candidate_eval_batch_size must be non-negative")


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    validate_fixed_configuration(arguments)
    run(arguments)
