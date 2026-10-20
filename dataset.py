import copy
import os
import random
import re
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.stats as scistats
import torch
from torch.utils.data import Dataset
from torch_geometric.datasets import PPI, TUDataset
from torch_geometric.utils import from_networkx, to_networkx


DATA_ROOT = Path(__file__).resolve().parent.parent / "dataset"

TU_DATASET_SPECS = {
    "enzymes": ("ENZYMES", "ENZYMES", "tu"),
    "proteins": ("PROTEINS", "PROTEINS", "tu"),
    "dd": ("DD", "DD", "tu"),
    "cox2": ("cox2", "COX2", "tu"),
    "reddit-binary": ("REDDIT-BINARY", "REDDIT-BINARY", "tu"),
    "msrc_21": ("MSRC_21", "MSRC_21", "tu"),
    "collab": ("COLLAB", "COLLAB", "tu"),
    "dblp": ("DBLP", "DBLP_v1", "tu"),
    "aids": ("AIDS", "AIDS", "tu"),
    "ptc_fm": ("PTC_FM", "PTC_FM", "tu"),
    "ptc_fr": ("PTC_FR", "PTC_FR", "tu"),
    "ptc_mm": ("PTC_MM", "PTC_MM", "tu"),
    "ptc_mr": ("PTC_MR", "PTC_MR", "tu"),
    "mutag": ("MUTAG", "MUTAG", "tu"),
    "firstmm_db": ("FIRSTMM_DB", "FIRSTMM_DB", "tu"),
    "ppi": ("PPI", None, "ppi"),
}


def normalize_dataset_name(dataset_name):
    normalized = os.path.normpath(dataset_name)
    for candidate in (normalized, os.path.basename(normalized)):
        key = candidate.lower()
        if key in TU_DATASET_SPECS:
            return key
    return dataset_name.lower()


def _one_hot(index, size):
    value = [0.0] * size
    value[index] = 1.0
    return value


def _read_grf_label_rows(path):
    rows = []
    with open(path, "r") as stream:
        for line in stream:
            parts = line.strip().split()
            if parts and parts[0] == "v":
                rows.append((int(parts[1]), int(parts[2]) if len(parts) > 2 else 0))
    return rows


def build_grf_label_to_index(roots):
    if isinstance(roots, (str, os.PathLike)):
        roots = [roots]
    labels = set()
    for root in roots:
        root = os.fspath(root)
        paths = []
        if os.path.isdir(root):
            for name in ("queries", "targets"):
                directory = os.path.join(root, name)
                if os.path.isdir(directory):
                    paths.extend(
                        os.path.join(directory, filename)
                        for filename in os.listdir(directory)
                        if filename.endswith(".grf")
                    )
        else:
            paths.append(root)
        for path in paths:
            labels.update(label for _, label in _read_grf_label_rows(path))
    if not labels:
        return {0: 0}
    return {label: index for index, label in enumerate(sorted(labels))}


def read_grf(path, label_to_index):
    graph = nx.Graph()
    feature_count = max(1, len(label_to_index))
    with open(path, "r") as stream:
        for line in stream:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v":
                node = int(parts[1])
                label = int(parts[2]) if len(parts) > 2 else 0
                if label not in label_to_index:
                    raise ValueError(f"GRF label {label} in {path} is not in the label vocabulary")
                graph.add_node(
                    node,
                    x=_one_hot(label_to_index[label], feature_count),
                    node_label=label,
                )
            elif parts[0] == "e":
                graph.add_edge(int(parts[1]), int(parts[2]))
    return nx.convert_node_labels_to_integers(graph)


def graph_to_data(graph):
    data = from_networkx(graph)
    if data.x is None:
        data.x = torch.ones((data.num_nodes, 1))
    else:
        data.x = data.x.float()
        if data.x.dim() == 1:
            data.x = data.x.unsqueeze(-1)
    return data


class GRFTripletDataset(Dataset):
    query_pattern = re.compile(r"^query_(\d+)(?:_(\d+))?\.grf$")

    def __init__(self, root, label_to_index=None, cache=True, preload=False):
        self.root = os.fspath(root)
        self.queries_dir = os.path.join(self.root, "queries")
        self.targets_dir = os.path.join(self.root, "targets")
        self.cache = cache
        self._graph_cache = {}
        self._triplet_cache = {}
        self.examples = self._index_examples()
        if not self.examples:
            raise ValueError(f"No valid GRF triplets found under {self.root}")
        self.label_to_index = dict(label_to_index or build_grf_label_to_index(self.root))
        self.num_node_features = max(1, len(self.label_to_index))
        if preload:
            self.preload()

    def __len__(self):
        return len(self.examples)

    def _index_examples(self):
        if not os.path.isdir(self.queries_dir):
            raise FileNotFoundError(f"Missing queries directory: {self.queries_dir}")
        if not os.path.isdir(self.targets_dir):
            raise FileNotFoundError(f"Missing targets directory: {self.targets_dir}")
        positives = {}
        negatives = {}
        for filename in sorted(os.listdir(self.queries_dir)):
            match = self.query_pattern.match(filename)
            if match is None:
                continue
            query_id, negative_id = match.groups()
            path = os.path.join(self.queries_dir, filename)
            if negative_id is None:
                positives[query_id] = path
            else:
                negatives.setdefault(query_id, []).append((int(negative_id), path))
        examples = []
        for query_id, positive_path in sorted(positives.items()):
            target_path = os.path.join(self.targets_dir, f"target_{query_id}.grf")
            if os.path.exists(target_path):
                for _, negative_path in sorted(negatives.get(query_id, [])):
                    examples.append((target_path, positive_path, negative_path))
        return examples

    def subset(self, indices):
        subset = copy.copy(self)
        subset.examples = [self.examples[index] for index in indices]
        return subset

    def _load_graph(self, path):
        if self.cache and path in self._graph_cache:
            return self._graph_cache[path]
        data = graph_to_data(read_grf(path, self.label_to_index))
        if self.cache:
            self._graph_cache[path] = data
        return data

    def _load_triplet(self, example):
        if self.cache and example in self._triplet_cache:
            return self._triplet_cache[example]
        triplet = tuple(self._load_graph(path) for path in example)
        if self.cache:
            self._triplet_cache[example] = triplet
        return triplet

    def preload(self):
        if not self.cache:
            self.cache = True
        for example in self.examples:
            self._load_triplet(example)

    def get_batch(self, batch_size):
        triplets = [self._load_triplet(random.choice(self.examples)) for _ in range(batch_size)]
        return [0] * batch_size, *map(list, zip(*triplets))

    def iter_batches(self, batch_size):
        for start in range(0, len(self.examples), batch_size):
            triplets = [self._load_triplet(item) for item in self.examples[start:start + batch_size]]
            yield [0] * len(triplets), *map(list, zip(*triplets))


def load_tu_dataset(name):
    name = normalize_dataset_name(name)
    if name not in TU_DATASET_SPECS:
        raise ValueError(f"Unknown TU dataset: {name}")
    root_name, dataset_id, dataset_kind = TU_DATASET_SPECS[name]
    root = DATA_ROOT / root_name
    dataset = PPI(root=str(root)) if dataset_kind == "ppi" else TUDataset(root=str(root), name=dataset_id)
    num_node_labels = 0 if dataset[0].x is None else dataset[0].x.size(-1)
    dataset = dataset.shuffle()
    graphs = []
    for data in dataset:
        if num_node_labels == 0:
            graphs.append(to_networkx(data).to_undirected())
        else:
            graphs.append(to_networkx(data, node_attrs=["x"]).to_undirected())
    train_length = int(len(dataset) * 0.8 * 0.8)
    validation_length = int(len(dataset) * 0.8) - train_length
    slices = (
        slice(0, train_length),
        slice(train_length, train_length + validation_length),
        slice(train_length + validation_length, None),
    )
    return tuple(graphs[item] for item in slices)


def rooted_random_walk(graphs, sampling_probability, data_size=0, fixed_id=None, task="data", name=None):
    while True:
        index = fixed_id
        if index is None:
            index = sampling_probability.rvs()
        elif task == "neg":
            alternative = index
            while alternative == index:
                alternative = sampling_probability.rvs()
            index = alternative
        graph = graphs[index].copy()
        start_node = random.choice(list(graph.nodes))
        node_count = graph.number_of_nodes()
        if task == "data":
            if name in {"firstmm_db", "collab", "dd", "ppi"} and node_count > 100:
                node_count = 101
            size = random.randint(min(29, int(node_count * 0.5)), node_count - 1)
        else:
            size = random.randint(5, max(6, int(data_size * 0.6)))
        nodes = [start_node]
        visited = {start_node}
        frontier = list(set(graph.neighbors(start_node)) - visited)
        while len(nodes) < size and frontier:
            node = random.choice(frontier)
            nodes.append(node)
            visited.add(node)
            frontier.extend(graph.neighbors(node))
            frontier = [item for item in frontier if item not in visited]
        if len(nodes) == size:
            return index, start_node, graph.subgraph(nodes)


def discrete_sampling(graphs):
    probabilities = np.array([len(graph) for graph in graphs], dtype=float)
    probabilities /= probabilities.sum()
    return scistats.rv_discrete(values=(np.arange(len(graphs)), probabilities))


class SCDataset(Dataset):
    def __init__(self, graphs, name=None):
        self.name = name
        self.graphs = graphs
        self.sampling_probability = discrete_sampling(graphs)

    def __len__(self):
        return len(self.graphs)

    def _get_graph(self):
        while True:
            index, _, target = rooted_random_walk(self.graphs, self.sampling_probability, name=self.name)
            if nx.is_connected(target) and target.number_of_edges() > 0 and target.number_of_nodes() > 5:
                return index, nx.convert_node_labels_to_integers(target)

    def _get_triplet(self):
        index, target = self._get_graph()
        node_count = target.number_of_nodes()
        _, center, positive = rooted_random_walk([target], self.sampling_probability, node_count, 0, "pos")
        _, _, negative = rooted_random_walk(self.graphs, self.sampling_probability, node_count, index, "neg")
        return center, target, positive, negative

    def get_batch(self, batch_size):
        centers = []
        targets = []
        positives = []
        negatives = []
        for _ in range(batch_size):
            center, target, positive, negative = self._get_triplet()
            centers.append(center)
            converted = [graph_to_data(graph) for graph in (target, positive, negative)]
            targets.append(converted[0])
            positives.append(converted[1])
            negatives.append(converted[2])
        return centers, targets, positives, negatives


class SynchronizedLabelPermutationEvalDataset:
    def __init__(self, dataset, num_labels, seed=0):
        self.dataset = dataset
        self.num_labels = int(num_labels)
        self.seed = seed
        self.label_to_index = getattr(dataset, "label_to_index", None)
        self.index_to_label = None
        if self.label_to_index is not None:
            self.index_to_label = {index: label for label, index in self.label_to_index.items()}

    def __len__(self):
        return len(self.dataset)

    def _permutation(self, sample_index):
        values = list(range(self.num_labels))
        if self.num_labels <= 1:
            return torch.tensor(values)
        if self.num_labels == 2:
            return torch.tensor([1, 0])
        rng = random.Random((self.seed + 1) * 1000003 + sample_index)
        while True:
            rng.shuffle(values)
            if all(source != target for source, target in enumerate(values)):
                return torch.tensor(values)

    def _permute_graph(self, graph, permutation):
        graph = graph.clone()
        if self.num_labels <= 1 or graph.x is None:
            return graph
        permutation = permutation.to(graph.x.device)
        label_ids = graph.x.argmax(dim=-1)
        features = torch.zeros_like(graph.x)
        features[torch.arange(graph.x.size(0), device=graph.x.device), permutation[label_ids]] = 1.0
        graph.x = features
        if self.index_to_label is not None and hasattr(graph, "node_label"):
            raw = graph.node_label.detach().cpu().view(-1).tolist()
            remapped = [
                self.index_to_label[int(permutation[self.label_to_index[int(label)]])]
                for label in raw
            ]
            graph.node_label = torch.tensor(remapped, dtype=graph.node_label.dtype)
        return graph

    def iter_batches(self, batch_size):
        sample_index = 0
        for centers, targets, positives, negatives in self.dataset.iter_batches(batch_size):
            output = [[], [], []]
            for offset, triplet in enumerate(zip(targets, positives, negatives)):
                permutation = self._permutation(sample_index + offset)
                for position, graph in enumerate(triplet):
                    output[position].append(self._permute_graph(graph, permutation))
            yield centers, output[0], output[1], output[2]
            sample_index += len(targets)


def split_tu_data(graphs, ratio, seed):
    if not 0.0 < ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    indices = list(range(len(graphs)))
    random.Random(seed).shuffle(indices)
    split = max(1, min(int(len(indices) * ratio), len(indices) - 1))
    train_indices, test_indices = indices[:split], indices[split:]
    return [graphs[index] for index in train_indices], [graphs[index] for index in test_indices]


def split_grf_data(dataset, ratio, seed):
    if not 0.0 < ratio < 1.0:
        raise ValueError("grf_valid_ratio must be in (0, 1)")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    validation_size = max(1, min(int(len(indices) * ratio), len(indices) - 1))
    return dataset.subset(indices[validation_size:]), dataset.subset(indices[:validation_size])


def resolve_grf_root(dataset_name):
    for candidate in (Path(dataset_name), DATA_ROOT / dataset_name):
        if (candidate / "queries").is_dir() and (candidate / "targets").is_dir():
            return candidate
    return None


def load_datasets(args, seed):
    grf_root = resolve_grf_root(args.dataset_name)
    explicit_path = os.path.isabs(args.dataset_name) or any(mark in args.dataset_name for mark in ("/", "\\"))
    if explicit_path and grf_root is None:
        raise FileNotFoundError(f"Not a GRF dataset root: {args.dataset_name}")
    if grf_root is not None:
        vocabulary = build_grf_label_to_index(grf_root)
        full = GRFTripletDataset(grf_root, vocabulary, args.grf_cache, args.grf_preload)
        train, validation = split_grf_data(full, args.grf_valid_ratio, seed)
        test = SynchronizedLabelPermutationEvalDataset(validation, full.num_node_features, seed)
        return train, validation, test
    graph_splits = load_tu_dataset(args.dataset_name)
    graphs_train, graphs_test = split_tu_data(list(graph_splits[0]), args.train_ratio, seed)
    return (
        SCDataset(graphs_train, args.dataset_name),
        SCDataset(graphs_test, args.dataset_name),
        SCDataset(graphs_test, args.dataset_name),
    )
