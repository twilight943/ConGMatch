import os
import random
import re

import networkx as nx
import numpy as np
import scipy.stats as scistats
import torch
from torch.utils.data import Dataset
from torch_geometric.datasets import PPI, TUDataset
from torch_geometric.utils import from_networkx, to_networkx


TU_DATASET_SPECS = {
    "enzymes": ("./dataset/ENZYMES", "ENZYMES", "tu"),
    "proteins": ("./dataset/PROTEINS", "PROTEINS", "tu"),
    "dd": ("./dataset/DD", "DD", "tu"),
    "cox2": ("./dataset/cox2", "COX2", "tu"),
    "reddit-binary": ("./dataset/REDDIT-BINARY", "REDDIT-BINARY", "tu"),
    "msrc_21": ("./dataset/MSRC_21", "MSRC_21", "tu"),
    "collab": ("./dataset/COLLAB", "COLLAB", "tu"),
    "dblp": ("./dataset/DBLP", "DBLP_v1", "tu"),
    "aids": ("./dataset/AIDS", "AIDS", "tu"),
    "ptc_fm": ("./dataset/PTC_FM", "PTC_FM", "tu"),
    "ptc_fr": ("./dataset/PTC_FR", "PTC_FR", "tu"),
    "ptc_mm": ("./dataset/PTC_MM", "PTC_MM", "tu"),
    "ptc_mr": ("./dataset/PTC_MR", "PTC_MR", "tu"),
    "mutag": ("./dataset/MUTAG", "MUTAG", "tu"),
    "firstmm_db": ("./dataset/FIRSTMM_DB", "FIRSTMM_DB", "tu"),
    "ppi": ("./dataset/PPI", None, "ppi"),
}


def normalize_dataset_name(dataset_name):
    norm_path = os.path.normpath(dataset_name)
    candidates = [norm_path, os.path.basename(norm_path)]
    candidates.extend(part for part in norm_path.split(os.sep) if part)
    for candidate in candidates:
        key = candidate.lower()
        if key in TU_DATASET_SPECS:
            return key
    return dataset_name.lower()


def _one_hot(index, size):
    vector = [0.0] * size
    vector[index] = 1.0
    return vector


def _read_grf_label_rows(path):
    rows = []
    with open(path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if not parts or parts[0] != "v":
                continue
            node = int(parts[1])
            label = int(parts[2]) if len(parts) > 2 else 0
            degree = int(parts[3]) if len(parts) > 3 else 0
            rows.append((node, label, degree))
    return rows


def build_grf_label_to_index(paths_or_roots):
    if isinstance(paths_or_roots, (str, os.PathLike)):
        paths_or_roots = [paths_or_roots]
    labels = set()
    for item in paths_or_roots:
        item = os.fspath(item)
        if os.path.isdir(item):
            search_dirs = [os.path.join(item, "queries"), os.path.join(item, "targets")]
            paths = []
            for search_dir in search_dirs:
                if os.path.isdir(search_dir):
                    paths.extend(
                        os.path.join(search_dir, filename)
                        for filename in os.listdir(search_dir)
                        if filename.endswith(".grf")
                    )
        else:
            paths = [item]
        for path in paths:
            for _, label, _ in _read_grf_label_rows(path):
                labels.add(label)
    if not labels:
        return {0: 0}
    return {label: idx for idx, label in enumerate(sorted(labels))}


def read_grf(path, label_to_index=None, num_node_features=None):
    graph = nx.Graph()
    if label_to_index is not None:
        num_node_features = num_node_features or len(label_to_index)
    with open(path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v":
                node = int(parts[1])
                label = int(parts[2]) if len(parts) > 2 else 0
                grf_degree = int(parts[3]) if len(parts) > 3 else 0
                if label_to_index is None:
                    feature = [float(label)]
                else:
                    if label not in label_to_index:
                        raise ValueError(f"GRF label {label} in {path} is missing from the provided label vocabulary.")
                    feature = _one_hot(label_to_index[label], num_node_features)
                graph.add_node(node, x=feature, node_label=label, grf_degree=grf_degree)
            elif parts[0] == "e":
                graph.add_edge(int(parts[1]), int(parts[2]))
    return nx.convert_node_labels_to_integers(graph)


def _graph_to_data(graph):
    data = from_networkx(graph)
    if data.x is None:
        data.x = torch.tensor(data.num_nodes * [1.0]).unsqueeze(-1)
    else:
        data.x = data.x.float()
        if data.x.dim() == 1:
            data.x = data.x.unsqueeze(-1)
    return data


class GRFTripletDataset(Dataset):
    query_pattern = re.compile(r"^query_(\d+)(?:_(\d+))?\.grf$")

    def __init__(self, root, label_to_index=None, cache=True, preload=False):
        self.root = root
        self.queries_dir = os.path.join(root, "queries")
        self.targets_dir = os.path.join(root, "targets")
        self.cache = cache
        self._triplet_cache = {}
        self._graph_cache = {}
        self.examples = self._index_examples()
        if not self.examples:
            raise ValueError(f"No valid .grf triplets found under {root}")
        if label_to_index is None:
            paths = sorted({path for example in self.examples for path in example})
            label_to_index = build_grf_label_to_index(paths)
        self.label_to_index = dict(label_to_index)
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
            query_id, neg_id = match.groups()
            path = os.path.join(self.queries_dir, filename)
            if neg_id is None:
                positives[query_id] = path
            else:
                negatives.setdefault(query_id, []).append(path)
        examples = []
        for query_id, pos_path in sorted(positives.items()):
            target_path = os.path.join(self.targets_dir, f"target_{query_id}.grf")
            if not os.path.exists(target_path):
                continue
            for neg_path in negatives.get(query_id, []):
                examples.append((target_path, pos_path, neg_path))
        return examples

    def _load_graph(self, path):
        if self.cache and path in self._graph_cache:
            return self._graph_cache[path]
        data = _graph_to_data(
            read_grf(
                path,
                label_to_index=self.label_to_index,
                num_node_features=self.num_node_features,
            )
        )
        if self.cache:
            self._graph_cache[path] = data
        return data

    def _load_triplet(self, example):
        if self.cache and example in self._triplet_cache:
            return self._triplet_cache[example]
        target_path, pos_path, neg_path = example
        triplet = (self._load_graph(target_path), self._load_graph(pos_path), self._load_graph(neg_path))
        if self.cache:
            self._triplet_cache[example] = triplet
        return triplet

    def preload(self):
        if not self.cache:
            self.cache = True
        for example in self.examples:
            self._load_triplet(example)

    def get_batch(self, batch_size):
        centers_pos = []
        g1s = []
        g2s = []
        g3s = []
        for _ in range(batch_size):
            g1, g2, g3 = self._load_triplet(random.choice(self.examples))
            centers_pos.append(0)
            g1s.append(g1)
            g2s.append(g2)
            g3s.append(g3)
        return centers_pos, g1s, g2s, g3s

    def iter_batches(self, batch_size):
        for start in range(0, len(self.examples), batch_size):
            centers_pos = []
            g1s = []
            g2s = []
            g3s = []
            for example in self.examples[start:start + batch_size]:
                g1, g2, g3 = self._load_triplet(example)
                centers_pos.append(0)
                g1s.append(g1)
                g2s.append(g2)
                g3s.append(g3)
            yield centers_pos, g1s, g2s, g3s


def load_tu_dataset(name, use_node_attr=False):
    name = normalize_dataset_name(name)
    if name not in TU_DATASET_SPECS:
        raise ValueError(f"Unrecognized dataset: {name}")
    root, dataset_id, dataset_kind = TU_DATASET_SPECS[name]
    if dataset_kind == "ppi":
        dataset = PPI(root=root)
    else:
        dataset = TUDataset(root=root, name=dataset_id, use_node_attr=use_node_attr)
    if dataset[0].x is None:
        num_node_labels = 0
    else:
        num_node_labels = dataset[0].x.size(-1)
    dataset = dataset.shuffle()
    train_len = int(len(dataset) * 0.8 * (1.0 - 0.2))
    val_len = int(len(dataset) * 0.8) - train_len
    graphs = []
    for data in dataset:
        if num_node_labels == 0:
            graph = to_networkx(data).to_undirected()
        else:
            graph = to_networkx(data, node_attrs=["x"]).to_undirected()
        graphs.append(graph)
    return (
        num_node_labels,
        (graphs[:train_len], graphs[train_len:train_len + val_len], graphs[train_len + val_len:]),
        (dataset[:train_len], dataset[train_len:train_len + val_len], dataset[train_len + val_len:]),
    )


def rooted_random_walk(graphs, sampling_prob, data_size=0, fix_id=None, task="data", name=None):
    while True:
        idx = fix_id
        if idx is None:
            idx = sampling_prob.rvs()
        elif task == "neg":
            tmp = idx
            while tmp == idx:
                tmp = sampling_prob.rvs()
            idx = tmp
        graph = graphs[idx].copy()
        if graph.number_of_nodes() < 2:
            continue
        start_node = random.choice(list(graph.nodes))
        num_nodes = graph.number_of_nodes()
        if task == "data":
            lower = min(29, max(1, int(num_nodes * 0.5)))
            upper = max(lower, num_nodes - 1)
            size = random.randint(lower, upper)
        else:
            size = random.randint(5, max(6, int(data_size * 0.6)))
        neigh = [start_node]
        frontier = list(set(graph.neighbors(start_node)) - set(neigh))
        visited = {start_node}
        while len(neigh) < size and frontier:
            new_node = random.choice(list(frontier))
            if new_node in neigh:
                continue
            neigh.append(new_node)
            visited.add(new_node)
            frontier += list(graph.neighbors(new_node))
            frontier = [node for node in frontier if node not in visited]
        if len(neigh) == size:
            return idx, start_node, graph.subgraph(neigh)


def discrete_sampling(graphs):
    probs = np.array([len(graph) for graph in graphs], dtype=float)
    probs /= np.sum(probs)
    return scistats.rv_discrete(values=(np.arange(len(graphs)), probs))


class SCDataset(Dataset):
    def __init__(self, dataset, graphs, name=None):
        self.name = name
        self.dataset = dataset
        self.graphs = graphs
        self.sampling_prob = discrete_sampling(self.graphs)

    def __getitem__(self, _):
        return None

    def _get_graph(self):
        while True:
            idx, _, target = rooted_random_walk(self.graphs, self.sampling_prob, name=self.name)
            if nx.is_connected(target) and len(target.edges) > 0 and len(target.nodes) > 5:
                return idx, nx.convert_node_labels_to_integers(target)

    def _get_triplet(self):
        idx, g = self._get_graph()
        num_nodes = g.number_of_nodes()
        _, center, pos_g = rooted_random_walk([g], self.sampling_prob, num_nodes, 0, task="pos")
        _, _, neg_g = rooted_random_walk(self.graphs, self.sampling_prob, num_nodes, idx, task="neg")
        return center, g, pos_g, neg_g

    def get_batch(self, batch_size):
        g1s = []
        g2s = []
        g3s = []
        centers_pos = []
        for _ in range(batch_size):
            center, g1, g2, g3 = self._get_triplet()
            centers_pos.append(center)
            data_items = [from_networkx(g1), from_networkx(g2), from_networkx(g3)]
            for data in data_items:
                if data.x is None:
                    data.x = torch.tensor(data.num_nodes * [1.0]).unsqueeze(-1)
                else:
                    data.x = data.x.float()
                    if data.x.dim() == 1:
                        data.x = data.x.unsqueeze(-1)
            g1s.append(data_items[0])
            g2s.append(data_items[1])
            g3s.append(data_items[2])
        return centers_pos, g1s, g2s, g3s
