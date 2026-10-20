import hashlib
import importlib
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.nn import Dropout, LayerNorm, Linear, Module, Parameter, ReLU, Sequential
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter, softmax, to_dense_batch


class CandidateGraphData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "pair_index":
            return 0
        if key in {"query_edge_index", "data_edge_index"}:
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == "pair_index":
            return 0
        if key in {"query_edge_index", "data_edge_index"}:
            return 0
        if key in {"query_edge_batch", "data_edge_batch"}:
            return 1
        return super().__inc__(key, value, *args, **kwargs)


_FUSED_MODULE = None


def _load_fused_module():
    global _FUSED_MODULE
    if _FUSED_MODULE is None:
        try:
            _FUSED_MODULE = importlib.import_module("_association_fused_cpp")
        except ImportError as error:
            raise ImportError(
                "Build the fused extension first with: python setup.py build_ext --inplace"
            ) from error
    return _FUSED_MODULE


def _labels(data):
    features = data.x.detach().cpu()
    if features.dim() == 1 or features.size(-1) == 1:
        return features.reshape(-1).round().long().contiguous()
    return features.argmax(dim=-1).long().contiguous()


def _edges(data):
    if data.edge_index is None:
        return torch.empty((2, 0), dtype=torch.long)
    return data.edge_index.detach().cpu().long().contiguous()


def _empty_graph(query, target):
    query_edges = _edges(query)
    target_edges = _edges(target)
    return CandidateGraphData(
        x=torch.zeros((1, 6), dtype=torch.float),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        pair_index=torch.full((1, 2), -1, dtype=torch.long),
        query_edge_index=query_edges,
        query_edge_batch=torch.zeros(query_edges.size(1), dtype=torch.long),
        data_edge_index=target_edges,
        data_edge_batch=torch.zeros(target_edges.size(1), dtype=torch.long),
        association_pair_count=torch.tensor([0.0]),
        association_edge_count=torch.tensor([0.0]),
        query_node_count=torch.tensor([max(int(query.num_nodes), 1)]),
        data_node_count=torch.tensor([max(int(target.num_nodes), 1)]),
        has_empty_candidate_set=torch.tensor([True]),
    )


def build_fused_association_graph(query, target, maxloop=2, max_pairs=0, max_edges=0):
    module = _load_fused_module()
    query_labels = _labels(query)
    target_labels = _labels(target)
    query_edges = _edges(query)
    target_edges = _edges(target)
    pair_index, has_empty = module.compute_dualsim_candidates(
        query_labels,
        query_edges,
        target_labels,
        target_edges,
        int(maxloop),
        int(max_pairs or 0),
    )
    if has_empty:
        return _empty_graph(query, target)
    association_edges = module.build_association_edges(
        query_labels,
        query_edges,
        target_labels,
        target_edges,
        pair_index,
        int(max_edges or 0),
    )
    query_count = max(int(query.num_nodes), 1)
    target_count = max(int(target.num_nodes), 1)
    pair_index = pair_index.long().contiguous()
    association_edges = association_edges.long().contiguous()
    query_degree = torch.bincount(query_edges[0], minlength=query_count).float()
    target_degree = torch.bincount(target_edges[0], minlength=target_count).float()
    query_candidates = torch.bincount(pair_index[:, 0], minlength=query_count).float()
    target_candidates = torch.bincount(pair_index[:, 1], minlength=target_count).float()
    query_ids = pair_index[:, 0]
    target_ids = pair_index[:, 1]
    features = torch.stack(
        [
            torch.zeros(pair_index.size(0)),
            torch.zeros(pair_index.size(0)),
            query_degree[query_ids] / query_count,
            target_degree[target_ids] / target_count,
            query_candidates[query_ids] / target_count,
            target_candidates[target_ids] / query_count,
        ],
        dim=-1,
    )
    return CandidateGraphData(
        x=features,
        edge_index=association_edges,
        pair_index=pair_index,
        query_edge_index=query_edges,
        query_edge_batch=torch.zeros(query_edges.size(1), dtype=torch.long),
        data_edge_index=target_edges,
        data_edge_batch=torch.zeros(target_edges.size(1), dtype=torch.long),
        association_pair_count=torch.tensor([float(pair_index.size(0))]),
        association_edge_count=torch.tensor([float(association_edges.size(1))]),
        query_node_count=torch.tensor([query_count]),
        data_node_count=torch.tensor([target_count]),
        has_empty_candidate_set=torch.tensor([False]),
    )


def _tensor_token(tensor):
    if tensor is None:
        return None
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.blake2b(tensor.numpy().tobytes(), digest_size=16).hexdigest()
    return tuple(tensor.shape), str(tensor.dtype), digest


def _cache_key(query, target, maxloop, max_pairs, max_edges):
    return (
        int(maxloop),
        int(max_pairs or 0),
        int(max_edges or 0),
        int(query.num_nodes),
        _tensor_token(query.x),
        _tensor_token(query.edge_index),
        int(target.num_nodes),
        _tensor_token(target.x),
        _tensor_token(target.edge_index),
    )


def build_candidate_batch(
    queries: Sequence[Data],
    targets: Sequence[Data],
    device,
    maxloop=2,
    max_pairs=0,
    max_edges=0,
    cache=None,
    cache_max_entries=0,
):
    if len(queries) != len(targets):
        raise ValueError("queries and targets must have equal lengths")
    graphs = []
    for query, target in zip(queries, targets):
        key = None
        if cache is not None:
            key = _cache_key(query, target, maxloop, max_pairs, max_edges)
            graph = cache.get(key)
            if graph is not None:
                graphs.append(graph)
                continue
        graph = build_fused_association_graph(query, target, maxloop, max_pairs, max_edges)
        if cache is not None:
            if cache_max_entries > 0 and len(cache) >= cache_max_entries:
                cache.pop(next(iter(cache)))
            cache[key] = graph
        graphs.append(graph)
    return Batch.from_data_list(graphs).to(device)


def _dense_local_positions(batch):
    counts = torch.bincount(batch)
    pointers = torch.cat([batch.new_zeros(1), counts.cumsum(dim=0)])
    return torch.arange(batch.numel(), device=batch.device) - pointers[batch]


def _dense_pair_adjacency(edge_index, batch, mask):
    graph_count, max_nodes = mask.size()
    adjacency = torch.zeros(
        (graph_count, max_nodes, max_nodes), dtype=torch.bool, device=mask.device
    )
    if edge_index is not None and edge_index.numel() > 0:
        edge_index = edge_index.to(mask.device)
        local_positions = _dense_local_positions(batch)
        source, target = edge_index
        graph_ids = batch[source]
        adjacency[graph_ids, local_positions[source], local_positions[target]] = True
    return adjacency


def _dense_graph_adjacency(edge_index, edge_batch, graph_count, node_count, device):
    adjacency = torch.zeros(
        (graph_count, node_count, node_count), dtype=torch.bool, device=device
    )
    if edge_index is None or edge_batch is None or edge_index.numel() == 0:
        return adjacency
    edge_index = edge_index.to(device)
    edge_batch = edge_batch.to(device)
    source, target = edge_index
    valid = (
        (edge_batch >= 0)
        & (edge_batch < graph_count)
        & (source >= 0)
        & (source < node_count)
        & (target >= 0)
        & (target < node_count)
    )
    if valid.any():
        adjacency[edge_batch[valid], source[valid], target[valid]] = True
    return adjacency


def _dual_relation_masks(graph, batch, mask):
    graph_count, max_nodes = mask.size()
    device = mask.device
    dense_pairs, _ = to_dense_batch(graph.pair_index.to(device), batch)
    valid = mask & (dense_pairs[..., 0] >= 0) & (dense_pairs[..., 1] >= 0)
    valid_pairs = valid.unsqueeze(1) & valid.unsqueeze(2)
    eye = torch.eye(max_nodes, dtype=torch.bool, device=device).unsqueeze(0)
    compatibility = _dense_pair_adjacency(graph.edge_index, batch, mask) & valid_pairs
    compatibility |= eye & valid_pairs
    conflict_type = torch.full(
        (graph_count, max_nodes, max_nodes), -1, dtype=torch.long, device=device
    )
    query_nodes = dense_pairs[..., 0].clamp_min(0)
    target_nodes = dense_pairs[..., 1].clamp_min(0)
    same_query = query_nodes.unsqueeze(1) == query_nodes.unsqueeze(2)
    same_target = target_nodes.unsqueeze(1) == target_nodes.unsqueeze(2)
    non_self = valid_pairs & ~eye
    conflict_type[non_self & same_query] = 0
    conflict_type[non_self & same_target] = 1
    if valid.any():
        query_count = int(query_nodes[valid].max().item()) + 1
        target_count = int(target_nodes[valid].max().item()) + 1
        query_adjacency = _dense_graph_adjacency(
            graph.query_edge_index,
            graph.query_edge_batch,
            graph_count,
            query_count,
            device,
        )
        target_adjacency = _dense_graph_adjacency(
            graph.data_edge_index,
            graph.data_edge_batch,
            graph_count,
            target_count,
            device,
        )
        graph_ids = torch.arange(graph_count, device=device).view(-1, 1, 1)
        query_has_edge = query_adjacency[
            graph_ids, query_nodes.unsqueeze(2), query_nodes.unsqueeze(1)
        ] & valid_pairs
        target_has_edge = target_adjacency[
            graph_ids, target_nodes.unsqueeze(2), target_nodes.unsqueeze(1)
        ] & valid_pairs
        violation = (
            non_self
            & ~same_query
            & ~same_target
            & query_has_edge
            & ~target_has_edge
        )
        conflict_type[violation] = 2
    return compatibility, conflict_type


class DenseMaskedRelationAttention(Module):
    def __init__(self, hidden_dim, head_count, dropout=0.0, relation_count=0):
        super().__init__()
        if hidden_dim % head_count:
            raise ValueError("candidate_hidden_dim must be divisible by association_heads")
        self.head_count = head_count
        self.head_dim = hidden_dim // head_count
        self.scale = self.head_dim ** -0.5
        self.query = Linear(hidden_dim, hidden_dim)
        self.key = Linear(hidden_dim, hidden_dim)
        self.value = Linear(hidden_dim, hidden_dim)
        self.output = Linear(hidden_dim, hidden_dim)
        self.dropout = Dropout(dropout)
        if relation_count:
            self.relation_bias = Parameter(torch.zeros(relation_count, head_count))
        else:
            self.register_parameter("relation_bias", None)

    def forward(self, x, allowed, relation_type=None):
        graph_count, max_nodes, hidden_dim = x.size()
        shape = (graph_count, max_nodes, self.head_count, self.head_dim)
        query = self.query(x).view(shape).transpose(1, 2)
        key = self.key(x).view(shape).transpose(1, 2)
        value = self.value(x).view(shape).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        allowed = allowed.to(device=x.device, dtype=torch.bool)
        if relation_type is not None:
            relation_type = relation_type.to(x.device)
            bias = self.relation_bias[relation_type.clamp_min(0)].permute(0, 3, 1, 2)
            scores = scores + bias.masked_fill((relation_type < 0).unsqueeze(1), 0.0)
        allowed_heads = allowed.unsqueeze(1)
        scores = scores.masked_fill(~allowed_heads, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1).masked_fill(~allowed_heads, 0.0)
        output = torch.matmul(self.dropout(weights), value)
        output = output.transpose(1, 2).reshape(graph_count, max_nodes, hidden_dim)
        return self.output(output)


class DualRelationTransformerLayer(Module):
    def __init__(self, hidden_dim, head_count, dropout):
        super().__init__()
        self.compatibility_attention = DenseMaskedRelationAttention(
            hidden_dim, head_count, dropout
        )
        self.conflict_attention = DenseMaskedRelationAttention(
            hidden_dim, head_count, dropout, relation_count=3
        )
        self.gate = Linear(hidden_dim, hidden_dim)
        self.linear1 = Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = Linear(hidden_dim * 4, hidden_dim)
        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)
        self.dropout = Dropout(dropout)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, x, compatibility, conflict_type):
        compatible = self.compatibility_attention(x, compatibility)
        conflict = self.conflict_attention(
            x, conflict_type >= 0, relation_type=conflict_type
        )
        relation = compatible - torch.sigmoid(self.gate(x)) * conflict
        x = self.norm1(x + self.dropout1(relation))
        feed_forward = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.norm2(x + self.dropout2(feed_forward))


class AssociationGraphMatcher(Module):
    def __init__(
        self,
        hidden_dim=32,
        layer_count=2,
        head_count=2,
        dropout=0.0,
        camh_gamma=0.984,
    ):
        super().__init__()
        if not 0.0 <= camh_gamma <= 1.0:
            raise ValueError("camh_gamma must be in [0, 1]")
        self.camh_gamma = camh_gamma
        self.input = Linear(6, hidden_dim)
        self.layers = torch.nn.ModuleList(
            [DualRelationTransformerLayer(hidden_dim, head_count, dropout) for _ in range(layer_count)]
        )
        self.assignment_head = Linear(hidden_dim, 1)
        self.graph_head = Sequential(
            Linear(hidden_dim + 3, hidden_dim),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, 1),
        )
        self.latest_constraint_losses = None

    @staticmethod
    def _num_graphs(graph, batch):
        return int(graph.num_graphs) if hasattr(graph, "num_graphs") else int(batch.max()) + 1

    @staticmethod
    def _counts(graph, name, batch, pairs, column, graph_count):
        value = getattr(graph, name, None)
        if value is not None and value.numel() >= graph_count:
            return value.to(batch.device).view(-1)[:graph_count].long().clamp_min(1)
        output = []
        for graph_id in range(graph_count):
            local = pairs[batch == graph_id]
            valid = (local[:, 0] >= 0) & (local[:, 1] >= 0)
            output.append(int(local[valid, column].max().item()) + 1 if valid.any() else 1)
        return torch.tensor(output, device=batch.device)

    @staticmethod
    def _local_edges(edge_index, edge_batch, graph_id, device):
        if edge_index is None:
            return None
        edge_index = edge_index.to(device)
        if edge_batch is None:
            return edge_index if graph_id == 0 else edge_index.new_empty((2, 0))
        return edge_index[:, edge_batch.to(device).view(-1) == graph_id]

    @staticmethod
    def _edge_loss(pairs, probabilities, query_edges, target_edges, query_count, target_count):
        if query_edges is None or query_edges.numel() == 0:
            return probabilities.new_zeros(())
        target_adjacency = torch.zeros(
            (target_count, target_count), dtype=torch.bool, device=probabilities.device
        )
        if target_edges is not None and target_edges.numel() > 0:
            source, target = target_edges.long()
            valid = (
                (source >= 0)
                & (source < target_count)
                & (target >= 0)
                & (target < target_count)
            )
            target_adjacency[source[valid], target[valid]] = True
        unique_query_edges = []
        seen = set()
        for source, target in query_edges.t().detach().cpu().tolist():
            source, target = int(source), int(target)
            key = (min(source, target), max(source, target))
            if source != target and 0 <= source < query_count and 0 <= target < query_count and key not in seen:
                seen.add(key)
                unique_query_edges.append((source, target))
        if not unique_query_edges:
            return probabilities.new_zeros(())
        loss = probabilities.new_zeros(())
        query_ids = pairs[:, 0]
        target_ids = pairs[:, 1]
        for query_source, query_target in unique_query_edges:
            source_mask = query_ids == query_source
            target_mask = query_ids == query_target
            if not source_mask.any() or not target_mask.any():
                continue
            mapped_source = target_ids[source_mask]
            mapped_target = target_ids[target_mask]
            incompatible = ~target_adjacency[
                mapped_source.unsqueeze(1), mapped_target.unsqueeze(0)
            ]
            weights = probabilities[source_mask].unsqueeze(1) * probabilities[target_mask].unsqueeze(0)
            loss = loss + weights[incompatible].sum()
        return loss / len(unique_query_edges)

    def _camh_readout(self, embeddings, assignment_scores, graph, batch):
        pairs = graph.pair_index.to(embeddings.device).long()
        graph_count = self._num_graphs(graph, batch)
        query_counts = self._counts(
            graph, "query_node_count", batch, pairs, 0, graph_count
        )
        target_counts = self._counts(
            graph, "data_node_count", batch, pairs, 1, graph_count
        )
        readouts = []
        cover_losses = []
        injectivity_losses = []
        edge_losses = []
        for graph_id in range(graph_count):
            local_mask = batch == graph_id
            local_embeddings = embeddings[local_mask]
            local_scores = assignment_scores[local_mask]
            local_pairs = pairs[local_mask]
            query_count = int(query_counts[graph_id].item())
            target_count = int(target_counts[graph_id].item())
            valid = (
                (local_pairs[:, 0] >= 0)
                & (local_pairs[:, 0] < query_count)
                & (local_pairs[:, 1] >= 0)
                & (local_pairs[:, 1] < target_count)
            )
            if valid.any():
                valid_embeddings = local_embeddings[valid]
                valid_scores = local_scores[valid]
                valid_pairs = local_pairs[valid]
                probabilities = softmax(
                    valid_scores, valid_pairs[:, 0], num_nodes=query_count
                )
                maximum = valid_embeddings.max(dim=0).values
                confidence = scatter(
                    probabilities,
                    valid_pairs[:, 0],
                    dim=0,
                    dim_size=query_count,
                    reduce="max",
                )
                cover = F.relu(self.camh_gamma - confidence).mean()
                target_mass = scatter(
                    probabilities,
                    valid_pairs[:, 1],
                    dim=0,
                    dim_size=target_count,
                    reduce="sum",
                )
                injectivity = F.relu(target_mass - 1.0).sum() / max(query_count - 1, 1)
                query_edges = self._local_edges(
                    graph.query_edge_index,
                    graph.query_edge_batch,
                    graph_id,
                    embeddings.device,
                )
                target_edges = self._local_edges(
                    graph.data_edge_index,
                    graph.data_edge_batch,
                    graph_id,
                    embeddings.device,
                )
                edge = self._edge_loss(
                    valid_pairs,
                    probabilities,
                    query_edges,
                    target_edges,
                    query_count,
                    target_count,
                )
            else:
                maximum = embeddings.new_zeros(embeddings.size(-1))
                cover = embeddings.new_tensor(self.camh_gamma)
                injectivity = embeddings.new_zeros(())
                edge = embeddings.new_zeros(())
            constraints = torch.stack([cover, injectivity, edge])
            readouts.append(torch.cat([maximum, constraints]))
            cover_losses.append(cover)
            injectivity_losses.append(injectivity)
            edge_losses.append(edge)
        losses = {
            "cover": torch.stack(cover_losses),
            "inj": torch.stack(injectivity_losses),
            "edge": torch.stack(edge_losses),
        }
        return torch.stack(readouts), losses

    def forward(self, graph):
        batch = graph.batch
        embeddings = F.relu(self.input(graph.x))
        dense, mask = to_dense_batch(embeddings, batch)
        compatibility, conflict_type = _dual_relation_masks(graph, batch, mask)
        for layer in self.layers:
            dense = layer(dense, compatibility, conflict_type)
        embeddings = dense[mask]
        assignment_scores = self.assignment_head(embeddings).squeeze(-1)
        readout, self.latest_constraint_losses = self._camh_readout(
            embeddings, assignment_scores, graph, batch
        )
        logits = self.graph_head(readout).squeeze(-1)
        return assignment_scores, logits
