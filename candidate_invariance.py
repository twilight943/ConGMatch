import hashlib
import importlib
from typing import Sequence

import networkx as nx
import torch
import torch.nn.functional as F
from torch.nn import Dropout, LayerNorm, Linear, Module, ReLU, Sequential
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import add_self_loops, remove_self_loops, scatter, softmax


class CandidateGraphData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "pair_index":
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == "pair_index":
            return 0
        return super().__inc__(key, value, *args, **kwargs)


ASSOCIATION_NODE_FEATURE_DIM = 6


def nx2list(graph):
    vid = list(graph.nodes)
    vlbl = [graph.nodes[node].get("label", None) for node in graph.nodes]
    edges = list(graph.edges)
    if not graph.is_directed():
        edges = edges + [(v, u) for u, v in edges]
    srcid = [u for u, v in edges]
    dstid = [v for u, v in edges]
    return [vid, vlbl, srcid, dstid]


def _load_simulation():
    return importlib.import_module("simulation")


def dualsim_candidate_sets(query: nx.Graph, data: nx.Graph, maxloop: int = 10, simulation=None):
    simulation = simulation if simulation is not None else _load_simulation()
    return simulation.dualsim_filter_qt(nx2list(query), nx2list(data), maxloop)


def candidate_sets_to_matrix(cands_set, num_query_nodes: int, num_data_nodes: int, device=None):
    matrix = torch.zeros((num_query_nodes, num_data_nodes), dtype=torch.bool, device=device)
    iterable = cands_set.items() if isinstance(cands_set, dict) else enumerate(cands_set)
    for q_idx, candidates in iterable:
        if q_idx >= num_query_nodes:
            break
        if isinstance(candidates, torch.Tensor):
            candidates = candidates.detach().cpu().view(-1).tolist()
        for g_idx in candidates:
            g_idx = int(g_idx)
            if 0 <= g_idx < num_data_nodes:
                matrix[q_idx, g_idx] = True
    return matrix


def pyg_data_to_nx(data: Data):
    graph = nx.Graph()
    x = getattr(data, "x", None)
    for node in range(int(data.num_nodes)):
        attrs = {}
        if x is None:
            attrs["label"] = 0
        else:
            node_x = x[node].detach().cpu()
            attrs["x"] = node_x.tolist()
            if node_x.numel() == 1:
                attrs["label"] = int(round(float(node_x.item())))
            else:
                attrs["label"] = int(torch.argmax(node_x).item())
        graph.add_node(node, **attrs)
    edge_index = getattr(data, "edge_index", None)
    if edge_index is not None:
        edges = edge_index.detach().cpu().t().tolist()
        graph.add_edges_from((int(u), int(v)) for u, v in edges if int(u) != int(v))
    return nx.convert_node_labels_to_integers(graph)


def dualsim_candidate_matrix(query: nx.Graph, data: nx.Graph, maxloop: int = 10, device=None, simulation=None):
    cands_set = dualsim_candidate_sets(query, data, maxloop=maxloop, simulation=simulation)
    return candidate_sets_to_matrix(
        cands_set,
        query.number_of_nodes(),
        data.number_of_nodes(),
        device=device,
    )


def _edge_set(graph: nx.Graph):
    graph = nx.convert_node_labels_to_integers(graph)
    return {tuple(sorted((int(u), int(v)))) for u, v in graph.edges()}


def _cap_candidate_matrix(candidate_matrix: torch.Tensor, max_pairs: int, query: nx.Graph = None, data: nx.Graph = None):
    if max_pairs is None or max_pairs <= 0:
        return candidate_matrix
    pair_count = int(candidate_matrix.sum().item())
    if pair_count <= max_pairs:
        return candidate_matrix
    capped = torch.zeros_like(candidate_matrix)
    active_rows = candidate_matrix.any(dim=1).nonzero(as_tuple=False).view(-1)
    if active_rows.numel() == 0:
        return capped
    per_row = max(1, (max_pairs + int(active_rows.numel()) - 1) // int(active_rows.numel()))
    kept = 0
    data_degrees = None
    col_counts = None
    if query is not None and data is not None:
        data_degrees = torch.tensor(
            [data.degree[int(g_idx)] for g_idx in range(candidate_matrix.size(1))],
            dtype=torch.float,
            device=candidate_matrix.device,
        )
        col_counts = candidate_matrix.float().sum(dim=0)
    for q_idx in active_rows.tolist():
        g_idx = candidate_matrix[q_idx].nonzero(as_tuple=False).view(-1)
        room = max_pairs - kept
        if room <= 0:
            break
        keep_count = min(per_row, room, int(g_idx.numel()))
        if data_degrees is not None and g_idx.numel() > keep_count:
            q_degree = float(query.degree[int(q_idx)])
            degree_gap = torch.relu(torch.tensor(q_degree, device=candidate_matrix.device) - data_degrees[g_idx])
            rarity = col_counts[g_idx] / max(candidate_matrix.size(0), 1)
            order = torch.argsort(degree_gap + 0.05 * rarity, stable=True)
            selected = g_idx[order[:keep_count]]
        else:
            selected = g_idx[:keep_count]
        capped[q_idx, selected] = True
        kept += int(selected.numel())
    return capped


def build_association_graph(
    query: nx.Graph,
    data: nx.Graph,
    candidate_matrix: torch.Tensor,
    max_pairs: int = 512,
    max_edges: int = 50000,
):
    query = nx.convert_node_labels_to_integers(query)
    data = nx.convert_node_labels_to_integers(data)
    candidate_matrix = _cap_candidate_matrix(candidate_matrix.detach().cpu(), max_pairs, query=query, data=data)
    device = candidate_matrix.device
    pair_index = candidate_matrix.nonzero(as_tuple=False)
    num_pairs = pair_index.size(0)
    if num_pairs == 0:
        return CandidateGraphData(
            x=torch.zeros((1, ASSOCIATION_NODE_FEATURE_DIM), dtype=torch.float, device=device),
            edge_index=torch.empty((2, 0), dtype=torch.long, device=device),
            pair_index=torch.full((1, 2), -1, dtype=torch.long, device=device),
        )
    query_edges = _edge_set(query)
    data_edges = _edge_set(data)
    qn = max(query.number_of_nodes(), 1)
    gn = max(data.number_of_nodes(), 1)
    q_deg = torch.tensor([query.degree[int(q)] for q, _ in pair_index.tolist()], dtype=torch.float, device=device)
    g_deg = torch.tensor([data.degree[int(g)] for _, g in pair_index.tolist()], dtype=torch.float, device=device)
    q_cand = candidate_matrix.float().sum(dim=1)[pair_index[:, 0]]
    g_cand = candidate_matrix.float().sum(dim=0)[pair_index[:, 1]]
    assoc_edges = []
    edge_limit = max_edges if max_edges and max_edges > 0 else None
    pairs = pair_index.tolist()
    for i, (q1, g1) in enumerate(pairs):
        for j in range(i + 1, num_pairs):
            q2, g2 = pairs[j]
            if q1 == q2 or g1 == g2:
                continue
            query_has_edge = tuple(sorted((int(q1), int(q2)))) in query_edges
            data_has_edge = tuple(sorted((int(g1), int(g2)))) in data_edges
            if query_has_edge and not data_has_edge:
                continue
            assoc_edges.append((i, j))
            assoc_edges.append((j, i))
            if edge_limit is not None and len(assoc_edges) >= edge_limit:
                break
        if edge_limit is not None and len(assoc_edges) >= edge_limit:
            break
    edge_index = (
        torch.tensor(assoc_edges, dtype=torch.long, device=device).t().contiguous()
        if assoc_edges
        else torch.empty((2, 0), dtype=torch.long, device=device)
    )
    x = torch.stack(
        [
            pair_index[:, 0].float() / qn,
            pair_index[:, 1].float() / gn,
            q_deg / qn,
            g_deg / gn,
            q_cand / gn,
            g_cand / qn,
        ],
        dim=-1,
    )
    return CandidateGraphData(x=x, edge_index=edge_index, pair_index=pair_index)


class SparseAssociationTransformerLayer(Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0, edge_chunk_size: int = 10000):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.edge_chunk_size = edge_chunk_size
        self.q_proj = Linear(hidden_dim, hidden_dim)
        self.k_proj = Linear(hidden_dim, hidden_dim)
        self.v_proj = Linear(hidden_dim, hidden_dim)
        self.out_proj = Linear(hidden_dim, hidden_dim)
        self.linear1 = Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = Linear(hidden_dim * 4, hidden_dim)
        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)
        self.dropout = Dropout(dropout)
        self.attn_dropout = Dropout(dropout)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def _attention_edges(self, edge_index: torch.Tensor, num_nodes: int):
        if edge_index.numel() > 0:
            edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        return edge_index

    def _edge_slices(self, edge_count: int):
        chunk_size = int(self.edge_chunk_size or 0)
        if chunk_size <= 0 or edge_count <= chunk_size:
            yield slice(None)
            return
        for start in range(0, edge_count, chunk_size):
            yield slice(start, min(start + chunk_size, edge_count))

    def _sparse_self_attention(self, x: torch.Tensor, edge_index: torch.Tensor):
        num_nodes, hidden_dim = x.size()
        edge_index = self._attention_edges(edge_index, num_nodes)
        query_index, key_index = edge_index
        edge_count = int(query_index.numel())
        q = self.q_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        score_chunks = []
        for edge_slice in self._edge_slices(edge_count):
            score_chunks.append((q[query_index[edge_slice]] * k[key_index[edge_slice]]).sum(dim=-1) * self.scale)
        scores = score_chunks[0] if len(score_chunks) == 1 else torch.cat(score_chunks, dim=0)
        alpha = softmax(scores, query_index, num_nodes=num_nodes)
        alpha = self.attn_dropout(alpha)
        out = x.new_zeros((num_nodes, self.num_heads, self.head_dim))
        for edge_slice in self._edge_slices(edge_count):
            messages = v[key_index[edge_slice]] * alpha[edge_slice].unsqueeze(-1)
            out = out + scatter(messages, query_index[edge_slice], dim=0, dim_size=num_nodes, reduce="sum")
        out = out.reshape(num_nodes, hidden_dim)
        return self.out_proj(out)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        attn_out = self._sparse_self_attention(x, edge_index)
        x = self.norm1(x + self.dropout1(attn_out))
        ff = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.norm2(x + self.dropout2(ff))


class AssociationGraphMatcher(Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.0,
        transformer_heads: int = 4,
        pool: str = "max",
    ):
        super().__init__()
        if pool not in {"max", "mean", "sum", "max_mean"}:
            raise ValueError("association_pool must be one of: max, mean, sum, max_mean.")
        self.pool = pool
        self.input = Linear(ASSOCIATION_NODE_FEATURE_DIM, hidden_dim)
        self.layers = torch.nn.ModuleList(
            [
                SparseAssociationTransformerLayer(
                    hidden_dim=hidden_dim,
                    num_heads=transformer_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.assignment_head = Linear(hidden_dim, 1)
        pooled_dim = hidden_dim * 2 if pool == "max_mean" else hidden_dim
        self.graph_head = Sequential(
            Linear(pooled_dim, hidden_dim),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, 1),
        )

    def forward(self, graph: Data):
        batch = getattr(graph, "batch", None)
        if batch is None:
            batch = graph.x.new_zeros(graph.x.size(0), dtype=torch.long)
        x = F.relu(self.input(graph.x))
        for layer in self.layers:
            x = layer(x, graph.edge_index)
        assignment_score = self.assignment_head(x).squeeze(-1)
        if self.pool == "mean":
            pooled = global_mean_pool(x, batch)
        elif self.pool == "sum":
            pooled = global_add_pool(x, batch)
        elif self.pool == "max_mean":
            pooled = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=-1)
        else:
            pooled = global_max_pool(x, batch)
        graph_score = self.graph_head(pooled).squeeze(-1)
        return assignment_score, graph_score


def build_association_from_dualsim(
    query: nx.Graph,
    data: nx.Graph,
    maxloop: int = 10,
    device=None,
    simulation=None,
    max_pairs: int = 512,
    max_edges: int = 50000,
):
    matrix = dualsim_candidate_matrix(query, data, maxloop=maxloop, device=None, simulation=simulation)
    return build_association_graph(query, data, matrix, max_pairs=max_pairs, max_edges=max_edges)


def _tensor_cache_token(tensor):
    if tensor is None:
        return None
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.blake2b(tensor.numpy().tobytes(), digest_size=16).hexdigest()
    return (tuple(tensor.shape), str(tensor.dtype), digest)


def _pyg_data_cache_token(data: Data):
    return (
        int(data.num_nodes),
        _tensor_cache_token(getattr(data, "x", None)),
        _tensor_cache_token(getattr(data, "edge_index", None)),
    )


def _candidate_graph_cache_key(query_data, data_data, maxloop, max_pairs, max_edges):
    return (
        int(maxloop),
        int(max_pairs or 0),
        int(max_edges or 0),
        _pyg_data_cache_token(query_data),
        _pyg_data_cache_token(data_data),
    )


def build_candidate_batch_from_pyg_pairs(
    queries: Sequence[Data],
    data_graphs: Sequence[Data],
    mode: str = "association",
    maxloop: int = 10,
    device=None,
    simulation=None,
    max_pairs: int = 512,
    max_edges: int = 50000,
    cache=None,
    cache_max_entries: int = 0,
):
    if mode != "association":
        raise ValueError("Only association mode is available in this transfer.")
    if len(queries) != len(data_graphs):
        raise ValueError(f"queries and data_graphs must be aligned, got {len(queries)} and {len(data_graphs)}")
    graphs = []
    for query_data, data_data in zip(queries, data_graphs):
        cache_key = None
        if cache is not None:
            cache_key = _candidate_graph_cache_key(query_data, data_data, maxloop, max_pairs, max_edges)
            cached_graph = cache.get(cache_key)
            if cached_graph is not None:
                graphs.append(cached_graph)
                continue
        query_graph = pyg_data_to_nx(query_data)
        data_graph = pyg_data_to_nx(data_data)
        graph = build_association_from_dualsim(
            query_graph,
            data_graph,
            maxloop=maxloop,
            device=None if cache is not None else device,
            simulation=simulation,
            max_pairs=max_pairs,
            max_edges=max_edges,
        )
        if cache is not None and (cache_max_entries is None or cache_max_entries <= 0 or len(cache) < cache_max_entries):
            cache[cache_key] = graph
        graphs.append(graph)
    return Batch.from_data_list(graphs).to(device)
