#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Graph {
    int64_t num_nodes = 0;
    std::vector<std::vector<int64_t>> out;
};

struct Pair {
    int64_t q = -1;
    int64_t g = -1;
};

void validate_cpu_long_tensor(const torch::Tensor& tensor, const char* name) {
    if (!tensor.device().is_cpu()) {
        throw std::runtime_error(std::string(name) + " must be a CPU tensor.");
    }
    if (tensor.scalar_type() != torch::kInt64) {
        throw std::runtime_error(std::string(name) + " must have dtype torch.long.");
    }
}

void sort_unique(std::vector<int64_t>& values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

Graph build_graph(torch::Tensor labels, torch::Tensor edge_index) {
    validate_cpu_long_tensor(labels, "labels");
    validate_cpu_long_tensor(edge_index, "edge_index");
    if (labels.dim() != 1) {
        throw std::runtime_error("labels must be one-dimensional.");
    }
    if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
        throw std::runtime_error("edge_index must have shape [2, num_edges].");
    }

    edge_index = edge_index.contiguous();
    Graph graph;
    graph.num_nodes = labels.size(0);
    graph.out.resize(static_cast<size_t>(graph.num_nodes));

    const int64_t* edge_ptr = edge_index.data_ptr<int64_t>();
    const int64_t edge_count = edge_index.size(1);
    for (int64_t edge = 0; edge < edge_count; ++edge) {
        const int64_t src = edge_ptr[edge];
        const int64_t dst = edge_ptr[edge_count + edge];
        if (src < 0 || src >= graph.num_nodes || dst < 0 || dst >= graph.num_nodes) {
            continue;
        }
        graph.out[static_cast<size_t>(src)].push_back(dst);
    }
    for (auto& neighbors : graph.out) {
        sort_unique(neighbors);
    }
    return graph;
}

std::vector<Pair> pairs_from_tensor(torch::Tensor pair_index, const Graph& query, const Graph& data) {
    validate_cpu_long_tensor(pair_index, "pair_index");
    if (pair_index.dim() != 2 || pair_index.size(1) != 2) {
        throw std::runtime_error("pair_index must have shape [num_pairs, 2].");
    }

    pair_index = pair_index.contiguous();
    const int64_t pair_count = pair_index.size(0);
    const int64_t* pair_ptr = pair_index.data_ptr<int64_t>();
    std::vector<Pair> pairs;
    pairs.reserve(static_cast<size_t>(pair_count));
    for (int64_t pair_id = 0; pair_id < pair_count; ++pair_id) {
        const int64_t q = pair_ptr[2 * pair_id];
        const int64_t g = pair_ptr[2 * pair_id + 1];
        if (q < 0 || q >= query.num_nodes || g < 0 || g >= data.num_nodes) {
            throw std::runtime_error("pair_index contains an out-of-range node index.");
        }
        pairs.push_back({q, g});
    }
    return pairs;
}

int64_t pair_key(int64_t q, int64_t g, int64_t data_num_nodes) {
    return q * data_num_nodes + g;
}

torch::Tensor edge_tensor_from_vectors(
    const std::vector<int64_t>& src,
    const std::vector<int64_t>& dst) {
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto edge_index = torch::empty({2, static_cast<int64_t>(src.size())}, options);
    int64_t* out = edge_index.data_ptr<int64_t>();
    const int64_t edge_count = static_cast<int64_t>(src.size());
    for (int64_t edge_id = 0; edge_id < edge_count; ++edge_id) {
        out[edge_id] = src[static_cast<size_t>(edge_id)];
        out[edge_count + edge_id] = dst[static_cast<size_t>(edge_id)];
    }
    return edge_index;
}

bool append_directed_pair(
    std::vector<int64_t>& src,
    std::vector<int64_t>& dst,
    int64_t lhs,
    int64_t rhs,
    int64_t edge_limit) {
    if (lhs == rhs) {
        return edge_limit <= 0 || static_cast<int64_t>(src.size()) < edge_limit;
    }
    if (edge_limit > 0 && static_cast<int64_t>(src.size()) >= edge_limit) {
        return false;
    }
    src.push_back(lhs);
    dst.push_back(rhs);
    src.push_back(rhs);
    dst.push_back(lhs);
    return edge_limit <= 0 || static_cast<int64_t>(src.size()) < edge_limit;
}

bool has_directed_edge(const Graph& graph, int64_t src, int64_t dst) {
    const auto& neighbors = graph.out[static_cast<size_t>(src)];
    return std::binary_search(neighbors.begin(), neighbors.end(), dst);
}

torch::Tensor build_association_edge_index(
    const Graph& query,
    const Graph& data,
    const std::vector<Pair>& pairs,
    int64_t max_edges) {
    if (pairs.size() <= 1) {
        return torch::empty(
            {2, 0},
            torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    }

    std::vector<std::vector<int64_t>> pairs_by_query(
        static_cast<size_t>(query.num_nodes));
    std::unordered_map<int64_t, int64_t> pair_lookup;
    pair_lookup.reserve(pairs.size() * 2 + 1);
    for (int64_t pair_id = 0;
         pair_id < static_cast<int64_t>(pairs.size());
         ++pair_id) {
        const Pair& pair = pairs[static_cast<size_t>(pair_id)];
        pairs_by_query[static_cast<size_t>(pair.q)].push_back(pair_id);
        pair_lookup[pair_key(pair.q, pair.g, data.num_nodes)] = pair_id;
    }

    const int64_t edge_limit = max_edges > 0 ? max_edges : -1;
    const int64_t reserve_edges = edge_limit > 0
        ? edge_limit + 1
        : std::min<int64_t>(
              static_cast<int64_t>(pairs.size()) *
                  std::max<int64_t>(
                      static_cast<int64_t>(pairs.size()) - 1,
                      0),
              1 << 20);

    std::vector<int64_t> src;
    std::vector<int64_t> dst;
    src.reserve(static_cast<size_t>(reserve_edges));
    dst.reserve(static_cast<size_t>(reserve_edges));
    bool reached_limit = false;

    for (int64_t q1 = 0; q1 < query.num_nodes; ++q1) {
        const auto& left_pairs = pairs_by_query[static_cast<size_t>(q1)];
        if (left_pairs.empty()) {
            continue;
        }

        for (int64_t q2 = q1 + 1; q2 < query.num_nodes; ++q2) {
            const auto& right_pairs = pairs_by_query[static_cast<size_t>(q2)];
            if (right_pairs.empty()) {
                continue;
            }

            const bool query_q1_to_q2 = has_directed_edge(query, q1, q2);
            const bool query_q2_to_q1 = has_directed_edge(query, q2, q1);

            if (!query_q1_to_q2 && !query_q2_to_q1) {
                for (int64_t left_id : left_pairs) {
                    const int64_t g1 = pairs[static_cast<size_t>(left_id)].g;
                    for (int64_t right_id : right_pairs) {
                        const int64_t g2 = pairs[static_cast<size_t>(right_id)].g;
                        if (g1 == g2) {
                            continue;
                        }
                        if (!append_directed_pair(src, dst, left_id, right_id, edge_limit)) {
                            reached_limit = true;
                            break;
                        }
                    }
                    if (reached_limit) {
                        break;
                    }
                }
            } else if (query_q1_to_q2 && !query_q2_to_q1) {
                for (int64_t left_id : left_pairs) {
                    const int64_t g1 = pairs[static_cast<size_t>(left_id)].g;
                    for (int64_t g2 : data.out[static_cast<size_t>(g1)]) {
                        if (g1 == g2) {
                            continue;
                        }
                        const auto right_it =
                            pair_lookup.find(pair_key(q2, g2, data.num_nodes));
                        if (right_it == pair_lookup.end()) {
                            continue;
                        }
                        if (!append_directed_pair(src, dst, left_id, right_it->second, edge_limit)) {
                            reached_limit = true;
                            break;
                        }
                    }
                    if (reached_limit) {
                        break;
                    }
                }
            } else if (!query_q1_to_q2 && query_q2_to_q1) {
                for (int64_t right_id : right_pairs) {
                    const int64_t g2 = pairs[static_cast<size_t>(right_id)].g;
                    for (int64_t g1 : data.out[static_cast<size_t>(g2)]) {
                        if (g1 == g2) {
                            continue;
                        }
                        const auto left_it =
                            pair_lookup.find(pair_key(q1, g1, data.num_nodes));
                        if (left_it == pair_lookup.end()) {
                            continue;
                        }
                        if (!append_directed_pair(src, dst, left_it->second, right_id, edge_limit)) {
                            reached_limit = true;
                            break;
                        }
                    }
                    if (reached_limit) {
                        break;
                    }
                }
            } else {
                for (int64_t left_id : left_pairs) {
                    const int64_t g1 = pairs[static_cast<size_t>(left_id)].g;
                    for (int64_t g2 : data.out[static_cast<size_t>(g1)]) {
                        if (g1 == g2 || !has_directed_edge(data, g2, g1)) {
                            continue;
                        }
                        const auto right_it =
                            pair_lookup.find(pair_key(q2, g2, data.num_nodes));
                        if (right_it == pair_lookup.end()) {
                            continue;
                        }
                        if (!append_directed_pair(src, dst, left_id, right_it->second, edge_limit)) {
                            reached_limit = true;
                            break;
                        }
                    }
                    if (reached_limit) {
                        break;
                    }
                }
            }

            if (reached_limit) {
                break;
            }
        }
        if (reached_limit) {
            break;
        }
    }

    return edge_tensor_from_vectors(src, dst);
}

}

torch::Tensor build_association_edges(
    torch::Tensor query_labels,
    torch::Tensor query_edge_index,
    torch::Tensor data_labels,
    torch::Tensor data_edge_index,
    torch::Tensor pair_index,
    int64_t max_edges) {
    const Graph query = build_graph(query_labels, query_edge_index);
    const Graph data = build_graph(data_labels, data_edge_index);
    const std::vector<Pair> pairs = pairs_from_tensor(pair_index, query, data);
    return build_association_edge_index(query, data, pairs, max_edges);
}
