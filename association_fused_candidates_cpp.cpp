#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace py = pybind11;

namespace {

using Bitset = std::vector<uint64_t>;

struct Graph {
    int64_t num_nodes = 0;
    std::vector<int64_t> labels;
    std::vector<std::vector<int64_t>> out;
    std::vector<std::vector<int64_t>> in;
    std::vector<int64_t> degree;
};

struct Pair {
    int64_t q = -1;
    int64_t g = -1;
};

int word_count(int64_t num_bits) {
    return static_cast<int>((num_bits + 63) / 64);
}

void set_bit(Bitset& bits, int64_t idx) {
    bits[static_cast<size_t>(idx / 64)] |= (uint64_t{1} << (idx % 64));
}

bool clear_bit(Bitset& bits, int64_t idx) {
    uint64_t& word = bits[static_cast<size_t>(idx / 64)];
    const uint64_t mask = uint64_t{1} << (idx % 64);
    if ((word & mask) == 0) {
        return false;
    }
    word &= ~mask;
    return true;
}

bool intersects(const Bitset& lhs, const Bitset& rhs) {
    const size_t count = std::min(lhs.size(), rhs.size());
    for (size_t i = 0; i < count; ++i) {
        if ((lhs[i] & rhs[i]) != 0) {
            return true;
        }
    }
    return false;
}

int trailing_zeroes(uint64_t value) {
#if defined(_MSC_VER)
    unsigned long idx = 0;
    _BitScanForward64(&idx, value);
    return static_cast<int>(idx);
#else
    return __builtin_ctzll(value);
#endif
}

void sort_unique(std::vector<int64_t>& values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

void validate_cpu_long_tensor(const torch::Tensor& tensor, const char* name) {
    if (!tensor.device().is_cpu()) {
        throw std::runtime_error(std::string(name) + " must be a CPU tensor.");
    }
    if (tensor.scalar_type() != torch::kInt64) {
        throw std::runtime_error(std::string(name) + " must have dtype torch.long.");
    }
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

    labels = labels.contiguous();
    edge_index = edge_index.contiguous();

    Graph graph;
    graph.num_nodes = labels.size(0);
    graph.labels.resize(static_cast<size_t>(graph.num_nodes));
    graph.out.resize(static_cast<size_t>(graph.num_nodes));
    graph.in.resize(static_cast<size_t>(graph.num_nodes));
    graph.degree.resize(static_cast<size_t>(graph.num_nodes), 0);

    const int64_t* label_ptr = labels.data_ptr<int64_t>();
    for (int64_t node = 0; node < graph.num_nodes; ++node) {
        graph.labels[static_cast<size_t>(node)] = label_ptr[node];
    }





    const int64_t* edge_ptr = edge_index.data_ptr<int64_t>();
    const int64_t edge_count = edge_index.size(1);
    for (int64_t edge = 0; edge < edge_count; ++edge) {
        const int64_t src = edge_ptr[edge];
        const int64_t dst = edge_ptr[edge_count + edge];
        if (src < 0 || src >= graph.num_nodes || dst < 0 || dst >= graph.num_nodes) {
            continue;
        }
        graph.out[static_cast<size_t>(src)].push_back(dst);
        graph.in[static_cast<size_t>(dst)].push_back(src);
    }



    for (int64_t node = 0; node < graph.num_nodes; ++node) {
        auto& out_neighbors = graph.out[static_cast<size_t>(node)];
        auto& in_neighbors = graph.in[static_cast<size_t>(node)];
        sort_unique(out_neighbors);
        sort_unique(in_neighbors);
        graph.degree[static_cast<size_t>(node)] =
            static_cast<int64_t>(out_neighbors.size());
    }
    return graph;
}

std::vector<Bitset> adjacency_bitsets(const std::vector<std::vector<int64_t>>& adjacency, int64_t num_nodes) {
    const int words = word_count(num_nodes);
    std::vector<Bitset> bits(adjacency.size(), Bitset(static_cast<size_t>(words), 0));
    for (size_t node = 0; node < adjacency.size(); ++node) {
        for (int64_t neighbor : adjacency[node]) {
            set_bit(bits[node], neighbor);
        }
    }
    return bits;
}

bool candidate_supported_outgoing(
    int64_t query_idx,
    int64_t data_idx,
    const Graph& query,
    const std::vector<Bitset>& data_out_bits,
    const std::vector<Bitset>& sim) {
    for (int64_t query_next : query.out[static_cast<size_t>(query_idx)]) {
        if (!intersects(
                data_out_bits[static_cast<size_t>(data_idx)],
                sim[static_cast<size_t>(query_next)])) {
            return false;
        }
    }
    return true;
}

bool candidate_supported_incoming(
    int64_t query_idx,
    int64_t data_idx,
    const Graph& query,
    const std::vector<Bitset>& data_in_bits,
    const std::vector<Bitset>& sim) {
    for (int64_t query_prev : query.in[static_cast<size_t>(query_idx)]) {
        if (!intersects(
                data_in_bits[static_cast<size_t>(data_idx)],
                sim[static_cast<size_t>(query_prev)])) {
            return false;
        }
    }
    return true;
}

bool dualsim(
    const Graph& query,
    const Graph& data,
    int64_t maxloop,
    std::vector<Bitset>& sim,
    std::vector<int64_t>& sim_count) {
    if (query.num_nodes == 0 || data.num_nodes == 0) {
        return false;
    }

    std::unordered_map<int64_t, std::vector<int64_t>> data_by_label;
    data_by_label.reserve(static_cast<size_t>(data.num_nodes) * 2 + 1);
    for (int64_t data_idx = 0; data_idx < data.num_nodes; ++data_idx) {
        data_by_label[data.labels[static_cast<size_t>(data_idx)]].push_back(data_idx);
    }

    const int words = word_count(data.num_nodes);
    sim.assign(
        static_cast<size_t>(query.num_nodes),
        Bitset(static_cast<size_t>(words), 0));
    sim_count.assign(static_cast<size_t>(query.num_nodes), 0);


    for (int64_t query_idx = 0; query_idx < query.num_nodes; ++query_idx) {
        const auto bucket_it =
            data_by_label.find(query.labels[static_cast<size_t>(query_idx)]);
        if (bucket_it == data_by_label.end()) {
            return false;
        }
        for (int64_t data_idx : bucket_it->second) {
            set_bit(sim[static_cast<size_t>(query_idx)], data_idx);
            ++sim_count[static_cast<size_t>(query_idx)];
        }
        if (sim_count[static_cast<size_t>(query_idx)] == 0) {
            return false;
        }
    }

    const auto data_out_bits = adjacency_bitsets(data.out, data.num_nodes);
    const auto data_in_bits = adjacency_bitsets(data.in, data.num_nodes);





    int64_t nowloop = 0;
    bool changed = false;
    do {
        changed = false;
        if (maxloop > 0 && nowloop == maxloop) {
            break;
        }
        ++nowloop;


        for (int64_t query_idx = 0; query_idx < query.num_nodes; ++query_idx) {
            Bitset& query_sim = sim[static_cast<size_t>(query_idx)];
            const Bitset query_sim_copy = query_sim;

            for (int word_idx = 0; word_idx < words; ++word_idx) {
                uint64_t word = query_sim_copy[static_cast<size_t>(word_idx)];
                while (word != 0) {
                    const int bit = trailing_zeroes(word);
                    const int64_t data_idx =
                        static_cast<int64_t>(word_idx) * 64 + bit;
                    word &= word - 1;
                    if (data_idx >= data.num_nodes) {
                        continue;
                    }

                    if (!candidate_supported_outgoing(
                            query_idx,
                            data_idx,
                            query,
                            data_out_bits,
                            sim)) {
                        if (clear_bit(query_sim, data_idx)) {
                            --sim_count[static_cast<size_t>(query_idx)];
                            changed = true;
                            if (sim_count[static_cast<size_t>(query_idx)] == 0) {
                                return false;
                            }
                        }
                    }
                }
            }
        }


        for (int64_t query_idx = 0; query_idx < query.num_nodes; ++query_idx) {
            Bitset& query_sim = sim[static_cast<size_t>(query_idx)];
            const Bitset query_sim_copy = query_sim;

            for (int word_idx = 0; word_idx < words; ++word_idx) {
                uint64_t word = query_sim_copy[static_cast<size_t>(word_idx)];
                while (word != 0) {
                    const int bit = trailing_zeroes(word);
                    const int64_t data_idx =
                        static_cast<int64_t>(word_idx) * 64 + bit;
                    word &= word - 1;
                    if (data_idx >= data.num_nodes) {
                        continue;
                    }

                    if (!candidate_supported_incoming(
                            query_idx,
                            data_idx,
                            query,
                            data_in_bits,
                            sim)) {
                        if (clear_bit(query_sim, data_idx)) {
                            --sim_count[static_cast<size_t>(query_idx)];
                            changed = true;
                            if (sim_count[static_cast<size_t>(query_idx)] == 0) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
    } while (changed);

    return true;
}

std::vector<int64_t> candidates_for_row(const Bitset& bits, int64_t num_data_nodes) {
    std::vector<int64_t> candidates;
    for (int word_idx = 0; word_idx < static_cast<int>(bits.size()); ++word_idx) {
        uint64_t word = bits[static_cast<size_t>(word_idx)];
        while (word != 0) {
            const int bit = trailing_zeroes(word);
            const int64_t data_idx = static_cast<int64_t>(word_idx) * 64 + bit;
            word &= word - 1;
            if (data_idx < num_data_nodes) {
                candidates.push_back(data_idx);
            }
        }
    }
    return candidates;
}

std::vector<Pair> collect_pairs(
    const Graph& query,
    const Graph& data,
    const std::vector<Bitset>& sim,
    const std::vector<int64_t>& sim_count,
    int64_t max_pairs) {
    int64_t pair_count = 0;
    std::vector<int64_t> active_rows;
    active_rows.reserve(static_cast<size_t>(query.num_nodes));
    for (int64_t query_idx = 0; query_idx < query.num_nodes; ++query_idx) {
        const int64_t count = sim_count[static_cast<size_t>(query_idx)];
        pair_count += count;
        if (count > 0) {
            active_rows.push_back(query_idx);
        }
    }

    std::vector<Pair> pairs;
    if (pair_count == 0) {
        return pairs;
    }
    if (max_pairs <= 0 || pair_count <= max_pairs) {
        pairs.reserve(static_cast<size_t>(pair_count));
        for (int64_t query_idx = 0; query_idx < query.num_nodes; ++query_idx) {
            for (int64_t data_idx : candidates_for_row(sim[static_cast<size_t>(query_idx)], data.num_nodes)) {
                pairs.push_back({query_idx, data_idx});
            }
        }
        return pairs;
    }

    const int64_t per_row = std::max<int64_t>(1, (max_pairs + static_cast<int64_t>(active_rows.size()) - 1) / static_cast<int64_t>(active_rows.size()));
    std::vector<int64_t> col_counts(static_cast<size_t>(data.num_nodes), 0);
    for (int64_t query_idx = 0; query_idx < query.num_nodes; ++query_idx) {
        for (int64_t data_idx : candidates_for_row(sim[static_cast<size_t>(query_idx)], data.num_nodes)) {
            ++col_counts[static_cast<size_t>(data_idx)];
        }
    }

    pairs.reserve(static_cast<size_t>(max_pairs));
    for (int64_t query_idx : active_rows) {
        const int64_t room = max_pairs - static_cast<int64_t>(pairs.size());
        if (room <= 0) {
            break;
        }
        std::vector<int64_t> candidates = candidates_for_row(sim[static_cast<size_t>(query_idx)], data.num_nodes);
        const int64_t keep_count = std::min(per_row, std::min(room, static_cast<int64_t>(candidates.size())));
        if (static_cast<int64_t>(candidates.size()) > keep_count) {
            const double q_degree = static_cast<double>(query.degree[static_cast<size_t>(query_idx)]);
            std::stable_sort(candidates.begin(), candidates.end(), [&](int64_t lhs, int64_t rhs) {
                const double lhs_gap = std::max(0.0, q_degree - static_cast<double>(data.degree[static_cast<size_t>(lhs)]));
                const double rhs_gap = std::max(0.0, q_degree - static_cast<double>(data.degree[static_cast<size_t>(rhs)]));
                const double lhs_score = lhs_gap + 0.05 * static_cast<double>(col_counts[static_cast<size_t>(lhs)]) / std::max<int64_t>(query.num_nodes, 1);
                const double rhs_score = rhs_gap + 0.05 * static_cast<double>(col_counts[static_cast<size_t>(rhs)]) / std::max<int64_t>(query.num_nodes, 1);
                if (lhs_score == rhs_score) {
                    return lhs < rhs;
                }
                return lhs_score < rhs_score;
            });
        }
        for (int64_t i = 0; i < keep_count; ++i) {
            pairs.push_back({query_idx, candidates[static_cast<size_t>(i)]});
        }
    }
    return pairs;
}

torch::Tensor pair_tensor_from_pairs(const std::vector<Pair>& pairs) {
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto pair_index = torch::empty({static_cast<int64_t>(pairs.size()), 2}, options);
    int64_t* out = pair_index.data_ptr<int64_t>();
    for (int64_t pair_id = 0; pair_id < static_cast<int64_t>(pairs.size()); ++pair_id) {
        const Pair& pair = pairs[static_cast<size_t>(pair_id)];
        out[2 * pair_id] = pair.q;
        out[2 * pair_id + 1] = pair.g;
    }
    return pair_index;
}

}

torch::Tensor build_association_edges(
    torch::Tensor query_labels,
    torch::Tensor query_edge_index,
    torch::Tensor data_labels,
    torch::Tensor data_edge_index,
    torch::Tensor pair_index,
    int64_t max_edges);

py::tuple compute_dualsim_candidates(
    torch::Tensor query_labels,
    torch::Tensor query_edge_index,
    torch::Tensor data_labels,
    torch::Tensor data_edge_index,
    int64_t maxloop,
    int64_t max_pairs) {
    const Graph query = build_graph(query_labels, query_edge_index);
    const Graph data = build_graph(data_labels, data_edge_index);

    auto empty_pairs = torch::empty(
        {0, 2},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    if (query.num_nodes == 0 || data.num_nodes == 0) {
        return py::make_tuple(empty_pairs, true);
    }

    std::vector<Bitset> sim;
    std::vector<int64_t> sim_count;
    if (!dualsim(query, data, maxloop, sim, sim_count)) {
        return py::make_tuple(empty_pairs, true);
    }

    std::vector<Pair> pairs = collect_pairs(query, data, sim, sim_count, max_pairs);
    return py::make_tuple(pair_tensor_from_pairs(pairs), false);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "compute_dualsim_candidates",
        &compute_dualsim_candidates);
    m.def(
        "build_association_edges",
        &build_association_edges);
}

