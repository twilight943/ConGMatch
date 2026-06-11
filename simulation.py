def _build_graph(parts):
    vid, vlbl, srcid, dstid = parts
    nodes = [int(node) for node in vid]
    labels = {}
    for idx, node in enumerate(nodes):
        labels[node] = vlbl[idx] if idx < len(vlbl) else None
    neighbors = {node: set() for node in nodes}
    for src, dst in zip(srcid, dstid):
        src = int(src)
        dst = int(dst)
        if src in neighbors and dst in neighbors and src != dst:
            neighbors[src].add(dst)
    return nodes, labels, neighbors


def _compatible(query_label, data_label):
    if query_label is None or data_label is None:
        return True
    return query_label == data_label


def dualsim_filter_qt(query_list, data_list, maxloop=10):
    query_nodes, query_labels, query_neighbors = _build_graph(query_list)
    data_nodes, data_labels, data_neighbors = _build_graph(data_list)
    query_pos = {node: idx for idx, node in enumerate(query_nodes)}
    data_pos = {node: idx for idx, node in enumerate(data_nodes)}
    candidates = {}
    for query_node in query_nodes:
        query_label = query_labels.get(query_node)
        candidates[query_node] = {
            data_node
            for data_node in data_nodes
            if _compatible(query_label, data_labels.get(data_node))
        }
    loop_count = max(1, int(maxloop or 1))
    for _ in range(loop_count):
        changed = False
        updated = {node: set(values) for node, values in candidates.items()}
        for query_node in query_nodes:
            for data_node in list(candidates[query_node]):
                valid = True
                for query_neighbor in query_neighbors.get(query_node, set()):
                    neighbor_candidates = candidates.get(query_neighbor, set())
                    if not neighbor_candidates or not data_neighbors.get(data_node, set()).intersection(neighbor_candidates):
                        valid = False
                        break
                if not valid:
                    updated[query_node].discard(data_node)
                    changed = True
        candidates = updated
        if not changed:
            break
    return {
        query_pos[query_node]: [data_pos[data_node] for data_node in sorted(candidates[query_node]) if data_node in data_pos]
        for query_node in query_nodes
    }
