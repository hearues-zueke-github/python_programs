import numpy as np

def get_cycles_of_1_directed_graph(edges_directed):
    nodes_from, nodes_to = list(zip(*edges_directed))
    all_nodes = sorted(set(nodes_from+nodes_to))

    unique_nodes_from, counts = np.unique(nodes_from, return_counts=True)
    assert np.all(counts==1)

    edges_directed_dict = {n1: n2 for n1, n2 in edges_directed}

    all_available_nodes = set(all_nodes)
    list_of_cycles = []

    while len(all_available_nodes) > 0:
        node_now = all_available_nodes.pop()
        lst_nodes = [node_now]

        is_found_cycle = False
        while True:
            node_next = edges_directed_dict[node_now]
            node_now = node_next
            if not node_next in all_available_nodes:
                if node_next in lst_nodes:
                    lst_nodes = lst_nodes[lst_nodes.index(node_next):]
                    argmin = np.argmin(lst_nodes)
                    lst_nodes = lst_nodes[argmin:]+lst_nodes[:argmin]
                    is_found_cycle = True
                break
            lst_nodes.append(node_next)
            all_available_nodes.remove(node_next)

        if is_found_cycle:
            list_of_cycles.append(lst_nodes)

    list_of_cycles_sorted = sorted(list_of_cycles, key=lambda x: (len(x), x))
    return list_of_cycles_sorted


def write_digraph_as_dotfile(path, arr_x, arr_y):
    with open(path, 'w') as f:
        f.write('digraph {\n')
        for x in arr_x:
            f.write(f'  x{x}[label="{x}"];\n')
        f.write('\n')
        for x, y in zip(arr_x, arr_y):
            f.write(f'  x{x} -> x{y};\n')

        f.write('}\n')


# d_node_pair_edge = {(0, 1): 2, ...}
# Node 0 to Node 1 with the Edge 2, etc.
# def write_many_digraph_as_dotfile(path, node_from, node_to):
def write_many_digraph_edges_as_dotfile(path, d_node_pair_edge):
    with open(path, 'w') as f:
        f.write('digraph {\n')
        for x in sorted(set(list(map(lambda x: x[0], d_node_pair_edge.keys())))):
            f.write(f'  x{x}[label="{x}"];\n')
        f.write('\n')
        for (n1, n2), e in d_node_pair_edge.items():
        # for x, y in zip(node_from, node_to):
            f.write(f'  x{n1} -> x{n2} [label="{e}"];\n')
            # f.write(f'  x{x} -> x{y} [label="{e}"];\n')

        f.write('}\n')
