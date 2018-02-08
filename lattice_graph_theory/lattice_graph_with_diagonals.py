#! /usr/bin/python2.7

import numpy as np

import matplotlib.pyplot as plt

from copy import deepcopy
from cycler import cycler

def find_edges_diagonals(h, w, nodes, edges):
    nodes = deepcopy(nodes)
    node_now = np.random.randint(0, h*w)
    # node_now = h//2*w+w//2
    # print("node_now: {}".format(node_now))

    choosen_nodes = [node_now]
    choosen_edges = []

    while len(nodes[node_now]) > 0:
        nodes_available = nodes[node_now]
        node_next_idx = np.random.randint(0, len(nodes_available))
        next_edge = nodes_available[node_next_idx]
        # print("node_now: {}, next_edge: {}, nodes_available: {}".format(node_now, next_edge, nodes_available))

        n1, n2 = edges[next_edge]
        nodes[n1].remove(next_edge)
        nodes[n2].remove(next_edge)
        
        if n1 == node_now:
            node_now = n2
            choosen_nodes.append(n2)
            choosen_edges.append((n1, n2))
        else:
            node_now = n1
            choosen_nodes.append(n1)
            choosen_edges.append((n2, n1))

    return choosen_nodes, choosen_edges

def find_most_full_squares(h, w, nodes, edges):
    max_len = 0
    best_choosen_nodes = []
    best_choosen_edges = []
    for i in xrange(0, 10000):
        choosen_nodes, choosen_edges = find_edges_diagonals(h, w, nodes, edges)
        if max_len < len(choosen_nodes):
            max_len = len(choosen_nodes)
            print("i: {}, max_len: {}".format(i, max_len))
            best_choosen_nodes = choosen_nodes
            best_choosen_edges = choosen_edges
            # plot_graph(h, w, choosen_nodes)

    return best_choosen_nodes, best_choosen_edges

def plot_graph(h, w, choosen_nodes):
    choosen_nodes_arr = np.array(choosen_nodes)
    x = choosen_nodes_arr%w
    y = choosen_nodes_arr//w

    # print("x:\n{}".format(x))
    # print("y:\n{}".format(y))

    plt.figure()
    plt.xlim([0, w-1])
    plt.ylim([0, h-1])
    plt.axis('equal')
    plt.plot(x, y, "b-")
    # plt.ion()
    # plt.show()
    # for i in xrange(2, len(x)+1):
    #     # plt.clf()
    #     plt.plot(x[i-2:i], y[i-2:i], "b-")
    #     # plt.show(block=False)
    #     # plt.show()
    #     plt.draw()
    #     # plt.pause(0.2)
    #     plt.pause(0.008)
    # plt.ioff()
    plt.show()

h = 5
w = 5

lat_nums = np.arange(0, h*w).reshape((h, w))
print("lat_nums:\n{}".format(lat_nums))

v_edges = [(w*y+x, w*(y+1)+x) for y in xrange(0, h-1) for x in xrange(0, w)]
h_edges = [(w*y+x, w*y+x+1) for y in xrange(0, h) for x in xrange(0, w-1)]
d_edges = [(w*y+x, w*(y+1)+x+1) for y in xrange(0, h-1) for x in xrange(0, w-1)]
t_edges = [(w*(y+1)+x, w*y+x+1) for y in xrange(0, h-1) for x in xrange(0, w-1)]

v_edge_coord_1 = {edge: (edge[0]//w, edge[0]%w) for edge in v_edges}
v_edge_coord_2 = {tuple(edge[::-1]): (edge[0]//w, edge[0]%w) for edge in v_edges}
h_edge_coord_1 = {edge: (edge[0]//w, edge[0]%w) for edge in h_edges}
h_edge_coord_2 = {tuple(edge[::-1]): (edge[0]//w, edge[0]%w) for edge in h_edges}
d_edge_coord_1 = {edge: (edge[0]//w, edge[0]%w) for edge in d_edges}
d_edge_coord_2 = {tuple(edge[::-1]): (edge[0]//w, edge[0]%w) for edge in d_edges}
t_edge_coord_1 = {edge: (edge[0]//w-1, edge[0]%w) for edge in t_edges}
t_edge_coord_2 = {tuple(edge[::-1]): (edge[0]//w-1, edge[0]%w) for edge in t_edges}

v_edge_coord = dict(v_edge_coord_1, **v_edge_coord_2)
h_edge_coord = dict(h_edge_coord_1, **h_edge_coord_2)
d_edge_coord = dict(d_edge_coord_1, **d_edge_coord_2)
t_edge_coord = dict(t_edge_coord_1, **t_edge_coord_2)

# print("v_edges:\n{}".format(v_edges))
# print("h_edges:\n{}".format(h_edges))
# print("d_edges:\n{}".format(d_edges))
# print("t_edges:\n{}".format(t_edges))

# print("v_edge_coord:\n{}".format(v_edge_coord))
# print("h_edge_coord:\n{}".format(h_edge_coord))
# print("d_edge_coord:\n{}".format(d_edge_coord))
# print("t_edge_coord:\n{}".format(t_edge_coord))

# edges = v_edges+h_edges #+d_edges+t_edges
edges = v_edges+h_edges+d_edges+t_edges
# edges_coordinates_1 = {edge: (edge[0]%w, edge[1]%w, edge[0]//w, edge[1]//w) for edge in edges}
# edges_coordinates_2 = {(edge[1], edge[0]): (edge[1]%w, edge[0]%w, edge[1]//w, edge[0]//w) for edge in edges}
# edges_coordinates = dict(edges_coordinates_1, **edges_coordinates_2)
nodes = {i: [] for i in xrange(0, h*w)}

for i, edge in enumerate(edges):
    n1, n2 = edge
    nodes[n1].append(i)
    nodes[n2].append(i)

# nodes = {key: np.array(nodes[key]) for key in nodes}

for key in nodes:
    print("node: {}, edges: {}".format(key, nodes[key]))

# choosen_nodes, choosen_edges = find_edges_diagonals(h, w, nodes, edges)
choosen_nodes, choosen_edges = find_most_full_squares(h, w, nodes, edges)
# print("choosen_edges:\n{}".format(choosen_edges))


# all_coordinates = []
# for edge in choosen_edges:
#     all_coordinates.append(edges_coordinates[edge])

# all_coordinates = np.array(all_coordinates)
# print("all_coordinates:\n{}".format(all_coordinates))

# x, y = np.array(choosen_edges).T

# all_points = np.vstack((x[:-1], x[1:], y[:-1], y[1:])).T
# print("all_points:\n{}".format(all_points))

# choosen_nodes_arr = np.array(choosen_nodes)
# x = choosen_nodes_arr%w
# y = choosen_nodes_arr//w

# print("x:\n{}".format(x))
# print("y:\n{}".format(y))

# plt.rc('axes', prop_cycle=(cycler('color', ['b']) +
#                            cycler('linestyle', ['-'])))
# # plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) +
# #                            cycler('linestyle', ['-', '--', ':', '-.'])))

# plt.figure()
# plt.plot(x, y, "b-")
# # for coordinate in all_coordinates:
# #     plt.plot(coordinate[:2], coordinate[2:])
# # plt.plot(*all_coordinates)
# plt.axis('equal')
# plt.show()

plot_graph(h, w, choosen_nodes)
