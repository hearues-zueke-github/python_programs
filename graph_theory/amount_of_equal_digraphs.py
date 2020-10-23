#! /usr/bin/python3

# -*- coding: utf-8 -*-

# Some other needed imports
import os
import sys

import matplotlib.pyplot as plt

import numpy as np

from copy import deepcopy
from math import factorial as fac

sys.path.append("../combinatorics/")
from different_combinations import get_all_combinations_repeat, get_permutation_table

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"

def write_digraph_as_dotfile(path, arr_x, arr_y):
    with open(path, 'w') as f:
        f.write('digraph {\n')
        for x in arr_x:
            f.write(f'  x{x}[label="{x}"];\n')
        f.write('\n')
        for x, y in zip(arr_x, arr_y):
            f.write(f'  x{x} -> x{y};\n')

        f.write('}\n')


def get_basic_digraph(arr_x, arr_y):
    # find all cycles
    d = {x: y for x, y in zip(arr_x, arr_y)}
    d_inv = {}
    for k, v in d.items():
        if not v in d_inv:
            d_inv[v] = []
        d_inv[v].append(k)
    # print("d: {}".format(d))
    # print("d_inv: {}".format(d_inv))

    l_true_cycles = []
    dc = deepcopy(d)
    while len(dc)>0:
        for n1 in dc:
            break
        l = [n1]
        is_found_cycle = False
        while True:
            n2 = dc[n1]
            if not n2 in dc:
                break
            if n2 in l:
                l.append(n2)
                is_found_cycle = True
                break
            l.append(n2)
            n1 = n2
        # remove all nodes from l except the last one
        if is_found_cycle:
            for node in l[:-1]:
                del dc[node]
            l_true_cycles.append(l[l.index(l[-1]):-1])
        else:
            for node in l:
                del dc[node]

    # print("l_true_cycles: {}".format(l_true_cycles))

    d_cycles = {}
    for l in l_true_cycles:
        length = len(l)
        if not length in d_cycles:
            d_cycles[length] = []
        d_cycles[length].append(l)

    # print("d_cycles: {}".format(d_cycles))

    l_root_nodes = []
    for l in l_true_cycles:
        for v in l:
            l_root_nodes.append(v)
    # print("l_root_nodes: {}".format(l_root_nodes))

    d_inv_true = {}
    for node, l_node_prev in d_inv.items():
        l = deepcopy(l_node_prev)
        is_node_prev_in_root_nodes = False
        for node_prev in l:
            if node_prev in l_root_nodes:
                is_node_prev_in_root_nodes = True
                break
        if is_node_prev_in_root_nodes:
            l.remove(node_prev)
        if len(l)>0:
            d_inv_true[node] = l
    # print("d_inv_true: {}".format(d_inv_true))

    d_depth = {k: -1 for k in d}
    d_amount_nodes = {k: -1 for k in d}

    def go_tree_rec(node_now):
        if not node_now in d_inv_true:
            d_depth[node_now] = 0
            d_amount_nodes[node_now] = 0
            return

        amount_nodes = 0
        depth = 0
        for node_prev in d_inv_true[node_now]:
            go_tree_rec(node_prev)
            amount_nodes += d_amount_nodes[node_prev]+1
            max_depth = d_depth[node_prev]+1
            if depth<max_depth:
                depth = max_depth
        d_amount_nodes[node_now] = amount_nodes
        d_depth[node_now] = depth


    for root_node in d_inv_true.keys():
        go_tree_rec(root_node)

    d_stats = {k: (d_amount_nodes[k], d_depth[k], len(d_inv_true[k]) if k in d_inv_true else -1 if k in l_root_nodes else 0) for k in d}
    # print("d_stats: {}".format(d_stats))
    
    l_true_root_nodes = []
    for root_node in l_root_nodes:
        if d_stats[root_node][2]>0:
            l_true_root_nodes.append(root_node)
    # print("l_true_root_nodes: {}".format(l_true_root_nodes))

    # find the correct mentioning of the nodes, started from the root_node!
    def construct_node_sequence_sorted(node_now, l_correct_nodes_sorted):
        if not node_now in d_inv_true:
            # l_correct_nodes_sorted.append(node_now)
            return
        l_node_prev = d_inv_true[node_now]
        l_stats = []
        for node_prev in l_node_prev:
            l_stats.append((node_prev, )+d_stats[node_prev])
        l_stats_sorted = sorted(l_stats, key=lambda x: (x[1], x[2], x[3]), reverse=True)
        for node_prev, _, depth, _ in l_stats_sorted:
            construct_node_sequence_sorted(node_prev, l_correct_nodes_sorted)
            l_correct_nodes_sorted.append((node_prev, depth))


    d_l_root_node_sorted = {}
    for root_node in l_true_root_nodes:
        l_correct_nodes_sorted = []
        construct_node_sequence_sorted(root_node, l_correct_nodes_sorted)
        # extract the nodes in sorted form and create the base form of the tree too!
        l_base_form = []
        l_nodes_sequence = []
        for i, (node, depth) in enumerate(l_correct_nodes_sorted, 0):
            l_base_form.append((i, depth))
            l_nodes_sequence.append(node)
        d_l_root_node_sorted[root_node] = (l_base_form, l_nodes_sequence)
        # d_l_root_node_sorted[root_node] = l_correct_nodes_sorted
    # print("d_l_root_node_sorted: {}".format(d_l_root_node_sorted))

    # combine all trees and cycles together, and create the mapping from the
    # special case to the base case (most simplified base node graph)
    l_tree_nodes = []
    l_root_nodes = []
    for length in sorted(d_cycles.keys()):
        l_cycles = d_cycles[length]

        # print("\nlength: {}".format(length))
        # l_used_
        l_tree_root_cycles = []
        for j, l_cycle in enumerate(l_cycles, 0):
            tree_nodes_base_temp = []
            tree_nodes_seq_temp = []
            # l_used_root_nodes = []
            for node_now in l_cycle:
            # for k, node_now in enumerate(l_cycle, 0):
                if not node_now in d_l_root_node_sorted:
                    tree_nodes_base_temp.append([])
                    tree_nodes_seq_temp.append([])
                    # tree_nodes_base_temp.append((k, []))
                    continue
                # l_used_root_nodes.append(node_now)
                l_base_form, l_nodes_sequence = d_l_root_node_sorted[node_now]
                # l_tree_nodes.append(l_nodes_sequence)
                tree_nodes_base_temp.append(l_base_form)
                tree_nodes_seq_temp.append(l_nodes_sequence)
                # tree_nodes_base_temp.append((k, l_base_form))

            # tree_nodes_base_temp_sorted = sorted(tree_nodes_base_temp, key=lambda x: (x[1], x[0]))
            # print("tree_nodes_base_temp_sorted: {}".format(tree_nodes_base_temp_sorted))

            # # get all idxs of the smallest same tree_node_base
            # l_smallest = tree_nodes_base_temp_sorted[0][1]
            # l_idxs = []
            # for i, l in tree_nodes_base_temp_sorted:
            #     if l!=l_smallest:
            #         break
            #     l_idxs.append(i)
            # print("l_idxs: {}".format(l_idxs))

            # check, if 

            # find the smallest sequence of the tree_nodes_base_temp list!
            tree_nodes_base_temp_best = tree_nodes_base_temp
            i_best = 0
            # tree_nodes_seq_temp_best = deepcopy(tree_nodes_seq_temp)
            for i in range(1, len(tree_nodes_base_temp)):
                tree_nodes_base_rearranged = tree_nodes_base_temp[i:]+tree_nodes_base_temp[:i]
                if tree_nodes_base_temp_best>tree_nodes_base_rearranged:
                    tree_nodes_base_temp_best = tree_nodes_base_rearranged
                    i_best = i
                # tree_nodes_seq_rearranged = tree_nodes_seq_temp[i:]+tree_nodes_seq_temp[:i]

            tree_nodes_base_temp_rearranged = tree_nodes_base_temp_best
            tree_nodes_seq_temp_rearranged = tree_nodes_seq_temp[i_best:]+tree_nodes_seq_temp[:i_best]
            l_cycle_rearranged = l_cycle if i_best==0 else l_cycle[i_best:]+l_cycle[:i_best]

            # index_smallest = 0
            # index2_smallest = 0
            # if len(tree_nodes_base_temp)>0:
            #     l_best = tree_nodes_base_temp[0][1]
            #     for i2, (i, l) in enumerate(tree_nodes_base_temp, 0):
            #         if l_best>l:
            #             l_best = l
            #             index_smallest = i
            #             index2_smallest = i2

            # l_cycle_rearranged = l_cycle if index_smallest==0 else l_cycle[index_smallest:]+l_cycle[:index_smallest]

            # if index_smallest==0:
            #     tree_nodes_base_temp_rearranged = tree_nodes_base_temp
            #     tree_nodes_seq_temp_rearranged = tree_nodes_seq_temp
            #     # tree_used_root_nodes_rearranged = l_used_root_nodes
            # else:
            #     tree_nodes_base_temp_rearranged = [(i2, l) for i2, (i, l) in enumerate(tree_nodes_base_temp[index2_smallest:]+tree_nodes_base_temp[:index2_smallest], 0)]
            #     tree_nodes_seq_temp_rearranged = tree_nodes_seq_temp[index2_smallest:]+tree_nodes_seq_temp[:index2_smallest]
            #     # tree_used_root_nodes_rearranged = l_used_root_nodes[index2_smallest:]+l_used_root_nodes[:index2_smallest]


            l_tree_root_cycles.append((tree_nodes_base_temp_rearranged, tree_nodes_seq_temp_rearranged, l_cycle_rearranged))
            # l_tree_root_cycles.append((tree_nodes_base_temp_rearranged, tree_nodes_seq_temp_rearranged, tree_used_root_nodes_rearranged, l_cycle_rearranged))
            
        l_tree_root_cycles_sorted = sorted(l_tree_root_cycles, key=lambda x: x[0])
        
        # print("l_tree_root_cycles_sorted: {}".format(l_tree_root_cycles_sorted))
        
        for _, tree_nodes_seq_temp_rearranged, l_cycle_rearranged in l_tree_root_cycles_sorted:
        # for _, tree_nodes_seq_temp_rearranged, _, l_cycle_rearranged in l_tree_root_cycles_sorted:
            l_tree_nodes.extend([l for l in tree_nodes_seq_temp_rearranged if len(l)>0])
            l_root_nodes.append(l_cycle_rearranged)

    # print("l_tree_nodes: {}".format(l_tree_nodes))
    # print("l_root_nodes: {}".format(l_root_nodes))

    nodes_new_order = np.array([v for l in l_tree_nodes for v in l]+[v for l in l_root_nodes for v in l])
    nodes_new_order_sort = np.argsort(nodes_new_order)

    # print("nodes_new_order: {}".format(nodes_new_order.tolist()))
    arr_x_new = nodes_new_order_sort[arr_x]
    arr_y_new = nodes_new_order_sort[arr_y]

    idxs = np.argsort(arr_x_new)
    arr_x_new = arr_x_new[idxs]
    arr_y_new = arr_y_new[idxs]

    return arr_x_new, arr_y_new


def try_some_cases(n, tries_new_arr=100, tries_per_perm=500):
    # n = 200
    # tries_new_arr = 100
    # tries_per_perm = 500
    arr_perm_row = np.arange(0, n)
    # arr_perm = get_permutation_table(n)
    # arr_perm_len = arr_perm.shape[0]

    for try_new_arr in range(0, tries_new_arr):
        arr_x = np.arange(0, n)
        arr_y = np.random.randint(0, n, (n, ))

        print("arr_x: {}".format(arr_x.tolist()))
        print("arr_y: {}".format(arr_y.tolist()))

        arr_perm_row = np.random.permutation(arr_perm_row)
        # arr_perm_row = arr_perm[0]

        arr_x_perm = arr_perm_row[arr_x]
        arr_y_perm = arr_perm_row[arr_y]
        arr_x_new, arr_y_new = get_basic_digraph(arr_x_perm, arr_y_perm)

        arr_y_new_ref = arr_y_new
        

        # for row_num in np.random.randint(0, arr_perm_len, (tries_per_perm, )):
        # for row_num in np.random.randint(0, arr_perm_len, (tries_per_perm, )):
        # for arr_perm_row in arr_perm[1:]:
            # arr_perm_row = arr_perm[row_num]
        for try_per_perm in range(0, tries_per_perm):
            arr_perm_row = np.random.permutation(arr_perm_row)

            arr_x_perm = arr_perm_row[arr_x]
            arr_y_perm = arr_perm_row[arr_y]
            arr_x_new, arr_y_new = get_basic_digraph(arr_x_perm, arr_y_perm)

            if not np.all(arr_y_new==arr_y_new_ref):
                print('Something was wrong!')
                print("arr_x: {}".format(arr_x))
                print("arr_y: {}".format(arr_y))
                print("arr_perm_row: {}".format(arr_perm_row))
                print("arr_x_perm: {}".format(arr_x_perm))
                print("arr_y_perm: {}".format(arr_y_perm))
                assert False

        print('Finished: try_new_arr: {}'.format(try_new_arr))


if __name__=='__main__':
    # try_some_cases(n=100)
    # sys.exit(0)

    n = 6
    arr = get_all_combinations_repeat(n, n)

    different_combs = set()
    mapping_from_to = []

    arr_x = np.arange(0, n)
    for i, arr_y in enumerate(arr, 0):
        if i%10000==0:
            print("i: {}".format(i))
        arr_x_new, arr_y_new = get_basic_digraph(arr_x, arr_y)
        t = tuple(arr_y_new.tolist())
        if not t in different_combs:
            different_combs.add(t)
            mapping_from_to.append([tuple(arr_y.tolist()), t])
            print("t: {}".format(t))

    mapping_from_to = sorted(mapping_from_to, key=lambda x: x[1])
    l_different_combs = sorted(list(different_combs))
    print("l_different_combs: {}".format(l_different_combs))
    print("mapping_from_to: {}".format(mapping_from_to))
    print("len(different_combs): {}".format(len(different_combs)))

    sys.exit(0)

    # n = 50
    # arr_x = np.arange(0, n)
    # arr_y = np.random.randint(0, n, (n, ))

    n = 3
    arr_x = np.arange(0, n)
    arr_y = np.array([1, 0, 0])
    # arr_y = np.array([1, 0, 1])

    # arr_y = np.array([1, 2, 1])
    # arr_y = np.array([1, 2, 1])
    # arr_y = np.array([1, 2, 3, 4, 0, 1, 1, 6])
    # arr_y = np.array([0, 0, 0, 2, 2])

    # remap = np.random.permutation(np.arange(0, n))
    # arr_x = remap[arr_x]
    # arr_y = remap[arr_y]

    print("arr_x: {}".format(arr_x.tolist()))
    print("arr_y: {}".format(arr_y.tolist()))
    write_digraph_as_dotfile(path='simple_digraph_n_{}.dot'.format(n), arr_x=arr_x, arr_y=arr_y)

    arr_x_new, arr_y_new = get_basic_digraph(arr_x, arr_y)

    print("arr_x_new: {}".format(arr_x_new.tolist()))
    print("arr_y_new: {}".format(arr_y_new.tolist()))
    write_digraph_as_dotfile(path='simple_digraph_n_{}_simplified.dot'.format(n), arr_x=arr_x_new, arr_y=arr_y_new)
