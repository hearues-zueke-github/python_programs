#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

from copy import deepcopy

from math import factorial as fac

sys.path.append("../combinatorics/")
from different_combinations import get_permutation_table

from utils_graph_theory import write_many_digraph_edges_as_dotfile

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"


if __name__ == "__main__":
    # arr0 = np.arange(0, 8)
    # arr1 = np.array([7, 0, 2, 4, 3, 5, 6, 1])
    # arr2 = np.array([4, 1, 0, 2, 3, 6, 5, 7])
    
    # print("arr0: {}".format(arr0))
    # print("arr1: {}".format(arr1))
    # print("arr2: {}".format(arr2))

    # arr1_arg = np.argsort(arr1)
    # arr2_arg = np.argsort(arr2)

    # print("arr1_arg: {}".format(arr1_arg))
    # print("arr2_arg: {}".format(arr2_arg))

    n = 5
    # arr = np.arange(0, n, dtype=np.uint8)
    # print("n: {}".format(n))
    # print("arr: {}".format(arr))

    # # idx1 = np.array([1, 0, 2, 3])
    # # idx2 = np.array([3, 0, 1, 2])
    # # idx3 = np.array([2, 0, 1, 3])
    # idx1 = np.array([1, 0, 2, 3, 4])
    # idx2 = np.array([4, 0, 1, 2, 3])
    # idx3 = np.array([2, 0, 1, 3, 4])
    # idxs_tbl = np.vstack((idx1, idx2, idx3))
    # # idxs_tbl = np.vstack((idx1, idx2))

    # amount_functions = idxs_tbl.shape[0]
    # print("amount_functions: {}".format(amount_functions))

    # arr1 = arr[idx1]
    # arr2 = arr[idx2]
    # print("arr1: {}".format(arr1))
    # print("arr2: {}".format(arr2))
    # print("arr1.dtype: {}".format(arr1.dtype))
    # print("arr2.dtype: {}".format(arr2.dtype))

    perm_tbl = get_permutation_table(n)
    l_t_perm_tbl = list(map(tuple, perm_tbl.tolist()))
    s_t_perm_tbl = set(l_t_perm_tbl)

    d_from_to_mapping = dict(list(map(
        lambda x: (x[0], list(map(tuple, x[1]))),
        zip(l_t_perm_tbl, perm_tbl[:, idxs_tbl].tolist())
    )))
    # print("d_from_to_mapping: {}".format(d_from_to_mapping))

    d_to_from_mapping = dict(list(map(
        lambda x: (x[0], list(map(tuple, x[1]))),
        zip(l_t_perm_tbl, perm_tbl[:, np.argsort(idxs_tbl, axis=1)].tolist())
    )))
    # print("d_to_from_mapping: {}".format(d_to_from_mapping))
    

    d_t_to_num = {t: i for i, t in enumerate(l_t_perm_tbl, 0)}

    d = {}
    for p1 in l_t_perm_tbl:
        for p2 in l_t_perm_tbl:
            print("p1: {}, p2: {}".format(p1, p2))
            d[(p1, p2)] = tuple(np.argsort(np.array(p1))[list(p2)].tolist())
    
    d_count_p1_p2_perms = {t: 0 for t in d_t_to_num}
    d_l_p1_p2_perms = {t: [] for t in d_t_to_num}
    for t_p1_p2, t in d.items():
        d_count_p1_p2_perms[t] += 1
        d_l_p1_p2_perms[t].append(t_p1_p2)


    # n = 4
    # idx1 = np.array([1, 0, 2, 3])
    # idx2 = np.array([3, 0, 1, 2])
    # idxs_tbl = np.vstack((idx1, idx2))

    # n = 5
    idx1 = np.array([1, 0, 2, 3, 4])
    idx2 = np.array([4, 0, 1, 2, 3])
    idxs_tbl = np.vstack((idx1, idx2))

    d_part = {}
    for p1 in perm_tbl:
        for p2 in idxs_tbl:
            d_part[(tuple(p1.tolist()), tuple(p2.tolist()))] = tuple(np.argsort(p1)[p2].tolist())

    d_part_num = {(d_t_to_num[t1], d_t_to_num[t2]): d_t_to_num[t12] for (t1, t2), t12 in d_part.items()}
    d_node_pair_edge = {(n1, n2): e1 for (n1, e1), n2 in d_part_num.items()}
    print("d_part_num: {}".format(d_part_num))
    # print("d_node_pair_edge: {}".format(d_node_pair_edge))

    # node_from, node_to = list(zip(*list(d_part_num.keys())))
    write_many_digraph_edges_as_dotfile(
        path=PATH_ROOT_DIR+'permutation_graph_part_n_{}.dot'.format(n),
        d_node_pair_edge=d_node_pair_edge
        # node_from=node_from,
        # node_to=node_to
    )

    def find_best_idxs(l_t_perm_tbl, d_from_to_mapping):
        t_start = l_t_perm_tbl[0]
        print("t_start: {}".format(t_start))

        i_pos = 0
        fac_n = perm_tbl.shape[0]
        l_idxs = np.array([-1]*fac_n, dtype=np.int8)
        l_t = [None]*fac_n
        l_t[0] = t_start
        s_free_t = set(s_t_perm_tbl)
        s_free_t.remove(t_start)
        print("s_free_t: {}".format(s_free_t))

        i_counter = 0
        i_pos_max = 0
        is_found_optimal_idxs = False
        while i_pos>=0:
            if i_pos_max<i_pos:
                i_pos_max = i_pos
            
            i_counter += 1
            if i_counter%100000==0:
                print(f"i_counter: {i_counter:10}, i_pos: {i_pos:3}, i_pos_max: {i_pos_max:3}")
            
            # print("i_pos: {}".format(i_pos))
            new_i = l_idxs[i_pos]+1
            if new_i>=amount_functions:
                l_idxs[i_pos] = -1
                s_free_t.add(l_t[i_pos])
                l_t[i_pos] = None
                i_pos -= 1
                continue

            # print("new_i: {}".format(new_i))

            t_from = l_t[i_pos]
            # print("t_from: {}".format(t_from))
            l_from_to = d_from_to_mapping[t_from]
            if i_pos<fac_n-1:
                is_found = False
                for i, t_to in enumerate(l_from_to[new_i:], new_i):
                    if t_to in s_free_t:
                        l_idxs[i_pos] = i
                        s_free_t.remove(t_to)
                        i_pos += 1
                        l_t[i_pos] = t_to
                        # print("i_pos: {}, l_idxs: {}".format(i_pos, l_idxs))
                        # print("- len(s_free_t): {}".format(len(s_free_t)))
                        # print("- l_t[:i_pos]: {}".format(l_t[:i_pos+1]))
                        is_found = True
                        break
                if is_found:
                    continue
            else:
                for i, t_to in enumerate(l_from_to[new_i:], new_i):
                    if t_to==t_start:
                        l_idxs[i_pos] = i
                        is_found_optimal_idxs = True
                        break
                if is_found_optimal_idxs:
                    break
            l_idxs[i_pos] = -1
            s_free_t.add(l_t[i_pos])
            l_t[i_pos] = None
            i_pos -= 1

        return is_found_optimal_idxs, l_idxs, l_t


    is_found_optimal_idxs, l_idxs, l_t = find_best_idxs(l_t_perm_tbl, d_from_to_mapping)

    if is_found_optimal_idxs:
        print('Found something!')
    else:
        print('Did not have found an optimal list of idxs for applying permuation functions!')

    sys.exit(0)

    idxs = np.arange(0, perm_tbl.shape[0])

    min_len_l_p_i = perm_tbl.shape[0]
    max_len_l_p_i = 0
    for i in range(0, 1000000):
        perm_tbl_perm = perm_tbl[np.random.permutation(idxs)]
        l_p_i = []
        for p1, p2 in zip(perm_tbl_perm, np.roll(perm_tbl_perm, -1, axis=0)):
            # print("p1: {}, p2: {}".format(p1, p2))
            p_i = np.argsort(p1)[p2]
            t_p_i = tuple(p_i.tolist())
            # print("- t_p_i: {}".format(t_p_i))
            l_p_i.append(t_p_i)

        len_l_p_i = len(set(l_p_i))
        # print("i: {}, len_l_p_i: {}".format(i, len_l_p_i))
        if min_len_l_p_i>len_l_p_i:
            min_len_l_p_i = len_l_p_i
            print("i: {}, min_len_l_p_i: {}".format(i, min_len_l_p_i))
        if max_len_l_p_i<len_l_p_i:
            max_len_l_p_i = len_l_p_i
            print("i: {}, max_len_l_p_i: {}".format(i, max_len_l_p_i))


    sys.exit(0)


    perm_tbl[idx1].tolist()

    l_t_perm_tbl = list(map(tuple, perm_tbl.tolist()))
    s_t_perm_tbl = set(l_t_perm_tbl)

    d_from_to_mapping = dict(list(map(
        lambda x: (x[0], list(map(tuple, x[1]))),
        zip(l_t_perm_tbl, perm_tbl[:, idxs_tbl].tolist())
    )))
    # print("d_from_to_mapping: {}".format(d_from_to_mapping))

    d_to_from_mapping = dict(list(map(
        lambda x: (x[0], list(map(tuple, x[1]))),
        zip(l_t_perm_tbl, perm_tbl[:, np.argsort(idxs_tbl, axis=1)].tolist())
    )))
    # print("d_to_from_mapping: {}".format(d_to_from_mapping))


    # do the algorithm for finding the most optimal l_idxs for using the functions!
    t_start = l_t_perm_tbl[0]
    print("t_start: {}".format(t_start))

    i_pos = 0
    fac_n = perm_tbl.shape[0]
    l_idxs = np.array([-1]*fac_n, dtype=np.int8)
    l_t = [None]*fac_n
    l_t[0] = t_start
    s_free_t = set(s_t_perm_tbl)
    s_free_t.remove(t_start)
    print("s_free_t: {}".format(s_free_t))

    i_counter = 0
    i_pos_max = 0
    is_found_optimal_idxs = False
    while i_pos>=0:
        if i_pos_max<i_pos:
            i_pos_max = i_pos
        
        i_counter += 1
        if i_counter%100000==0:
            print(f"i_counter: {i_counter:10}, i_pos: {i_pos:3}, i_pos_max: {i_pos_max:3}")
        
        # print("i_pos: {}".format(i_pos))
        new_i = l_idxs[i_pos]+1
        if new_i>=amount_functions:
            l_idxs[i_pos] = -1
            s_free_t.add(l_t[i_pos])
            l_t[i_pos] = None
            i_pos -= 1
            continue

        # print("new_i: {}".format(new_i))

        t_from = l_t[i_pos]
        # print("t_from: {}".format(t_from))
        l_from_to = d_from_to_mapping[t_from]
        if i_pos<fac_n-1:
            is_found = False
            for i, t_to in enumerate(l_from_to[new_i:], new_i):
                if t_to in s_free_t:
                    l_idxs[i_pos] = i
                    s_free_t.remove(t_to)
                    i_pos += 1
                    l_t[i_pos] = t_to
                    # print("i_pos: {}, l_idxs: {}".format(i_pos, l_idxs))
                    # print("- len(s_free_t): {}".format(len(s_free_t)))
                    # print("- l_t[:i_pos]: {}".format(l_t[:i_pos+1]))
                    is_found = True
                    break
            if is_found:
                continue
        else:
            for i, t_to in enumerate(l_from_to[new_i:], new_i):
                if t_to==t_start:
                    l_idxs[i_pos] = i
                    is_found_optimal_idxs = True
                    break
            if is_found_optimal_idxs:
                break
        l_idxs[i_pos] = -1
        s_free_t.add(l_t[i_pos])
        l_t[i_pos] = None
        i_pos -= 1

    if is_found_optimal_idxs:
        print('Found something!')
    else:
        print('Did not have found an optimal list of idxs for applying permutation functions!')
