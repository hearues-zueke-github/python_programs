#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

from copy import deepcopy

from math import factorial as fac

sys.path.append("../combinatorics/")
from different_combinations import get_permutation_table

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"


if __name__ == "__main__":
    # print("n: {}".format(n))

    # n = 3
    # idx1 = np.array([1, 0, 2])
    # idx2 = np.array([2, 0, 1])
    # idxs_tbl = np.vstack((idx1, idx2))

    # n = 4
    # idx1 = np.array([1, 0, 2, 3])
    # idx2 = np.array([2, 0, 1, 3])
    # idx3 = np.array([3, 0, 1, 2])
    # # idx2 = np.array([3, 2, 0, 1])
    # # idxs_tbl = np.vstack((idx1, idx2))
    # idxs_tbl = np.vstack((idx1, idx2, idx3))
    
    n = 5
    idx1 = np.array([1, 0, 2, 3, 4])
    idx2 = np.array([4, 0, 1, 2, 3])
    idxs_tbl = np.vstack((idx1, idx2))
    # idx3 = np.array([2, 0, 1, 3, 4])
    # idxs_tbl = np.vstack((idx1, idx2, idx3))

    # n = 6
    # idx1 = np.array([1, 0, 2, 3, 4, 5])
    # idx2 = np.array([5, 0, 1, 2, 3, 4])
    # idxs_tbl = np.vstack((idx1, idx2))

    """
    idxs_tbl = np.array([[1, 0, 2, 3, 4], [4, 0, 1, 2, 3]])
    # idxs_tbl = np.array([[2, 0, 1, 3, 4], [0, 1, 4, 2, 3]])
    l_idxs = [
        0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0,
        1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1,
        1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1,
        1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0,
        1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1,
    ]
    [0 1 0 1 1 0 1 1 0 1 1 1 1 0 1 0 1 0 1 1 0 1 0 1 1 1 1 0 1 0 1 1 1 1 0 1 0
 1 0 1 1 0 1 1 1 1 0 1 1 1 1 0 1 0 1 1 0 1 1 0 1 1 1 1 0 1 1 0 1 1 0 1 1 1
 1 0 1 1 1 1 0 1 1 0 1 1 0 1 1 1 1 0 1 1 1 1 0 1 1 1 1 0 1 0 1 0 1 1 0 1 1
 0 1 0 1 0 1 1 1 1]
    [1 1 0 1 0 1 0 1 1 1 1 0 1 1 1 1 0 1 1 1 1 0 1 1 0 1 1 0 1 1 1 1 0 1 1 1 1
 0 1 1 0 1 1 0 1 1 1 1 0 1 1 0 1 1 0 1 0 1 1 1 1 0 1 1 1 1 0 1 1 0 1 0 1 0
 1 1 1 1 0 1 0 1 1 1 1 0 1 0 1 1 0 1 0 1 0 1 1 1 1 0 1 1 0 1 1 0 1 0 1 1 1
 1 0 1 0 1 0 1 1 0]
    [0 1 1 0 1 1 0 1 0 1 1 1 1 0 1 1 1 1 0 1 1 0 1 0 1 0 1 1 1 1 0 1 0 1 1 1 1
 0 1 0 1 1 0 1 0 1 0 1 1 1 1 0 1 1 0 1 1 0 1 0 1 1 1 1 0 1 0 1 0 1 1 0 1 1
 0 1 0 1 0 1 1 1 1 0 1 1 1 1 0 1 1 1 1 0 1 1 0 1 1 0 1 1 1 1 0 1 1 1 1 0 1
 1 0 1 1 0 1 1 1 1]

    l_t = [
        (0, 1, 2, 3, 4), (1, 0, 2, 3, 4), (4, 1, 0, 2, 3), (1, 4, 0, 2, 3), (3, 1, 4, 0, 2), (1, 3, 4, 0, 2),
        (2, 1, 3, 4, 0), (0, 2, 1, 3, 4), (2, 0, 1, 3, 4), (4, 2, 0, 1, 3), (2, 4, 0, 1, 3), (3, 2, 4, 0, 1),
        (1, 3, 2, 4, 0), (0, 1, 3, 2, 4), (4, 0, 1, 3, 2), (0, 4, 1, 3, 2), (2, 0, 4, 1, 3), (0, 2, 4, 1, 3),
        (3, 0, 2, 4, 1), (1, 3, 0, 2, 4), (4, 1, 3, 0, 2), (2, 4, 1, 3, 0), (4, 2, 1, 3, 0), (0, 4, 2, 1, 3),
        (4, 0, 2, 1, 3), (3, 4, 0, 2, 1), (4, 3, 0, 2, 1), (1, 4, 3, 0, 2), (2, 1, 4, 3, 0), (1, 2, 4, 3, 0),
        (0, 1, 2, 4, 3), (3, 0, 1, 2, 4), (4, 3, 0, 1, 2), (2, 4, 3, 0, 1), (4, 2, 3, 0, 1), (1, 4, 2, 3, 0),
        (0, 1, 4, 2, 3), (3, 0, 1, 4, 2), (2, 3, 0, 1, 4), (3, 2, 0, 1, 4), (4, 3, 2, 0, 1), (3, 4, 2, 0, 1),
        (1, 3, 4, 2, 0), (0, 1, 3, 4, 2), (1, 0, 3, 4, 2), (2, 1, 0, 3, 4), (4, 2, 1, 0, 3), (2, 4, 1, 0, 3),
        (3, 2, 4, 1, 0), (0, 3, 2, 4, 1), (1, 0, 3, 2, 4), (4, 1, 0, 3, 2), (1, 4, 0, 3, 2), (2, 1, 4, 0, 3),
        (3, 2, 1, 4, 0), (2, 3, 1, 4, 0), (0, 2, 3, 1, 4), (4, 0, 2, 3, 1), (0, 4, 2, 3, 1), (1, 0, 4, 2, 3),
        (3, 1, 0, 4, 2), (2, 3, 1, 0, 4), (4, 2, 3, 1, 0), (2, 4, 3, 1, 0), (0, 2, 4, 3, 1), (1, 0, 2, 4, 3),
        (3, 1, 0, 2, 4), (4, 3, 1, 0, 2), (3, 4, 1, 0, 2), (2, 3, 4, 1, 0), (0, 2, 3, 4, 1), (2, 0, 3, 4, 1),
        (1, 2, 0, 3, 4), (4, 1, 2, 0, 3), (1, 4, 2, 0, 3), (3, 1, 4, 2, 0), (0, 3, 1, 4, 2), (2, 0, 3, 1, 4),
        (4, 2, 0, 3, 1), (2, 4, 0, 3, 1), (1, 2, 4, 0, 3), (3, 1, 2, 4, 0), (0, 3, 1, 2, 4), (4, 0, 3, 1, 2),
        (0, 4, 3, 1, 2), (2, 0, 4, 3, 1), (1, 2, 0, 4, 3), (3, 1, 2, 0, 4), (4, 3, 1, 2, 0), (3, 4, 1, 2, 0),
        (0, 3, 4, 1, 2), (3, 0, 4, 1, 2), (2, 3, 0, 4, 1), (3, 2, 0, 4, 1), (1, 3, 2, 0, 4), (4, 1, 3, 2, 0),
        (1, 4, 3, 2, 0), (0, 1, 4, 3, 2), (2, 0, 1, 4, 3), (0, 2, 1, 4, 3), (3, 0, 2, 1, 4), (0, 3, 2, 1, 4),
        (4, 0, 3, 2, 1), (0, 4, 3, 2, 1), (1, 0, 4, 3, 2), (2, 1, 0, 4, 3), (3, 2, 1, 0, 4), (4, 3, 2, 1, 0),
        (3, 4, 2, 1, 0), (0, 3, 4, 2, 1), (3, 0, 4, 2, 1), (1, 3, 0, 4, 2), (2, 1, 3, 0, 4), (1, 2, 3, 0, 4),
        (4, 1, 2, 3, 0), (0, 4, 1, 2, 3), (4, 0, 1, 2, 3), (3, 4, 0, 1, 2), (2, 3, 4, 0, 1), (1, 2, 3, 4, 0),
    ]
    """
    # idxs_tbl = np.vstack((idx1, idx2, idx3))
    
    # n = 6
    # idx1 = np.array([1, 0, 2, 3, 4, 5])
    # idx2 = np.array([2, 0, 1, 3, 4, 5])
    # idx3 = np.array([5, 0, 1, 2, 3, 4])
    # idx4 = np.array([3, 0, 1, 2, 4, 5])
    # idx5 = np.array([3, 0, 1, 2, 5, 4])
    # idxs_tbl = np.vstack((idx1, idx2, idx3, idx4, idx5))

    print("n: {}".format(n))
    arr = np.arange(0, n, dtype=np.uint8)
    print("arr: {}".format(arr))

    amount_functions = idxs_tbl.shape[0]
    print("amount_functions: {}".format(amount_functions))

    arr1 = arr[idx1]
    arr2 = arr[idx2]
    # print("arr1: {}".format(arr1))
    # print("arr2: {}".format(arr2))
    # print("arr1.dtype: {}".format(arr1.dtype))
    # print("arr2.dtype: {}".format(arr2.dtype))

    perm_tbl = get_permutation_table(n)
    # print("perm_tbl:\n{}".format(perm_tbl))

    # perm_tbl[idx1].tolist()

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
    d_from_to_num_mapping = {d_t_to_num[t]: list(map(lambda x: d_t_to_num[x], l_t)) for t, l_t in d_from_to_mapping.items()}

    l_t_num_perm_tbl = [d_t_to_num[t] for t in l_t_perm_tbl]
    s_t_num_perm_tbl = set(l_t_num_perm_tbl)

    # do the algorithm for finding the most optimal l_idxs for using the functions!
    def find_best_idxs(d_from_to_mapping, define_random_idxs=False):
        for k in d_from_to_mapping:
            t_start = k
            break

        i_pos = 0
        fac_n = len(d_from_to_mapping)
        fac_n_1 = fac_n-1
        l_idxs = np.array([-1]*fac_n, dtype=np.int8)
        l_t = [None]*fac_n
        l_t[0] = t_start
        s_free_t = set(d_from_to_mapping)
        s_free_t.remove(t_start)
        
        if define_random_idxs:
            # len_s_free_t = len(s_free_t)
            while True:
                func_idxs = np.random.permutation(np.arange(0, amount_functions))
                t_from = l_t[i_pos]
                l_from_to = d_from_to_mapping[t_from]
                is_found = False

                # if len_s_free_t!=0:
                if i_pos<fac_n_1:
                    for i in func_idxs:
                        t_to = l_from_to[i]
                        if t_to in s_free_t:
                            s_free_t.remove(t_to)
                            # len_s_free_t -= 1
                            l_idxs[i_pos] = i
                            i_pos += 1
                            l_t[i_pos] = t_to
                            is_found = True
                            break
                else:
                    is_found = False
                    for i, t_to in enumerate(l_from_to, 0):
                        if t_to==t_start:
                            is_found = True
                            break

                    if is_found:
                        return True, l_idxs, l_t

                if not is_found:
                    s_free_t.add(l_t[i_pos])
                    l_t[i_pos] = None
                    l_idxs[i_pos] = -1
                    i_pos -= 1
                    break

            print("starting at i_pos: {}".format(i_pos))

        print("t_start: {}".format(t_start))
        print("len(d_from_to_mapping): {}".format(len(d_from_to_mapping)))

        i_counter = 0
        i_pos_max = 0
        is_found_optimal_idxs = False
        while i_pos>=0:
            # if i_counter>10000000:
            #     break

            if i_pos_max<i_pos:
                i_pos_max = i_pos
            
            i_counter += 1
            if i_counter%100000==0:
                print(f"i_counter: {i_counter:11}, i_pos: {i_pos:3}, i_pos_max: {i_pos_max:3}")
            
            new_i = l_idxs[i_pos]+1
            if new_i>=amount_functions:
                l_idxs[i_pos] = -1
                s_free_t.add(l_t[i_pos])
                l_t[i_pos] = None
                i_pos -= 1
                continue

            t_from = l_t[i_pos]
            l_from_to = d_from_to_mapping[t_from]
            if i_pos<fac_n_1:
                is_found = False
                for i, t_to in enumerate(l_from_to[new_i:], new_i):
                    if t_to in s_free_t:
                        l_idxs[i_pos] = i
                        s_free_t.remove(t_to)
                        i_pos += 1
                        l_t[i_pos] = t_to
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

    # is_found_optimal_idxs, l_idxs, l_t = find_best_idxs(d_from_to_mapping, define_random_idxs=True)
    is_found_optimal_idxs, l_idxs, l_t = find_best_idxs(d_from_to_num_mapping, define_random_idxs=True)

    if is_found_optimal_idxs:
        print('Found something!')
        print("l_t: {}".format(l_t))
        print("l_idxs: {}".format(l_idxs))
    else:
        print('Did not have found an optimal list of idxs for applying permuation functions!')
