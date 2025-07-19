#! /usr/bin/python3

# -*- coding: utf-8 -*-

# Some other needed imports
import os
import sys

import matplotlib.pyplot as plt

import numpy as np

from math import factorial as fac

sys.path.append("../../combinatorics/")
from different_combinations import get_permutation_table

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"

"""
    Problem:
    - Find the smallest amount of different permutation functions m where
      the creation of all possible permuations beginning with the starting sequence
      [1, 2, ..., n] for a value of n
    - e.g. n = 5 with 2 functions
      f_1([x_1, x_2, x_3, x_4, x_5]) = [x_2, x_3, x_1, x_5, x_4]
      f_2([x_1, x_2, x_3, x_4, x_5]) = [x_1, x_3, x_4, x_5, x_2]

      for sequence starting with l_0 = [1, 2, 3, 4, 5]
      apply functions f_1 and f_2 in such a way that n! (for n = 5 is n! = 120) is creating a full
      cycle of permutations

      e.g.
      l_1 = f_1(l_0) = [2, 3, 1, 5, 4]
      l_2 = f_2(l_1) = [2, 1, 5, 4, 3] # = f_2(f_1(l_0)))
      l_3 = f_1(l_2) = [1, 5, 2, 3, 4] # = f_1(f_2(f_1(l_0)))
      etc.
      until
      l_n = f_{i_n} ( f_{i_{n-1}} ( f_{i_{n-2}} ( ... f_2 ( f_1 ( l_0 ) ) ... ) ) ) = l_0
      where i_k elemnt of [1, 2] and k element of [1, 2, ..., n]
    - In short: Find the minimum amount of needed functions m and find a sequence
      (if possible) for n values where the functions can be applied to create
      a full cycle of permuations.
"""


def do_one_example_cycle(n):
    d = {}
    do_one_example_cycle.d = d
    print(f"n: {n}")

    arr_pt = get_permutation_table(n)
    print(f"- arr_pt.shape: {arr_pt.shape}")

    arr_perm_1 = np.array([1, 0] + list(range(2, n)), dtype=np.uint8)
    arr_perm_2 = np.array([0] + list(range(2, n)) + [1], dtype=np.uint8)

    arr_arr_perm = np.vstack((arr_perm_1, arr_perm_2))
    
    max_seq_len = arr_pt.shape[0]
    arr_seq = np.zeros((max_seq_len + 1, n), dtype=np.uint8)
    arr_seq[0] = np.arange(0, n)

    for amount_f_1 in range(1, n - 1):
        arr_seq_small = np.array([0] + [1] * amount_f_1, dtype=np.uint8)

        i = 0
        while i <= max_seq_len:
            i += 1
            arr_seq[i] = arr_seq[i-1][arr_arr_perm[arr_seq_small[(i+1)%arr_seq_small.shape[0]]]]

            arr_idx = np.all(arr_seq[i] == arr_seq[:i], 1)
            if np.any(arr_idx):
                break

        arr_idx_num = np.where(arr_idx)[0]
        print(f"- amount_f_1: {amount_f_1}, arr_idx_num: {arr_idx_num}, i: {i}, max_amount: {i / (amount_f_1 + 1)}")


def calc_arr_pt_func_all(arr_pt, arr_pt_func):
    for i, (arr_prev, arr_next) in enumerate(zip(arr_pt[:-1], arr_pt[1:]), 1):
        arr_pt_func[i] = arr_next[np.argsort(arr_prev)]
    arr_pt_func[0] = arr_pt[0][np.argsort(arr_pt[-1])]


def calc_arr_pt_func(arr_pt, arr_idx, arr_pt_func):
    for idx_next in arr_idx:
        idx_prev = (idx_next - 1) % n_pt
        arr_prev = arr_pt[idx_prev]
        arr_next = arr_pt[idx_next]

        arr_pt_func[idx_next] = arr_next[np.argsort(arr_prev)]


if __name__=='__main__':
    arr_pt = get_permutation_table(n)
    # arr_pt = arr_pt[np.random.permutation(np.arange(0, arr_pt.shape[0]))]
    arr_pt[:] = arr_pt[np.argsort(arr_pt.reshape((-1, )).view('u1'+',u1'*(n-1)))]
    n_pt = arr_pt.shape[0]

    arr_pt_cpy = arr_pt.copy()
    print(f"- arr_pt.shape: {arr_pt.shape}")

    arr_pt_func = np.zeros(arr_pt.shape, dtype=np.uint8)

    calc_arr_pt_func_all(arr_pt=arr_pt, arr_pt_func=arr_pt_func)
    amount_unique_func = np.unique(arr_pt_func.reshape((-1, )).view('u1'+',u1'*(n-1))).shape[0]
    min_amount_unique_func = amount_unique_func
    print(f"min_amount_unique_func: {min_amount_unique_func}")

    arr_pt_func_cpy = arr_pt_func.copy()

    sys.exit()

    n_parts = 5

    l_t_idx_used = []

    for i_iter_find_next in range(0, 200):
        print(f"i_iter_find_next: {i_iter_find_next}")

        l_t_t_idx_amount_unique_func = []
        for _ in range(0, 1000):
            arr_idx_1 = np.random.permutation(np.arange(0, n_pt))[:n_parts]
            arr_idx_1 = np.roll(arr_idx_1, -np.argmin(arr_idx_1))
            arr_idx_2 = np.roll(arr_idx_1, 1)

            arr_pt[arr_idx_1] = arr_pt[arr_idx_2]

            arr_idx_prev_next = np.unique(np.hstack((arr_idx_1, (arr_idx_1 + 1) % n_pt)))
            calc_arr_pt_func(arr_pt=arr_pt, arr_idx=arr_idx_prev_next, arr_pt_func=arr_pt_func)

            amount_unique_func = np.unique(arr_pt_func.reshape((-1, )).view('u1'+',u1'*(n-1))).shape[0]

            arr_pt[arr_idx_2] = arr_pt[arr_idx_1]
            arr_pt_func[arr_idx_prev_next] = arr_pt_func_cpy[arr_idx_prev_next]

            assert np.all(arr_pt == arr_pt_cpy)
            assert np.all(arr_pt_func == arr_pt_func_cpy)

            l_t_t_idx_amount_unique_func.append((tuple(arr_idx_1.tolist()), amount_unique_func))

        d_t_idx_to_amount_unique_func = dict(l_t_t_idx_amount_unique_func)

        d_amount_unique_func_to_l_t_idx = {}

        for key, val in d_t_idx_to_amount_unique_func.items():
            if val not in d_amount_unique_func_to_l_t_idx:
                d_amount_unique_func_to_l_t_idx[val] = [key]
                continue
            
            d_amount_unique_func_to_l_t_idx[val].append(key)

        local_min_amount_unique_func = min(d_amount_unique_func_to_l_t_idx)
        if local_min_amount_unique_func <= min_amount_unique_func:
            print(f"local_min_amount_unique_func: {local_min_amount_unique_func}")
            min_amount_unique_func = local_min_amount_unique_func
            l_t_idx = d_amount_unique_func_to_l_t_idx[local_min_amount_unique_func]
            t_idx = l_t_idx[np.random.randint(0, len(l_t_idx))]

            l_t_idx_used.append(t_idx)

            arr_idx_1 = np.array(t_idx)
            arr_idx_2 = np.roll(arr_idx_1, 1)

            arr_pt[arr_idx_1] = arr_pt[arr_idx_2]

            arr_idx_prev_next = np.unique(np.hstack((arr_idx_1, (arr_idx_1 + 1) % n_pt)))
            calc_arr_pt_func(arr_pt=arr_pt, arr_idx=arr_idx_prev_next, arr_pt_func=arr_pt_func)

            amount_unique_func = np.unique(arr_pt_func.reshape((-1, )).view('u1'+',u1'*(n-1))).shape[0]

            assert amount_unique_func == local_min_amount_unique_func

            arr_pt_cpy = arr_pt.copy()
            arr_pt_func_cpy = arr_pt_func.copy()
        else:
            print(f"No new local_min_amount_unique_func found!")

    print(f"min_amount_unique_func: {min_amount_unique_func}")

    # calc_arr_pt_func(arr_pt=arr_pt, arr_t_idx=arr_arr_t_idx[i_iter], arr_pt_func=arr_pt_func)
    # amount_unique_func = np.unique(arr_pt_func.reshape((-1, )).view('u1'+',u1'*(n-1))).shape[0]
    # arr_amount_unique_func[i_iter] = amount_unique_func

    # n_pt = arr_pt.shape[0]
    # for i in range(0, n_pt):


    # l_unique_funcs = []
    # for i in range(0, 1000000):
    #     if i % 100000 == 0:
    #         print(f"i: {i}")

    #     arr_pt_rnd = arr_pt[np.random.permutation(np.arange(0, arr_pt.shape[0]))]

        

    #     unique_funcs = np.unique(arr_pt_func.reshape((-1, )).view('u1'+',u1'*(n-1))).shape[0]
    #     l_unique_funcs.append(unique_funcs)

    # print(f"np.min(l_unique_funcs): {np.min(l_unique_funcs)}")

    # a_0 = [0, 1, 2, 3, 4]
    # f_1([0, 1, 2, 3, 4]) = [2, 1, 4, 3, 0]

    # [4, 1, 0, 3, 2]

    # d_stat = {}
    # for row1 in arr_pt:
    #     a1 = row1.copy()
        
    #     for row2 in arr_pt:
    #         a2 = row2.copy()

    #         t = tuple(a1.tolist())+tuple(a2.tolist())
    #         l = [t]
    #         while True:
    #             a3 = a1[a2].copy()
    #             a1 = a2
    #             a2 = a3
    #             t = tuple(a1.tolist())+tuple(a2.tolist())
    #             # t = tuple(a.tolist())
    #             if t in l:
    #                 l.append(t)
    #                 break
    #             l.append(t)

    #         length = len(l)
    #         if not length in d_stat:
    #             d_stat[length] = []
    #         d_stat[length].append(l)

    # print("d_stat.keys(): {}".format(d_stat.keys()))

    # d_stat_true_cycle_length = {}
    # for v in d_stat.values():
    #     for l in v:
    #         length = len(l)-l.index(l[-1])-1
    #         if not length in d_stat_true_cycle_length:
    #             d_stat_true_cycle_length[length] = []
    #         d_stat_true_cycle_length[length].append(l[-1])

    # l_cycle_lengths = sorted([(k, len(v)) for k, v in d_stat_true_cycle_length.items()])
    # print("l_cycle_lengths: {}".format(l_cycle_lengths))
    # print("len(l_cycle_lengths): {}".format(len(l_cycle_lengths)))
