#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.8.6 -i

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import dill
import gzip
import os
import pdb
import re
import sys
import traceback
import itertools

import multiprocessing as mp
import numpy as np
import pandas as pd

from typing import List, Tuple, Set, Any

from copy import deepcopy, copy
from dotmap import DotMap
from functools import reduce
from memory_tempfile import MemoryTempfile
from pprint import pprint
from shutil import copyfile

from itertools import chain

sys.path.append('..')
from utils import mkdirs
from utils_multiprocessing_manager import MultiprocessingManager

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"
HOME_DIR = os.path.expanduser("~")+"/"
TEMP_DIR = MemoryTempfile().gettempdir()+"/"

OBJS_DIR_PATH = PATH_ROOT_DIR+'objs/'
mkdirs(OBJS_DIR_PATH)

if __name__ == '__main__':
    # l = np.random.randint(1, 100, (200, ))
    l = [1, 1, 1, 1, 1, 2, 3]
    s = set(l)
    partitions = 3

    u, c = np.unique(l, return_counts=True)
    l_l_num_amount = [np.diff(np.hstack(((0, ), np.sort(np.random.randint(0, amount+1, (partitions-1, ))), (amount, )))) for v, amount in zip(u, c)]

    l_d_parts = [{v: 0 for v in s} for i in range(0, partitions)]

    u, c = np.unique(l, return_counts=True)
    for v, l_num_amount in zip(s, l_l_num_amount):
        for i, amount in enumerate(l_num_amount, 0):
            l_d_parts[i][v] = amount

    # d0 = l_d_parts[0]
    # for v, amount in zip(u, c):
    #     d0[v] += amount

    # print('l_d_parts:')
    # pprint(l_d_parts)
    print("l_d_parts: {}".format(l_d_parts))

    # TODO: add later more stuff here! and fix finding all the possible partitions!

    sys.exit()

    def calc_diff_sum(l_d_parts):
        l_sum = [sum([v*amount for v, amount in d.items()]) for d in l_d_parts]
        # print('l_sum:')
        # pprint(l_sum)

        diff_sum = 0
        for i1, v1 in enumerate(l_sum[:-1], 0):
            for i2, v2 in enumerate(l_sum[i1+1:], i1+1):
                diff_sum += abs(v1 - v2)

        return diff_sum

    diff_sum = calc_diff_sum(l_d_parts)
    diff_sum_prev = diff_sum
    # print("diff_sum: {}".format(diff_sum))

    l_prev_tpl_parts = [
        # [
        tuple(sorted([tuple(sorted(tuple(d.items()))) for d in l_d_parts])),
        # ],
    ]
    l_l_prev_tpl_parts = [(deepcopy(l_prev_tpl_parts), diff_sum)]

    print("Start:")
    print("- l_d_parts: {}".format(l_d_parts))
    print("- diff_sum: {}".format(diff_sum))

    l_parts_pos_move = sorted(
        set([(i1, i2) for i1 in range(0, partitions) for i2 in range(0, partitions)]) -
        set([(i, i) for i in range(0, partitions)])
    )

    # d_moving_num_diff_sum = {}
    for i_iter in range(0, 400):
        print("i_iter: {}".format(i_iter))

        l_diff_sum_moving_num_pos = []
        for t_tpl in l_prev_tpl_parts:
            if len(l_diff_sum_moving_num_pos) > 100:
                break

            l_d_parts = [dict(t) for t in t_tpl]
            l_d_parts_orig = deepcopy(l_d_parts)
            # t = tuple(sorted([tuple(sorted(tuple(d.items()))) for d in l_d_parts_orig]))
            for num in s:
                for i1, i2 in l_parts_pos_move:
                    if l_d_parts[i1][num] == 0:
                        continue

                    l_d_parts[i1][num] -= 1
                    l_d_parts[i2][num] += 1

                    # d_moving_num_diff_sum[(num, i1, i2)] = calc_diff_sum(l_d_parts)
                    
                    diff_sum = calc_diff_sum(l_d_parts)

                    if diff_sum < diff_sum_prev:
                        l_diff_sum_moving_num_pos.append((diff_sum, num, i1, i2, t_tpl))

                    l_d_parts[i1][num] += 1
                    l_d_parts[i2][num] -= 1

            assert l_d_parts_orig == l_d_parts

            # print('d_moving_num_diff_sum:')
            # pprint(d_moving_num_diff_sum.items())

        print("len(l_diff_sum_moving_num_pos): {}".format(len(l_diff_sum_moving_num_pos)))
        if len(l_diff_sum_moving_num_pos) == 0:
            print('- No more possible optimization found!')
            break

        l_diff_sum_moving_num_pos_sorted = sorted(l_diff_sum_moving_num_pos)
        diff_sum_min = l_diff_sum_moving_num_pos_sorted[0][0]
        l_diff_sum_moving_num_pos_filtered = [t for t in l_diff_sum_moving_num_pos_sorted if t[0] == diff_sum_min]

        # TODO: do all possible minimal diff_sum combinations! add them to a tuple and then combine it to a set
        # next step: 
        
        s_prev_tpl_parts_next = set()
        for diff_sum_calc_prev, num, i1, i2, t_tpl in l_diff_sum_moving_num_pos_filtered:
            l_d_parts = [dict(t) for t in t_tpl]
            assert l_d_parts[i1][num] > 0
            l_d_parts[i1][num] -= 1
            l_d_parts[i2][num] += 1
            diff_sum = calc_diff_sum(l_d_parts)
            assert diff_sum == diff_sum_calc_prev

            # print("- l_d_parts: {}".format(l_d_parts))
            # print("- diff_sum: {}".format(diff_sum))

            # l_prev_l_d_parts.append((deepcopy(l_d_parts), diff_sum))
            t = tuple(sorted([tuple(sorted(tuple(d.items()))) for d in l_d_parts]))
            
            if t not in s_prev_tpl_parts_next:
                s_prev_tpl_parts_next.add(t)

            # l_d_parts_orig = deepcopy(l_d_parts)
            # diff_sum_prev = diff_sum
        diff_sum_prev = diff_sum_min
        l_prev_tpl_parts = list(s_prev_tpl_parts_next)
        print("- len(l_prev_tpl_parts): {}".format(len(l_prev_tpl_parts)))

        l_l_prev_tpl_parts.append((l_prev_tpl_parts, diff_sum_prev))

        print("- len(l_prev_tpl_parts): {}, diff_sum_prev: {}".format(len(l_prev_tpl_parts), diff_sum_prev))


    # print('l_diff_sum_moving_num_pos:')
    # pprint(l_diff_sum_moving_num_pos)

    l_diff_sum = [diff_sum for _, diff_sum in l_l_prev_tpl_parts]
    # l_diff_sum = [diff_sum for _, diff_sum in l_prev_l_d_parts]
    print("l_diff_sum: {}".format(l_diff_sum))

    l = l_l_prev_tpl_parts[-1][0]
    l_chain = [[list(itertools.chain.from_iterable([[k]*v for k, v in dict(t2).items()])) for t2 in t] for t in l]
    
    print("l_chain: {}".format(l_chain))

    sys.exit()

    def split_number_in_parts(n: int, parts: int) -> List[int]:
        assert parts >= 1

        l: List[int] = []
        b: int = n - parts + 1
        for _ in range(0, parts-1):
            a: int = np.random.randint(1, b+1)
            l.append(a)
            b = b - a + 1
        l.append(b)

        # print("l: {}".format(l))
        assert sum(l) == n

        return l


    def split_list_in_parts(l: List[int], parts: int) -> List[int]:
        assert parts >= 1

        n = len(l)
        l_split_num = split_number_in_parts(n, parts)

        l_parts = []

        arr_cumsum = np.cumsum([0] + l_split_num)
        assert arr_cumsum[-1] == n
        # print("arr_cumsum: {}".format(arr_cumsum))

        for i1, i2 in zip(arr_cumsum[:-1], arr_cumsum[1:]):
            l_parts.append(l[i1:i2])

        return l_parts


    # sys.exit()

    # l = np.array([1, 2, 3, 3, 3, 4, 4, 5, 6, 6])
    # l = [1, 2, 3, 3, 3, 4, 4, 5, 6, 6]

    def calc_random_partion_sum_len_diff_sum(l: List[int], partitions: int) -> Tuple[Any]:
        arr_idxs_nr = np.arange(0, len(l))
        arr_idxs_nr_perm = np.random.permutation(arr_idxs_nr)

        arr = np.array(l)[arr_idxs_nr_perm]

        l_parts_arr = split_list_in_parts(arr, partitions)

        # l_parts_arr = [np.sort(l[arr_idxs_nr_perm[i::partitions]]) for i in range(0, partitions)]
        # print("l_parts: {}".format(l_parts))
        l_tpl_parts = list(map(lambda x: tuple(sorted(x)), l_parts_arr))

        l_tpl_parts_sorted = [t for (_, t) in sorted([(len(t), t) for t in l_tpl_parts])]

        # l_parts = sorted(list(map(list, l_parts_arr)))

        l_sum = [sum(a) for a in l_tpl_parts_sorted]
        l_len = [len(a) for a in l_tpl_parts_sorted]

        # print("l_sum: {}".format(l_sum))
        # print("l_len: {}".format(l_len))

        def calc_absolute_diff_sum(l_sum: List[List[int]]) -> int:
            s = 0
            
            for i1, v1 in enumerate(l_sum[:-1], 0):
                for i2, v2 in enumerate(l_sum[i1+1:], i1+1):
                    s += abs(v1 - v2)

            return s

        absolute_diff_sum = calc_absolute_diff_sum(l_sum)

        # sorte the list of lists by the length first!

        return tuple(l_tpl_parts_sorted), tuple(l_sum), tuple(l_len), absolute_diff_sum

    l: np.ndarray = np.arange(1, 55)**2
    partitions: int = 7
    max_iters: int = 20000

    def get_random_s_tpl_splits(l: List[int], max_iters: int, partitions: int) -> Set[Tuple[Any]]:
        np.random.seed()

        s_tpl = set()
        for i in range(0, max_iters):
            # print("i: {}".format(i))
            tpl = calc_random_partion_sum_len_diff_sum(l=l, partitions=partitions)
            t_tpl_parts, l_sum, l_len, absolute_diff_sum = tpl
            tpl2 = (absolute_diff_sum, l_sum, l_len, t_tpl_parts)
            
            if tpl2 in s_tpl:
                # print("- Ignore this one!")
                continue
            
            s_tpl.add(tpl2)

        # return s_tpl
        l_one = [sorted(s_tpl)[0]]
        min_sum = l_one[0][0]
        print("min_sum: {}".format(min_sum))

        s_one = set(l_one)

        return s_one

    # def f(x):
    #     return x**2

    mult_proc_mng = MultiprocessingManager(cpu_count=mp.cpu_count())

    # # only for testing the responsivness!
    # mult_proc_mng.test_worker_threads_response()

    print('Define new Function!')
    mult_proc_mng.define_new_func('func_get_random_s_tpl_splits', get_random_s_tpl_splits)
    print('Do the jobs!!')
    l_arguments = [(l, max_iters, partitions)]*mult_proc_mng.worker_amount*200
    l_ret = mult_proc_mng.do_new_jobs(
        ['func_get_random_s_tpl_splits']*len(l_arguments),
        l_arguments,
    )
    print("len(l_ret): {}".format(len(l_ret)))
    # print("l_ret: {}".format(l_ret))

    # # testing the responsivness again!
    # mult_proc_mng.test_worker_threads_response()
    del mult_proc_mng

    # s_tpl = get_random_s_tpl_splits(max_iters, partitions)
    
    s_tpl = set(chain(*l_ret))

    l_tpl = sorted(s_tpl, reverse=True)
    
    print('l_tpl[:10]')
    pprint(l_tpl[:10])

    print('l_tpl[-10:]')
    pprint(l_tpl[-10:])
    
    print("len(l_tpl): {}".format(len(l_tpl)))
