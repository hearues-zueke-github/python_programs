#! /usr/bin/python3

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

import numpy as np
import pandas as pd

from copy import deepcopy, copy
from dotmap import DotMap
from functools import reduce
from memory_tempfile import MemoryTempfile
from shutil import copyfile

sys.path.append('..')
from utils import mkdirs
from utils_multiprocessing_manager import MultiprocessingManager
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj
sys.path.append('../combinatorics')
import different_combinations

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"
HOME_DIR = os.path.expanduser("~")+"/"
TEMP_DIR = MemoryTempfile().gettempdir()+"/"

OBJS_DIR_PATH = PATH_ROOT_DIR+'objs/'
mkdirs(OBJS_DIR_PATH)


def create_d_n_mod_x_len():
    return {}


def calc_new_cycles(n, modulo):
    obj_d_n_mod_x_len_file_path = OBJS_DIR_PATH+f'd_n_{n}_mod_{modulo}_x_len.pkl.gz'
    d_n_mod_x_len = get_pkl_gz_obj(create_d_n_mod_x_len, obj_d_n_mod_x_len_file_path)

    if n not in d_n_mod_x_len:
        d_n_mod_x_len[n] = {}
    d_mod_x_len = d_n_mod_x_len[n]

    if modulo not in d_mod_x_len:
        d_mod_x_len[modulo] = {}
    d_x_len = d_mod_x_len[modulo]

    max_cycle_length = 0
    s_start_t = set()
    for i_loop in range(0, 10000):
        x = np.random.randint(0, modulo, (n, ))
        A = np.random.randint(0, modulo, (n, n))

        d = {}

        idx_nr = 0
        t_prev = tuple(x.tolist())
        while True:
            x_next = np.dot(A, x) % modulo
            t_next = tuple(x_next.tolist())
            if t_prev in d:
                break
            d[t_prev] = t_next
            t_prev = t_next
            x = x_next

        l_seq = [t_prev]
        t = d[t_prev]
        while t != t_prev:
            l_seq.append(t)
            t = d[t]
        assert l_seq[0] == d[l_seq[-1]]

        min_idx = sorted(zip(l_seq, range(0, len(l_seq))))[0][1]
        if min_idx > 0:
            l_seq = l_seq[min_idx:]+l_seq[:min_idx]
        assert l_seq[0] == d[l_seq[-1]]

        for t1, t2 in zip(l_seq, l_seq[1:]+l_seq[:1]):
            assert t2 == d[t1]

        # print("- len(l_seq): {}".format(len(l_seq)))

        cycle_length = len(l_seq)
        t0 = l_seq[0]
        len_l_seq = len(l_seq)
        A_t = tuple(map(tuple, A.tolist()))

        if t0 not in d_x_len:
            d_x_len[t0] = (len_l_seq, A_t)
        else:
            len_l_seq_old, A_t_old = d_x_len[t0]

            if len_l_seq_old < len_l_seq:
                d_x_len[t0] = (len_l_seq, A_t)
            elif len_l_seq_old == len_l_seq and A_t_old > A_t:
                d_x_len[t0] = (len_l_seq, A_t)
        
        if max_cycle_length < cycle_length:
            max_cycle_length = cycle_length
            s_start_t = {t0}
        elif max_cycle_length == cycle_length:
            if t0 not in s_start_t:
                s_start_t.add(t0)

    print("max_cycle_length: {}".format(max_cycle_length))
    print("s_start_t: {}".format(s_start_t))

    u, c = np.unique([v[0] for k, v in d_x_len.items()], return_counts=True)
    print("u[-1]: {}, c[-1]: {}".format(u[-1], c[-1]))

    save_pkl_gz_obj(d_n_mod_x_len, obj_d_n_mod_x_len_file_path)


if __name__ == '__main__':
    # for file, d_n_mod_x_len in l_values:
    #     for n, d_mod_x_len in d_n_mod_x_len.items():
    #         for modulo, d_x_len in d_mod_x_len.items():
    #             print("n: {}, modulo: {}".format(n, modulo))
    #             max_cycle_length = max([v[0] for k, v in d_x_len.items()])

    # print("l_values: {}".format(l_values))

    # sys.exit()


    # n = int(sys.argv[1])
    # if n > 10:
    #     n = 10
    # if n < 1:
    #     n = 1

    # modulo = int(sys.argv[2])
    # if modulo > 10:
    #     modulo = 10
    # if modulo < 1:
    #     modulo = 1

    # n = 4
    # modulo = 3
    
    mult_proc_manag = MultiprocessingManager(cpu_count=4)
    mult_proc_manag.define_new_func(name='calc_new_cycles', func=calc_new_cycles)
    # l_func_args = [
    #     (2, 2), (2, 3), (2, 4),
    #     (3, 2), (3, 3), (3, 4),
    #     (4, 2), (4, 3), (4, 4),
    # ]
    l_func_args = [
        (n, modulo) for n in range(2, 8) for modulo in range(2, 6)
    ]
    l_func_args = [l_func_args[i] for i in np.random.permutation(np.arange(0, len(l_func_args)))]
    l_func_name = ['calc_new_cycles'] * len(l_func_args)
    # mult_proc_manag.do_new_jobs(l_func_name=['calc_new_cycles']*, l_func_args)
    mult_proc_manag.do_new_jobs(l_func_name=l_func_name, l_func_args=l_func_args)
    # calc_new_cycles(n=n, modulo=modulo)

    del mult_proc_manag

    # if n == 1 and modulo == 1:
    l_files = [os.path.join(root, file) for root, dirs, files in os.walk(OBJS_DIR_PATH) for file in files]
    # sys.exit()
    
    l_values_d = [
        (file, get_pkl_gz_obj(create_d_n_mod_x_len, file))
        for file in l_files
    ]

    l_values = [
        (n, modulo, max([v[0]
        for k, v in d_x_len.items()]))
        for file, d_n_mod_x_len in l_values_d
        for n, d_mod_x_len in d_n_mod_x_len.items()
        for modulo, d_x_len in d_mod_x_len.items()
    ]

    l_values_sort = sorted(l_values)
    print("l_values_sort: {}".format(l_values_sort))
    sys.exit()

    # if n == 1 and modulo == 1:
    #     l_values = [
    #         (n, modulo, max([v[0] for k, v in d_x_len.items()]))
    #         for n, d_mod_x_len in d_n_mod_x_len.items()
    #         for modulo, d_x_len in d_mod_x_len.items()
    #         # for x, (cycle_length, A_t) in d_x_len.items()
    #     ]
    #     print("l_values: {}".format(l_values))

    # y = np.dot(A, x) % modulo

    # print("x:\n{}".format(x))
    # print("A:\n{}".format(A))
    # print("y:\n{}".format(y))
    # # sys.exit()

    # # a = np.random.randint(0, 2, (n, ))
    # # b = np.random.randint(0, 2, (n, ))

    # # C = np.add.outer(a, b) % modulo

    # # an1 = np.dot(C, a) % modulo
    # # bn1 = np.dot(C, b) % modulo

    # # an2 = np.dot(a, C) % modulo
    # # bn2 = np.dot(b, C) % modulo

    # print("a:\n{}".format(a))
    # print("b:\n{}".format(b))
    # print("C:\n{}".format(C))
    # print("an1:\n{}".format(an1))
    # print("bn1:\n{}".format(bn1))
    # print("an2:\n{}".format(an2))
    # print("bn2:\n{}".format(bn2))
