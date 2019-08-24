#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import sys

from copy import deepcopy

sys.path.append("../combinatorics/")
from different_combinations import get_permutation_table
import different_combinations
sys.path.append("../math_numbers/")
from prime_numbers_fun import get_primes

import numpy as np

from PIL import Image
from functools import reduce
from math import factorial
from time import time

from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj
from utils_modulo_sequences import prettyprint_dict

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"



if __name__ == "__main__":
    n = 3

    xs = np.arange(0, n)
    choosen_fs = get_permutation_table(n)[1:]
    perm_parts_tbl = different_combinations.get_all_permutations_parts(choosen_fs.shape[0], 3)

    def mix_choosen_fs():
        choosen_fs[:] = choosen_fs[np.random.permutation(np.arange(0, factorial(n)-1))]
    
    def get_perm_parts_gen():
        for row in perm_parts_tbl:
            yield row

    different_working_tpls = []
    # for nr_iter, row in enumerate(get_perm_parts_gen()):
    #     print("nr_iter: {}, row: {}".format(nr_iter, row))
    for nr_iter in range(0, 1000):
        if nr_iter%1000==0:
            print("nr_iter: {}".format(nr_iter))
            print(" - len(different_working_tpls): {}".format(len(different_working_tpls)))
        while True:
            # get_new_perm = lambda: np.random.permutation(xs)

            # k = 20
            # fs = [get_new_perm() for _ in range(0, k)]

            # # delete duplicates!
            # fs_tpls = [tuple(f) for f in fs]

            # choosen_fs = []
            # choosen_tpls = []

            # for f, t in zip(fs, fs_tpls):
            #     if not t in choosen_tpls:
            #         choosen_fs.append(f)
            #         choosen_tpls.append(t)

            # print("k: {}".format(k))
            # print("len(choosen_fs): {}".format(len(choosen_fs)))
            # print("choosen_fs: {}".format(choosen_fs))

            mix_choosen_fs()
            # print("choosen_fs: {}".format(choosen_fs))

            current_x = xs
            lst_tpls = [tuple(current_x)]
            lst_fs_num = []

            n_max = factorial(n)
            for i in range(1, n_max):
                # print("i: {}".format(i))
                found_f = False
                for j, f in enumerate(choosen_fs, 0):
                    new_x = f[current_x]
                    t = tuple(new_x)
                    if not t in lst_tpls:
                        found_f = True
                        break

                if not found_f:
                    break

                lst_tpls.append(t)
                lst_fs_num.append(j)
                current_x = new_x.copy()

            if i==n_max-1:
                found_f = False
                for j, f in enumerate(choosen_fs, 0):
                    new_x = f[current_x]
                    t = tuple(new_x)
                    if t==lst_tpls[0]:
                        found_f = True
                        break

                if found_f:
                    lst_tpls.append(t)
                    lst_fs_num.append(j)
                    current_x = new_x

            if len(lst_tpls)<n_max+1:
                # print("Not a cyclic of functions found!")
                # print("len(lst_tpls): {}".format(len(lst_tpls)))
                continue

            # print("best: i: {}, len(lst_tpls): {}".format(i, len(lst_tpls)))
            # print("lst_tpls: {}".format(lst_tpls))
            # print("lst_fs_num: {}".format(lst_fs_num))
            # print("current_x: {}".format(current_x))
            unique, counts = np.unique(lst_fs_num, return_counts=True)
            # print("")
            # print("unique: {}".format(unique))
            # print("counts: {}".format(counts))
            # print("unique.shape[0]: {}".format(unique.shape[0]))
            break

        working_tpl = tuple(tuple(choosen_fs[i]) for i in unique)

        if not working_tpl in different_working_tpls:
            different_working_tpls.append(working_tpl)

    different_working_tpls = sorted(different_working_tpls)
    print("different_working_tpls: {}".format(different_working_tpls))

    lengths = [len(t) for t in different_working_tpls]
    u2, c2 = np.unique(lengths, return_counts=True)
    lengths2 = list(zip(u2, c2))
    print("lengths2: {}".format(lengths2))
    print("len(lengths): {}".format(len(lengths)))

    """
    getting sequence for the lowest amount of needed functions
    too get all distinct permutations of n:
    [(1, 1), (2, 1), (3, 2), (4, 3), (5, 3), (6, 4)]
    """
