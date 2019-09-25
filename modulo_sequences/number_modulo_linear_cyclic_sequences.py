#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import sys

from copy import deepcopy
from dotmap import DotMap

from sortedcontainers import SortedSet

sys.path.append("../combinatorics/")
import different_combinations as combinations
sys.path.append("../math_numbers/")
from prime_numbers_fun import get_primes

from time import time
from functools import reduce

from PIL import Image

import numpy as np

from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

from PIL import ImageTk
from tkinter import Tk, Label, BOTH
from tkinter.ttk import Frame, Style


def time_measure(f, args):
    start_time = time()
    ret = f(*args)
    end_time = time()
    diff_time = end_time-start_time
    return ret, diff_time


class ShowImg(Frame, object):
    def __init__(self, img):
        parent = Tk()
        Frame.__init__(self, parent)
        self.pack(fill=BOTH, expand=1)
        label1 = Label(self)
        label1.photo= ImageTk.PhotoImage(img)
        label1.config(image=label1.photo)
        label1.pack(fill=BOTH, expand=1)
        parent.mainloop()


def find_cyclics(d):
    keys = list(d.keys())
    values = list(d.values())

    set_keys = set(keys)
    set_values = set(values)
    assert set_values.intersection(set_keys)==set_values

    tpl_cyclics = []
    tpl_cyclics_length = []
    d = deepcopy(d)
    keys_set = SortedSet(d.keys())
    while len(d) > 0:
        k = keys_set[0]
        if k not in d.values():
            del d[k]
            keys_set.remove(k)
            continue

        lst = []
        do_continue = False
        while k not in lst:
            if k not in d:
                do_continue = True
                for k in lst:
                    del d[k]
                    keys_set.remove(k)
                break
            lst.append(k)
            k = d[k]

        if do_continue:
            continue

        cyclic_length = len(lst)-lst.index(k)
        
        for k in lst:
            del d[k]
            keys_set.remove(k)

        lst1 = lst[-cyclic_length:]
        i = lst1.index(sorted(lst1)[0])
        tpl = tuple(lst1[i:]+lst1[:i])
        tpl_cyclics.append(tpl)
        tpl_cyclics_length.append(cyclic_length)

    return tpl_cyclics, tpl_cyclics_length


def do_2_variables_cycle(n, a_1, a_2, c):
    d = {}
    for x_1 in range(0, n):
        n1 = (c+a_1*x_1)%n
        for x_2 in range(0, n):
            n2 = (n1+a_2*x_2)%n
            d[(x_1, x_2)] = (x_2, n2)

    tpl_cyclics, tpl_cyclics_length = find_cyclics(d) 

    return tpl_cyclics, tpl_cyclics_length


def do_3_variables_cycle(n, a_1, a_2, a_3, c):
    d = {}
    for x_1 in range(0, n):
        n1 = (c+a_1*x_1)%n
        for x_2 in range(0, n):
            n2 = (n1+a_2*x_2)%n
            for x_3 in range(0, n):
                n3 = (n2+a_2*x_3)%n
                d[(x_1, x_2, x_3)] = (x_2, x_3, n3)

    tpl_cyclics, tpl_cyclics_length = find_cyclics(d) 

    return tpl_cyclics, tpl_cyclics_length


if __name__ == "__main__":
    max_lengths = []
    for n in range(2, 11):
        dict_cyclic_length = {}
        for a_1 in range(1, n):
            for a_2 in range(1, n):
                for c in range(1, n):
                    tpl = (a_1, a_2, c)
                    tpl_cyclics, tpl_cyclics_length = do_2_variables_cycle(n, a_1, a_2, c)
                    for tpl_cyclic, tpl_cyclic_length in zip(tpl_cyclics, tpl_cyclics_length):
                        dict_cyclic_length[tpl_cyclic] = tpl_cyclic_length

        # dict_cyclic_length = {}
        # for a_1 in range(1, n):
        #     for a_2 in range(1, n):
        #         for a_3 in range(1, n):
        #             for c in range(1, n):
        #                 tpl = (a_1, a_2, a_3, c)
        #                 tpl_cyclics, tpl_cyclics_length = do_3_variables_cycle(n, a_1, a_2, a_3, c)
        #                 for tpl_cyclic, tpl_cyclic_length in zip(tpl_cyclics, tpl_cyclics_length):
        #                     dict_cyclic_length[tpl_cyclic] = tpl_cyclic_length

        lengths = np.array(list(dict_cyclic_length.values()))
        unique, counts = np.unique(lengths, return_counts=True)
        print("n: {}".format(n))
        print("unique: {}".format(unique))
        print("counts: {}".format(counts))

        max_lengths.append((n, unique[-1]))

    # print("max_lengths: {}".format(max_lengths))
    print("max_lengths:")
    for n, length in max_lengths:
        print("{:2}: {:2}".format(n, length))