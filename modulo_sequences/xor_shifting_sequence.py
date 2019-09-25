#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import sys

from copy import deepcopy

from time import time
from functools import reduce
from collections import defaultdict

from PIL import Image

import numpy as np

from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

from PIL import ImageTk
from tkinter import Tk, Label, BOTH
from tkinter.ttk import Frame, Style


def get_found_cycles(d):
    s_k = set(d.keys())
    s_v = set(d.values())
    assert s_v.intersection(s_k)==s_v

    l_cycles = []
    d = deepcopy(d)
    while len(d)>0:
        for k in d:
            break
        l = []
        while k not in l and k in d:
            l.append(k)
            k = d[k]
        l_seq = []
        if k in l:
            l_seq = l[l.index(k):]
        for k in l:
            del d[k]

        if len(l_seq)>0:
            # print("l_seq: {}".format(l_seq))
            l_cycles.append(tuple(l_seq))

    l_cycles = sorted(l_cycles, key=lambda x: (len(x), x))

    return l_cycles


if __name__ == "__main__":
    dot_folder = PATH_ROOT_DIR+"dot_files/"
    if not os.path.exists(dot_folder):
        os.makedirs(dot_folder)

    l_sequence_nums = []
    l_sequence_nums_only = []
    n = 6
    for base in range(n, n+1):
        def calc_value(x, base):
            s = 0
            b = 1
            for v in reversed(x):
                s += b*v
                b *= base
            return s

        def do_one_prev_value(base, length):
            def do_next_step(base, x, mask):
                x_next = tuple((i+j)%base for i, j in zip(x, mask))
                x_next_s = x_next[1:]+x_next[:1]
                x_next_2 = tuple((i+j)%base for i, j in zip(x, x_next_s))
                return x_next_2

            # length = 3
            # base = 4

            # mask = (0, 0, 0, 0, 1)
            mask = [0]*length
            mask[-1] = 1
            mask = tuple(mask)
            # mask = (0, 0, 1)
            # mask = (1, 2)
            d = {}
            for i1 in range(0, base):
                for i2 in range(0, base):
                    for i3 in range(0, base):
                        for i4 in range(0, base):
                        #     for i5 in range(0, base):
                        #         x = (i1, i2, i3, i4, i5)
                            x = (i1, i2, i3, i4)
                            x_n = do_next_step(base, x, mask)
                            d[x] = x_n
            return d, mask

        xor_add = lambda x, y, base: tuple((i+j)%base for i, j in zip(x, y))
        xor_sub = lambda x, y, base: tuple((i-j)%base for i, j in zip(x, y))
        shift_list_left = lambda x: x[1:]+x[:1]

        def do_two_prev_value(base, length):
            def do_next_step(base, x, mask):
                x1, x2 = x
                x_next = xor_add(x1, mask, base)
                x_next_s = shift_list_left(x_next)
                x_next_2 = xor_add(x2, x_next_s, base)
                x_next_s2 = shift_list_left(x_next_2)
                
                x_12 = xor_add(x1, x2, base)
                x_next_3 = xor_sub(x_12, x_next_s2, base)
                # x_next = tuple((i+j)%base for i, j in zip(x1, mask))
                # x_next_s = x_next[1:]+x_next[:1]
                # x_next_2 = tuple((i+j)%base for i, j in zip(x2, x_next_s))
                
                # return (x2, x_next_3)
                return (x2, x_next_2)

        def do_three_prev_value(base, length):
            def do_next_step(base, x, mask):
                x1, x2, x3 = x
                x_next = xor_add(x1, mask, base)
                x_next_s = shift_list_left(x_next)
                x_next_2 = xor_add(x2, x_next_s, base)
                x_next_s2 = shift_list_left(x_next_2)
                x_next_3 = xor_add(x3, x_next_s2, base)
                # x_next_s3 = shift_list_left(x_next_3)
                
                # x_12 = xor_add(x1, x2, base)
                # x_next_3 = xor_sub(x_12, x_next_s2, base)
                return (x2, x3, x_next_3)

            mask = [0]*length
            mask[-1] = 1
            mask = tuple(mask)

            d = {}
            for i1 in range(0, base):
                for i2 in range(0, base):
                    for i3 in range(0, base):
                        for i4 in range(0, base):
                            for i5 in range(0, base):
                                for i6 in range(0, base):
                                    x = ((i1, i2), (i3, i4), (i5, i6))
                                    x_n = do_next_step(base, x, mask)
                                    d[x] = x_n
                                    # x = ((i1, i2, i3), (i4, i5, i6))
                                    # x_n = do_next_step(base, x, mask)
                                    # d[x] = x_n
            return d, mask

        # base = 10
        length = 2
        # d, mask = do_two_prev_value(base=base, length=length)
        # d2 = {calc_value(k[0]+k[1], base): calc_value(v[0]+v[1], base) for k, v in d.items()}
        
        d, mask = do_three_prev_value(base=base, length=length)
        d2 = {calc_value(k[0]+k[1]+k[2], base): calc_value(v[0]+v[1]+v[2], base) for k, v in d.items()}
        
        # d, mask = do_one_prev_value(base=base, length=length)
        # d2 = {calc_value(k, base): calc_value(v, base) for k, v in d.items()}
        
        l_cycles = get_found_cycles(d2)
        l_cycles = sorted(l_cycles, key=lambda x: (len(x), x))
        for i, l in enumerate(l_cycles, 0):
            print("i: {}, l: {}".format(i, l))

        cycles_lens = list(map(len, l_cycles))
        uniques, counts = np.unique(cycles_lens, return_counts=True)
        idxs_sort = np.argsort(uniques)
        uniques = uniques[idxs_sort]
        counts = counts[idxs_sort]
        l_uniq_counts = np.array([(u, c) for u, c in zip(uniques, counts)]).T
        print("l_uniq_counts:\n{}".format(l_uniq_counts))
        l_sequence_nums.append((base, uniques[-1]))
        l_sequence_nums_only.append(uniques[-1])

    print()
    print("l_sequence_nums: {}".format(l_sequence_nums))
    print("l_sequence_nums_only: {}".format(l_sequence_nums_only))

    # for i, (k, v) in enumerate(d.items(), 0):
    #     print("i: {}, k: {}, v: {}".format(i, k, v))

    # print("d: {}".format(d))
    # print("d2: {}".format(d2))
    def create_dot_graph(d, file_path):
        s = "digraph {\n"
        for i in d2:
            s += '    a_{}[label="{}"];\n'.format(i, i)
        s += '\n'
        for k, v in d2.items():
            s += '    a_{} -> a_{};\n'.format(k, v)
        s += "}\n"

        with open(file_path, "w") as f:
            f.write(s)

    # file_path = dot_folder+"simple_graph_two_vals_xor_shifting_b_{}_l_{}_m_{}.dot".format(base, length, ",".join(map(str, mask)))
    file_path = dot_folder+"simple_graph_xor_shifting_b_{}_l_{}_m_{}.dot".format(base, length, ",".join(map(str, mask)))
    create_dot_graph(d2, file_path)

    sys.exit(-1)
    
    base = 3
    lst_max_seq_len = []
    for length in range(1, 21):
    # length = 4


        max_seq_len = 0
        # mask = [0]*length
        # mask[-1] = 1
        # for _ in range(0, 10000):
        for try_num in range(0, 10):
            mask = tuple(np.random.randint(0, base, (length, )).tolist())
            # mask_2 = tuple(np.random.randint(0, base, (length, )).tolist())
            
            # mask = [0]*length
            # mask[-1] = 1
            # mask = tuple(mask)
            
            # mask = (0, 0, 0, 1)
            sequence = []
            # seq_int = []

            x = (0, )*length
            # print("x: {}".format(x))

            print("try_num: {}".format(try_num))
            for idx in range(0, base**length):
                if idx%10000==0:
                    print("idx: {}".format(idx))
                # x_next = tuple((i+j)%base for i, j in zip(x, mask))
                # x_next = x_next[1:]+x_next[:1]
                x_next = do_next_step(base, x, mask)

                # x_next_2 = tuple((i+j)%base for i, j in zip(x_next, mask_2))
                # x_next_2 = x_next_2[1:]+x_next_2[:1]
                # x_next_2 = x_next_2[-1:]+x_next_2[:-1]
                # print("x_next: {}".format(x_next))
                # x = x_next_2
                x = x_next
                if x in sequence:
                    sequence.append(x)
                    sequence_part = sequence[sequence.index(x):-1]
                    break
                sequence.append(x)

            def calc_value(t, b):
                x = 1
                s = 0
                for i in reversed(t):
                    s += x*i
                    x *= b
                return s

            seq_int = [calc_value(t, base) for t in sequence_part]

            # print("sequence: {}".format(sequence))
            # print("sequence_part: {}".format(sequence_part))
            # print("seq_int: {}".format(seq_int))

            s = set(sequence)
            # print("s: {}".format(s))

            # print("base: {}, length: {}".format(base, length))
            # print("base**length: {}".format(base**length))
            # print("mask: {}".format(mask))
            # print("len(s): {}".format(len(s)))

            len_s = len(s)
            if max_seq_len<len_s:
                max_seq_len = len_s
                print("mask: {}".format(mask))
                # print("mask_2: {}".format(mask_2))
                print("max_seq_len: {}".format(max_seq_len))
        print("base: {}, length: {}".format(base, length))
        print("last max_seq_len: {}".format(max_seq_len))

        lst_max_seq_len.append((length, max_seq_len))

    print("lst_max_seq_len: {}".format(lst_max_seq_len))
