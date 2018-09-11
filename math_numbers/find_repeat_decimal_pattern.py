#! /usr/bin/python3

# -*- coding: utf-8 -*-

import decimal
import dill
import marshal
import pickle
import os

import multiprocessing as mp
import numpy as np

from dotmap import DotMap
from multiprocessing import Process, Queue

from decimal import Decimal as Dec

def find_repeats_length(str_num):
    length = len(str_num)

    first_char = str_num[0]
    found_length = False
    for j in range(1, 10):
        if first_char != str_num[j]:
            found_length = True
            j += 1
            break

    if not found_length:
        return "-", 0

    for i in range(j, (length+1)//2):
        if str_num[:i] == str_num[i:i*2]:
            return str_num[:i], i

    return "-", 0


if __name__ == "__main__":
    from os.path import expanduser
    home = expanduser("~")
    print("home: {}".format(home))

    path_files = home+"/Documents/"
    file_path = path_files+"primes.pkl"

    with open(file_path, "rb") as fin:
        dm = dill.load(fin)

    print("dm.ps: {}".format(dm.ps))

    ps = dm.ps

    precision = 2500
    decimal.getcontext().prec = precision

    repeat_num_lens = []
    pattern_lengths = []
    mult_factor = []

    dec_1 = Dec(1)
    for p in ps[3:8000]:
        decimal.getcontext().prec = int(p)+4

        num = dec_1 / Dec(int(p))

        str_num = str(num)[2:]
        
        repeat_num, repeat_length = find_repeats_length(str_num)

        if repeat_num == "-":
            l_rep = p-1
        else:
            l_rep = len(repeat_num)

        if l_rep == 32:
            print("l_rep: 32, p: {}".format(p))

        factor = (p-1)//l_rep
        repeat_num_lens.append((p, l_rep, factor))
        pattern_lengths.append(l_rep)
        mult_factor.append(factor)

    pattern_lengths = np.array(pattern_lengths)
    unique_pattern_lengths = np.array(sorted(list(set(pattern_lengths))))
    pattern_lengths_hist_nums = np.sum(pattern_lengths == unique_pattern_lengths.reshape((-1, 1)), axis=-1)
    # use only lengths, which comes more then once!
    print("before: len(unique_pattern_lengths): {}".format(len(unique_pattern_lengths)))
    choose_nums = pattern_lengths_hist_nums > 1

    unique_pattern_lengths = unique_pattern_lengths[choose_nums]
    pattern_lengths_hist_nums = pattern_lengths_hist_nums[choose_nums]
    print("after: len(unique_pattern_lengths): {}".format(len(unique_pattern_lengths)))

    pattern_lengths_hist = [(length, amount) for length, amount in zip(unique_pattern_lengths, pattern_lengths_hist_nums)]

    mult_factor = np.array(mult_factor)
    unique_mult_factor = np.array(sorted(list(set(mult_factor))))
    mult_factor_hist_nums = np.sum(mult_factor == unique_mult_factor.reshape((-1, 1)), axis=-1)
    mult_factor_hist = [(factor, factor_hist) for factor, factor_hist in zip(unique_mult_factor, mult_factor_hist_nums)]

    # print("repeat_num_lens: {}".format(repeat_num_lens))
    print("mult_factor: {}".format(mult_factor))
    print("mult_factor_hist: {}".format(mult_factor_hist))
    # print("unique_mult_factor: {}".format(unique_mult_factor))
    # print("mult_factor_hist_nums: {}".format(mult_factor_hist_nums))

    print("pattern_lengths: {}".format(pattern_lengths))
    print("pattern_lengths_hist: {}".format(pattern_lengths_hist))
    # print("unique_pattern_lengths: {}".format(unique_pattern_lengths))
    # print("pattern_lengths_hist_nums: {}".format(pattern_lengths_hist_nums))
    # print("start_idxs: {}".format(start_idxs))
