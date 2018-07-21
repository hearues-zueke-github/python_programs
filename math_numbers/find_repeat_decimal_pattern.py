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


def find_beginning_index_pattern(str_num, repeat):
    start_idx = str_num.find(repeat)
    if start_idx <= 0:
        return start_idx, repeat

    length = len(repeat)

    # print("start_idx: {}".format(start_idx))
    temp_pattern = str(repeat)
    for i in range(start_idx-1, -1, -1):
        temp2_pattern = temp_pattern[-1]+temp_pattern[:-1]
        if str_num[i:i+length] != temp2_pattern:
            break
        temp_pattern = temp2_pattern

    return i, temp_pattern


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
    mult_factor = []
    # start_idxs = []

    dec_1 = Dec(1)
    for p in ps[3:2000]:
        decimal.getcontext().prec = int(p)+4

        # dec_p = Dec(int(p))
        # num = dec_1 / dec_p

        num = dec_1 / Dec(int(p))
        # print("num: {}".format(num))

        str_num = str(num)[2:]
        # print("p: {}".format(p))
        # print("p: {}, str_num: {}".format(p, str_num))

        # print("type(str_num): {}".format(type(str_num)))
        # print("p: {}, len(str_num): {}".format(p, len(str_num)))

        # print("str_num[::-1]: {}".format(str_num[::-1]))

        # repeat_num, repeat_length = find_repeats_length(str_num[::-1])
        # repeat_num = repeat_num[::-1]
        
        repeat_num, repeat_length = find_repeats_length(str_num)

        # print("repeat_num: {}, repeat_length: {}".format(repeat_num, repeat_length))

        # start_idx, true_pattern = find_beginning_index_pattern(str_num, repeat_num)
        # print("start_idx: {}, true_pattern: {}".format(start_idx, true_pattern))

        if repeat_num == "-":
            l_rep = p-1
        else:
            l_rep = len(repeat_num)

        factor = (p-1)//l_rep
        repeat_num_lens.append((p, l_rep, factor))
        mult_factor.append(factor)
        # start_idxs.append(start_idx)

    mult_factor = np.array(mult_factor)
    unique_mult_factor = sorted(list(set(mult_factor)))
    mult_factor_hist_nums = np.sum(mult_factor==np.array(unique_mult_factor).reshape((-1, 1)), axis=-1)
    mult_factor_hist = {factor: factor_hist for factor, factor_hist in zip(unique_mult_factor, mult_factor_hist_nums)}

    print("repeat_num_lens: {}".format(repeat_num_lens))
    print("mult_factor: {}".format(mult_factor))
    print("mult_factor_hist: {}".format(mult_factor_hist))
    print("unique_mult_factor: {}".format(unique_mult_factor))
    print("mult_factor_hist_nums: {}".format(mult_factor_hist_nums))
    # print("start_idxs: {}".format(start_idxs))
