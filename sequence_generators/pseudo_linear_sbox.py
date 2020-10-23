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

# for plotting
import matplotlib.pyplot as plt

# for image
from PIL import Image

# Needed for excel tables
import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.utils.cell import column_index_from_string
from openpyxl.styles.borders import Border, Side, BORDER_THIN, BORDER_MEDIUM
from openpyxl.styles import Alignment, borders, Font, PatternFill

import matplotlib.pyplot as plt

# Needed for DataFrame (databases etc.)
import numpy as np
import pandas as pd
import pandas.io.sql as sql

from copy import deepcopy
from functools import reduce

from dotmap import DotMap

import utils_sequence

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"


if __name__=='__main__':
    n1 = 2**16
    m1 = 5
    n2 = 100
    m2 = 23
    rounds1 = 4
    rounds2 = 1
    
    l1 = np.arange(0, n1*m1)
    l2 = np.arange(0, n2*m2)

    print("l1: {}".format(l1))
    print("l2: {}".format(l2))
    print("rounds1: {}".format(rounds1))
    print("rounds2: {}".format(rounds2))

    def calc_equal_sbox_positions(sboxes):
        count_equal = 0
        rows = sboxes.shape[0]
        l = []
        for i in range(0, rows-1):
            for j in range(i+1, rows):
                two_rows_count_equal = np.sum(sboxes[i]==sboxes[j])
                l.append((i, j, two_rows_count_equal))
                count_equal += two_rows_count_equal
        l.append(count_equal)
        return l
        # return count_equal

    l1_orig = l1.copy()
    # ls = [tuple(l1.copy().tolist())+tuple(l2.copy().tolist())]
    ls = [tuple(l1.copy().tolist())]
    sboxes11 = utils_sequence.create_sboxes(l1%n1, n1)
    ls_sboxes = [sboxes11]
    ls_equal_nums = [calc_equal_sbox_positions(sboxes11)]
    # ls = [(tuple(l1.copy().tolist()), tuple(l2.copy().tolist()))]
    for i in range(0, 4):
        if i%1==0:
            print("i: {}".format(i))
        l1 = utils_sequence.mix_l1_with_l2_method_2(l1, l2, rounds=rounds1)
        l2 = utils_sequence.mix_l1_with_l2_method_2(l2, l1, rounds=rounds2)
        l2 = np.roll(l2, 1)
        
        sboxes11 = utils_sequence.create_sboxes(l1%n1, n1)
        sboxes12 = utils_sequence.create_sboxes(l1%m1, m1)
        l1_mat = l1.reshape((m1, n1))
        for j in range(0, m1):
            l1_mat[j] = l1_mat[j][sboxes11[j]]
        for k in range(0, n1):
            l1_mat[:, k] = l1_mat[:, k][sboxes12[k]]
        l1 = l1_mat.flatten()

        sboxes21 = utils_sequence.create_sboxes(l2%n2, n2)
        sboxes22 = utils_sequence.create_sboxes(l2%m2, m2)
        l2_mat = l2.reshape((m2, n2))
        for j in range(0, m2):
            l2_mat[j] = l2_mat[j][sboxes21[j]]
        for k in range(0, n2):
            l2_mat[:, k] = l2_mat[:, k][sboxes22[k]]
        l2 = l2_mat.flatten()

        t1 = tuple(l1.tolist())
        # t2 = tuple(l2.tolist())
        # t = (t1, t2)
        t = t1
        # t = t1+t2
        if t in ls:
            idx = ls.index(t)
            print("idx: {}".format(idx))
            ls.append(t)
            ls_sboxes.append(sboxes11)
            ls_equal_nums.append(calc_equal_sbox_positions(sboxes11))
            break
        ls.append(t)
        ls_sboxes.append(sboxes11)
        ls_equal_nums.append(calc_equal_sbox_positions(sboxes11))

    arr = np.array(ls)
    arr_sboxes = np.array(ls_sboxes)
    print("arr:\n{}".format(arr))
    print("arr.shape: {}".format(arr.shape))

    # create the one master sbox!
    assert n1==2**16
    master_sbox_prepare = np.roll(np.arange(0, 2**16, dtype=np.uint16), 1)
    # master_sbox_prepare = np.zeros((n1, ), dtype=np.uint16)
    
    rounds_master_key = 4
    # arr_master_sbox = np.roll(np.arange(0, 2**16, dtype=np.uint16))
    arr_master_sbox = np.empty((m1*rounds_master_key+1, n1), dtype=np.uint16)
    arr_master_sbox[0] = master_sbox_prepare
    
    jump = 1
    sboxes = arr_sboxes[-1]
    for round_master_key in range(0, rounds_master_key):
        for k in range(0, m1):
            sbox_1= sboxes[k]
            sbox_2 = sboxes[(k+1)%m1]
            for i in range(0, n1):
                i1 = sbox_1[i]
                i2 = sbox_2[i]

                if i1==i2:
                    i2 = (i2+1)%n1

                v1, v2 = master_sbox_prepare[i1], master_sbox_prepare[i2]
                if v1!=i2 and v2!=i1:
                    master_sbox_prepare[i1], master_sbox_prepare[i2] = v2, v1

                # i2 = (i1+jump)%n1
                # v1, v2 = master_sbox_prepare[i1], master_sbox_prepare[i2]
                # v_p = (v1<<8)+v2
                # v_e = sbox_now[v_p]
                # v1_e, v2_e = ((v_e>>8)&0xFF), (v_e&0xFF)
                # master_sbox_prepare[i1], master_sbox_prepare[i2] = v1_e, v2_e

            arr_master_sbox[jump] = master_sbox_prepare
            jump = jump+1

    sys.exit(0)

    d = {}
    n_max = 50

    d_lc = {}
    print("Prepare linear coefficients for 1 to {}.".format(n_max))
    for i in range(1, n_max+1):
        print("i: {}".format(i))
        d_lc[i] = utils_sequence.get_all_linear_coefficients(i)
    print('Done Preparation.')


    # try combining the same 
    n = 32 # max number n (modulo)
    m = 20 # amount of sboxes

    l_ac = d_lc[n]
    arr_a, arr_c = np.array(l_ac).T
    x = np.array((n, ), dtype=np.int64)
    l2 = []
    for _ in range(0, n):
        x = (arr_a*x+arr_c)%n
        l2.append(x)
    X = np.array(l2).T
    X = np.roll(X, 1, axis=1)
    X_flatten = X.flatten()
    X_length = X_flatten.shape[0]

    arr = np.tile(np.arange(0, n), m)
    arr_len = arr.shape[0]
    xs = arr.copy()
    xss = [xs.copy().reshape((m, n))]

    acc_j1 = 0
    for it_round in range(0, 1000):
        print("it_round: {}".format(it_round))
        for i1 in range(0, arr_len):
            j1 = acc_j1
            acc_j1 = (acc_j1+1)%X_length
            k1 = X_flatten[acc_j1]
            i2 = (i1+k1+1)%arr_len

            xs[i1], xs[i2] = xs[i2], xs[i1]

        arr_pseudo_random = np.roll(xs, 1)
        # arr_pseudo_random = xss[4]

        sboxes = np.zeros((m, n), dtype=np.int)
        sbox_num_index = np.zeros((n, ), dtype=np.int)
        sbox_num_pos = np.zeros((m, ), dtype=np.int)

        # create sboxes from the arr_pseudo_random aka np.roll(xs, 1)
        for v in arr_pseudo_random:
            sbox_num = sbox_num_index[v]
            sboxes[sbox_num, sbox_num_pos[sbox_num]] = v
            sbox_num_index[v] += 1
            sbox_num_pos[sbox_num] += 1

        # check, if for any i is s[i]==i true!
        idxs_nums = np.arange(0, n)
        for sbox_nr, sbox in enumerate(sboxes, 0):
            idxs = sbox==idxs_nums
            amount_same_pos = np.sum(idxs)
            if amount_same_pos>0:
                # print("sbox_nr: {}".format(sbox_nr))
                if amount_same_pos==1:
                    i = np.where(idxs)[0]
                    j = 0
                    if i==j:
                        j = 1
                    v1, v2 = sbox[i], sbox[j]
                    sbox[j], sbox[i] = v1, v2
                else:
                    sbox[idxs] = np.roll(sbox[idxs], 1)

        xss.append(sboxes.copy())
        xs = sboxes.flatten()

    xss = np.array(xss)

    # arr_pseudo_random = xss[4]

    # sboxes = np.zeros((m, n), dtype=np.int)
    # sbox_num_index = np.zeros((n, ), dtype=np.int)
    # sbox_num_pos = np.zeros((m, ), dtype=np.int)

    # for v in arr_pseudo_random:
    #     sbox_num = sbox_num_index[v]
    #     sboxes[sbox_num, sbox_num_pos[sbox_num]] = v
    #     sbox_num_index[v] += 1
    #     sbox_num_pos[sbox_num] += 1
