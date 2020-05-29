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

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"

import utils_sequence

def check_two_linear_coefficients(n1, n2, a1, c1, a2, c2):
    l1 = []
    l2 = []
    x1 = 0
    x2 = 0
    for i in range(0, n1*n2):
        x1 = (a1*x1+c1)%n1
        x2 = (a2*x2+c2+x1)%n2
        l1.append(x1)
        l2.append(x2)
    arr1 = np.array(l1)
    arr2 = np.array(l2)
    u1, c1 = np.unique(arr1, return_counts=True)
    u2, c2 = np.unique(arr2, return_counts=True)
    assert u1.shape[0]==n1 and np.all(c1==c1[0])
    assert u2.shape[0]==n2 and np.all(c2==c2[0])


def get_all_two_linear_coefficients(d_lc, n1, n2):
    l_ac1 = d_lc[n1]
    l_ac2 = d_lc[n2]

    l_possible_two_linear_coefficients = []

    t_a1, t_c1 = list(zip(*l_ac1))
    arr_a1 = np.array(t_a1)
    arr_c1 = np.array(t_c1)

    t_a2, t_c2 = list(zip(*l_ac2))
    arr_a2 = np.array(t_a2)
    arr_c2 = np.array(t_c2)

    arr_a2_T = arr_a2.reshape((-1, 1))
    arr_c2_T = arr_c2.reshape((-1, 1))

    l1 = []
    l2 = []
    x1 = np.zeros((arr_a1.shape[0], ), dtype=np.int64)
    x2 = np.zeros((arr_a2_T.shape[0], arr_a1.shape[0]), dtype=np.int64)
    for i in range(0, n1*n2):
        x1 = (arr_a1*x1+arr_c1)%n1
        x2 = (arr_a2_T*x2+arr_c2_T+x1)%n2

        l1.append(x1)
        l2.append(x2)

    X1 = np.array(l1).T
    X1v = X1.flatten().view('i8'+',i8'*(X1.shape[1]-1))
    lu, lcounts = np.frompyfunc(lambda x: np.unique(x, return_counts=True), 1, 2)(X1v)
    for u, counts in zip(lu, lcounts):
        assert u.shape[0]==n1 and np.all(counts==counts[0])

    X2_orig = np.array(l2)
    X2 = np.array(l2).transpose(1, 2, 0).reshape((len(l_ac2), -1))
    X2v = X2.copy().view('i8'+',i8'*(n1*n2-1))
    lu, lcounts = np.frompyfunc(lambda x: np.unique(x, return_counts=True), 1, 2)(X2v)
    for a2, c2, u_row, counts_row in zip(arr_a2, arr_c2, lu, lcounts):
        for a1, c1, u, counts in zip(arr_a1, arr_c1, u_row, counts_row):
            if u.shape[0]==n2 and np.all(counts==counts[0]):
                l_possible_two_linear_coefficients.append((a1, c1, a2, c2))

    return l_possible_two_linear_coefficients


def get_any_two_linear_coefficients(n1, n2):
    l_ac1 = utils_sequence.get_all_linear_coefficients(n1)
    l_ac2 = utils_sequence.get_all_linear_coefficients(n2)

    while True:
        i1 = np.random.randint(0, len(l_ac1))
        i2 = np.random.randint(0, len(l_ac2))
        a1, c1 = l_ac1[i1]
        a2, c2 = l_ac2[i2]
        l1 = []
        l2 = []
        x1 = 0
        x2 = 0
        for i in range(0, n1*n2):
            x1 = (a1*x1+c1)%n1
            x2 = (a2*x2+x1+c2)%n2

            l1.append(x1)
            l2.append(x2)

        u1, counts1 = np.unique(l1, return_counts=True)
        u2, counts2 = np.unique(l2, return_counts=True)

        assert u1.shape[0]==n1 and np.all(counts1==counts1[0])

        if u2.shape[0]==n2 and np.all(counts2==counts2[0]):
            print("i1: {}, i2: {}".format(i1, i2))
            return (a1, c1, a2, c2)


def get_all_three_linear_coefficients(d_lc, n1, n2, n3):
    l_ac1 = d_lc[n1]
    l_ac2 = d_lc[n2]
    l_ac3 = d_lc[n3]

    # d_possible_three_linear_coefficients = {}
    # d_combos = {}
    l_possible_three_linear_coefficients = []
    l_combos = []

    # max_cycle_length = np.multiply.reduce((n1, n2, n3))
    # max_cycle_length = utils_sequence.calc_lsm((n1, n2, n3))

    cycle_lengths = []
    cycle_lengths_approven = []

    for a1, c1 in l_ac1:
        for a2, c2 in l_ac2:
            for a3, c3 in l_ac3:
                l = []
                l1 = []
                l2 = []
                l3 = []
                x1 = 0
                x2 = 0
                x3 = 0


                # is_full_length_cycle = True
                for i in range(0, n1*n2*n3):
                    x1 = (a1*x1+c1)%n1
                    x2 = (a2*x2+c2+x1)%n2
                    # x3 = (a3*x3+c3+x2)%n3
                    x3 = (a3*x3+c3+x1+x2)%n3

                    t = (x1, x2, x3)
                    if t in l:
                        break
                        # pass
                    else:
                        l.append(t)

                    l1.append(x1)
                    l2.append(x2)
                    l3.append(x3)

                # print("l: {}".format(l))
                # print("len(l): {}".format(len(l)))
                # u1, ct1 = np.unique(l1, return_counts=True)
                # u2, ct2 = np.unique(l2, return_counts=True)
                u3, ct3 = np.unique(l3, return_counts=True)
                # assert u1.shape[0]==n1 and np.all(ct1==ct1[0])
                # # assert u2.shape[0]==n2 and np.all(ct2==ct2[0])

                length = len(l)
                cycle_lengths.append(length)

                # if not length in d_possible_three_linear_coefficients:
                #     d_possible_three_linear_coefficients[length] = []
                #     d_combos[length] = []
                # d_possible_three_linear_coefficients[length].append((a1, c1, a2, c2, a3, c3))
                # d_combos[length].append(l3)

                # if length==max_cycle_length:
                if u3.shape[0]==n3 and np.all(ct3==ct3[0]):
                    l_possible_three_linear_coefficients.append((a1, c1, a2, c2, a3, c3))
                    l_combos.append(l3)
                    cycle_lengths_approven.append(length)

    u, c = np.unique(cycle_lengths, return_counts=True)
    # print("n1: {}, n2: {}, n3: {}".format(n1, n2, n3))
    # print("cycle_lengths: u: {}".format(u))
    # print("cycle_lengths: c: {}".format(c))

    u, c = np.unique(cycle_lengths_approven, return_counts=True)
    # print("n1: {}, n2: {}, n3: {}".format(n1, n2, n3))
    # print("cycle_lengths_approven: u: {}".format(u))
    # print("cycle_lengths_approven: c: {}".format(c))


    # l_possible_three_linear_coefficients = d_possible_three_linear_coefficients[u[-1]]
    # l_combos = d_combos[u[-1]]

    # l_possible_three_linear_coefficients_approve = []
    # l_combos_approve = []

    # for l, l_comb in zip(l_possible_three_linear_coefficients, l_combos):
    #     u, c = np.unique(l, return_counts=True)
    #     if u.shape[0]==n3 and np.all(c==c[0]):
    #         l_possible_three_linear_coefficients_approve.append(l)
    #         l_combos_approve.append(l_comb)

    # return l_possible_three_linear_coefficients_approve, l_combos_approve
    return l_possible_three_linear_coefficients, l_combos

    # t_a1, t_c1 = list(zip(*l_ac1))
    # arr_a1 = np.array(t_a1)
    # arr_c1 = np.array(t_c1)

    # t_a2, t_c2 = list(zip(*l_ac2))
    # arr_a2 = np.array(t_a2)
    # arr_c2 = np.array(t_c2)

    # arr_a2_T = arr_a2.reshape((-1, 1))
    # arr_c2_T = arr_c2.reshape((-1, 1))

    # l1 = []
    # l2 = []
    # x1 = np.zeros((arr_a1.shape[0], ), dtype=np.int64)
    # x2 = np.zeros((arr_a2_T.shape[0], arr_a1.shape[0]), dtype=np.int64)
    # for i in range(0, n1*n2):
    #     x1 = (arr_a1*x1+arr_c1)%n1
    #     x2 = (arr_a2_T*x2+arr_c2_T+x1)%n2

    #     l1.append(x1)
    #     l2.append(x2)

    # X1 = np.array(l1).T
    # X1v = X1.flatten().view('i8'+',i8'*(X1.shape[1]-1))
    # lu, lcounts = np.frompyfunc(lambda x: np.unique(x, return_counts=True), 1, 2)(X1v)
    # for u, counts in zip(lu, lcounts):
    #     assert u.shape[0]==n1 and np.all(counts==counts[0])

    # X2_orig = np.array(l2)
    # X2 = np.array(l2).transpose(1, 2, 0).reshape((len(l_ac2), -1))
    # X2v = X2.copy().view('i8'+',i8'*(n1*n2-1))
    # lu, lcounts = np.frompyfunc(lambda x: np.unique(x, return_counts=True), 1, 2)(X2v)
    # for a2, c2, u_row, counts_row in zip(arr_a2, arr_c2, lu, lcounts):
    #     for a1, c1, u, counts in zip(arr_a1, arr_c1, u_row, counts_row):
    #         if u.shape[0]==n2 and np.all(counts==counts[0]):
    #             l_possible_three_linear_coefficients.append((a1, c1, a2, c2))

    return l_possible_three_linear_coefficients


if __name__=='__main__':
    # lens_1_lin_coeffs = []
    # for i in range(1, 51):
    #     l_lin_coeffs = utils_sequence.get_all_linear_coefficients(i)
    #     lens_1_lin_coeffs.append(len(l_lin_coeffs))
    # print("lens_1_lin_coeffs: {}".format(lens_1_lin_coeffs))
    # sys.exit(0)

    lens = []
    d = {}
    n_max = 50

    d_lc = {}
    print("Prepare linear coefficients for 1 to {}.".format(n_max))
    for i in range(1, n_max+1):
        print("i: {}".format(i))
        d_lc[i] = utils_sequence.get_all_linear_coefficients(i)
    print('Done Preparation.')


    # try combining the same 
    n = 50
    l = d_lc[n]

    print("l: {}".format(l))

    arr_a, arr_c = np.array(l).T
    x = np.array((n, ), dtype=np.int64)
    l2 = []
    for _ in range(0, n):
        x = (arr_a*x+arr_c)%n
        l2.append(x)
    X = np.array(l2).T
    X = np.roll(X, 1, axis=1)

    x1 = X[1]
    x2 = X[2]

    xs = X.flatten()
    xs_orig = xs.copy()
    # xs = X.flatten()
    length = xs_orig.shape[0]
    xs = np.arange(0, length)

    xss = [xs.copy()]

    acc_i1 = 0
    for _ in range(0, 1000):
        for i1 in xs_orig:
            j1 = acc_i1
            acc_i1 = (acc_i1+i1+1)%length
            j2 = acc_i1

            xs[j1], xs[j2] = xs[j2], xs[j1]
        acc_i1 = (acc_i1+1)%length
        xss.append(xs.copy())
    xss = np.array(xss)

    sys.exit(0)


    # check, if the function get_all_two_linear_coefficients is working properly!
    
    # Do some hardcoded testcases!
    assert set(get_all_two_linear_coefficients(d_lc, 4, 6))==set([(1, 1, 1, 1), (1, 1, 1, 5), (1, 3, 1, 1), (1, 3, 1, 5)])
    l_5_9 = get_all_two_linear_coefficients(d_lc, 5, 9)
    assert set(l_5_9)==set([(1, 1, 1, 2), (1, 1, 1, 5), (1, 1, 1, 8), (1, 1, 4, 2),
    (1, 1, 4, 5), (1, 1, 4, 8), (1, 1, 7, 2), (1, 1, 7, 5), (1, 1, 7, 8), (1, 2, 1, 2), (1, 2, 1, 5), (1, 2, 1, 8),
    (1, 2, 4, 2), (1, 2, 4, 5), (1, 2, 4, 8), (1, 2, 7, 2), (1, 2, 7, 5), (1, 2, 7, 8), (1, 3, 1, 2), (1, 3, 1, 5),
    (1, 3, 1, 8), (1, 3, 4, 2), (1, 3, 4, 5), (1, 3, 4, 8), (1, 3, 7, 2), (1, 3, 7, 5), (1, 3, 7, 8), (1, 4, 1, 2),
    (1, 4, 1, 5), (1, 4, 1, 8), (1, 4, 4, 2), (1, 4, 4, 5), (1, 4, 4, 8), (1, 4, 7, 2), (1, 4, 7, 5), (1, 4, 7, 8)])

    # Do some random choosen testcases!
    ln12 = np.random.randint(1, n_max, (10, 2)).tolist()
    for n1, n2 in ln12:
        print("n1: {}, n2: {}".format(n1, n2))
        l = get_all_two_linear_coefficients(d_lc, n1, n2)
        l_ac1 = d_lc[n1]
        l_ac2 = d_lc[n2]
        l_t = []
        for a1, c1 in l_ac1:
            for a2, c2 in l_ac2:
                t = (a1, c1, a2, c2)
                l_t.append(t)

                if t in l:
                    check_two_linear_coefficients(n1, n2, a1, c1, a2, c2)
                else:
                    try:
                        check_two_linear_coefficients(n1, n2, a1, c1, a2, c2)
                        assert False
                    except:
                        pass
    print('Done Testcases!')

    d = {}
    for n1 in range(1, 11):
        for n2 in range(1, 11):
            for n3 in range(1, 11):
                l2, l_combos = get_all_three_linear_coefficients(d_lc, n1, n2, n3)
                # l2, l_combos = get_all_three_linear_coefficients(d_lc, 4, 8, 13)
                len_l2 = len(l2)
                print("n1: {}, n2: {}, n3: {}, len_l2: {}".format(n1, n2, n3, len_l2))
                # print("len_l2: {}".format(len_l2))
                d[(n1, n2, n3)] = len_l2

    d_v_to_k = {}
    for k, v in d.items():
        if not v in d_v_to_k:
            d_v_to_k[v] = []
        d_v_to_k[v].append(k)
    sys.exit(0)

    for n2 in range(1, n_max+1):
        for n1 in range(1, n_max+1):
            print("n1: {}, n2: {}".format(n1, n2))
            l_two_lin_coeffs = get_all_two_linear_coefficients(d_lc, n1, n2)
            d[(n1, n2)] = l_two_lin_coeffs
            lens.append(len(l_two_lin_coeffs))

    print("lens: {}".format(lens))

    # find all zero coefficients!
    l_zero_coeffs = []
    for k, v in d.items():
        if len(v)==0:
            l_zero_coeffs.append(k)
    print("l_zero_coeffs: {}".format(l_zero_coeffs))

    x, y = list(zip(*l_zero_coeffs))

    pix = np.zeros((n_max, n_max), dtype=np.uint8)
    for i, j in l_zero_coeffs:
        pix[j-1, i-1] = 255
    # d2 = {k: len(v) for k, v in d.items()}
    # max_v = np.max(list(d2.values()))
    
    # for (n1, n2), v in d2.items():
    #     pix[n1-1, n2-1] = int((v/max_v)*255.999)

    pix = np.vstack((np.zeros((n_max, ), dtype=np.uint8), pix))
    pix = np.hstack((np.zeros((n_max+1, 1), dtype=np.uint8), pix))

    pix[0] = 32
    pix[:, 0] = 32
    pix[0, np.arange(2, n_max+1, 2)] = 224
    pix[np.arange(2, n_max+1, 2), 0] = 224
    pix[0, 0] = 128

    img = Image.fromarray(pix)
    img = img.resize((img.width*10, img.height*10))
    # this image contains all coefficients, which will deliver no two linear combination!
    img.show()


