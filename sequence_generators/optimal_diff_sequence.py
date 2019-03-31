#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import functools
import os
import sys
import time

import numpy as np

from indexed import IndexedOrderedDict

from PIL import Image, ImageTk

import multiprocessing as mp
from multiprocessing import Process, Pipe, Lock

from collections import namedtuple

import utils_sequence

from numpy.lib.recfunctions import append_fields

path_root_dir = os.path.dirname(os.path.abspath(__file__))+"/"

sys.path.append(path_root_dir+"../combinatorics")
from different_combinations import get_all_combinations_repeat

# class Partition(namedtuple('PartitionVals', 'n m q p')):
#     [...]
class Partition(Exception):
    __slots__ = ('n', 'm', 'q', 'p')

    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.q = n//(m-1)
        self.p = n%(m-1)

    def __str__(self):
        return '({n}, {m}, {q}, {p})'.format(n=self.n, m=self.m, q=self.q, p=self.p)


def get_multipliers(l, base): # base 2
    return np.array([base**i for i in range(l-1, -1, -1)], dtype=object)


def get_amount_of_01_sequence(arr):
    assert np.sum(arr==0) > 0
    assert np.sum(arr==1) > 0
    assert np.sum(arr<0)+np.sum(arr>1) == 0
    arr = np.hstack(((-1, ), arr, (-1, )))
    idx = np.where((arr[1:]==arr[:-1])==False)[0]
    amounts = idx[1:]-idx[:-1]
    # print("arr: {}".format(arr))
    # print("idx: {}".format(idx))
    # print("amounts: {}".format(amounts))

    return amounts


if __name__ == "__main__":
    print("path_root_dir: {}".format(path_root_dir))

    n = 20
    partitions = [] # will have tuples of ((n, m), (m, q, p))
    partitions2 = []
    # q...n//m
    # p...n%m

    # partition go from 0 to 20
    # e.g. if we want 4 partitions, we have 20//3 == 6 and 20%3 == 2
    # which means that we need 2x 7 and 1x 6 diffs
    # in this case we will get the sequence: 7, 7, 6 -> which gives 0, 7, 14, 20
    # we could simplify the diff sequence as: 1, 1, 0
    # also the optimal diff sequence would be: 1, 0, 1
    # which gives the following sequence: 0, 7, 13, 20
    for m in range(2, n+1):
        partitions.append(((n, m), (m, n//(m-1), n%(m-1))))
        partitions2.append(Partition(n, m))
    # n = (q+1)*p+q*(m-1-p)
    # print("partitions:\n{}".format(partitions))


    # # try different fix m, p and changing only p to get n:
    # ns = []
    # m = 8
    # p = 2
    # print("m: {}, p: {}".format(m, p))
    # def get_n(m, q, p):
    #     return (q+1)*p+q*(m-1-p)
    # for q in range(1, 21):
    #     ns.append((q, get_n(m, q, p)))


    lens_max_lengths = []

    all_choosen_amounts_per_row = {}
    all_opt_arrs = {}
    # m = 7
    for m in range(2, 15):
        print("m: {}".format(m))
        arr = get_all_combinations_repeat(2, m-1)
        ps = {}
        for p in range(1, m-1):
            ps[p] = arr[np.sum(arr, axis=1)==p]

        lens_max_lengths_p = []

        arr_opt = np.zeros((m-2, m-1), dtype=np.int)

        # p...1:m-1
        lsts = []
        for p in range(1, m-1):
            print("p: {}".format(p))
            # p = 5
            arr_p = ps[p]
            # print("arr_p:\n{}".format(arr_p))

            amounts_per_row = [get_amount_of_01_sequence(row) for row in arr_p]
            len_per_row = [row.shape[0] for row in amounts_per_row]

            # print("amounts_per_row:\n{}".format(amounts_per_row))
            # print("len_per_row:\n{}".format(len_per_row))


            idxs = np.where(len_per_row==np.max(len_per_row))[0]
            print("idxs: {}".format(idxs))
            print("p: {}, len(idxs): {}".format(p, len(idxs)))
            choosen_amounts_per_row = np.array([amounts_per_row[idx] for idx in idxs])
            print("choosen_amounts_per_row:\n{}".format(choosen_amounts_per_row))
            all_choosen_amounts_per_row[(m, p)] = choosen_amounts_per_row
            len_idxs = len(idxs)
            lens_max_lengths_p.append(len(idxs))

            arr_p = arr_p[idxs]
            # if len_idxs > 1:

            # input("ENTER...")

            arr_p_inverse = arr_p.copy()
            for i in range(0, arr_p.shape[0]):
                arr_p_inverse[i] = arr_p[i][::-1]

            print("arr_p:\n{}".format(arr_p))
            print("arr_p_inverse:\n{}".format(arr_p_inverse))

            # multipliers = np.array([2**i for i in range(arr_p.shape[1]-1, -1, -1)], dtype=object)
            multipliers = get_multipliers(arr_p.shape[1]-1, 2)
            # print("multipliers: {}".format(multipliers))

            sums_p = np.sum(arr_p*multipliers, axis=1)
            sums_inv_p = np.sum(arr_p_inverse*multipliers, axis=1)

            print("sums_p: {}".format(sums_p))
            print("sums_inv_p: {}".format(sums_inv_p))

            diffs = sums_p-sums_inv_p
            print("diffs: {}".format(diffs))

            # arr = np.array([[0, 3], [1, 7], [2, 5], [3, 1]], dtype=object)
            idx = np.argsort(diffs)

            diffs = diffs[idx]
            arr_p = arr_p[idx]
            arr_p_inverse = arr_p_inverse[idx]
            sums_p = sums_p[idx]
            sums_inv_p = sums_inv_p[idx]

            i_opt = np.where(diffs >= 0)[0][0]
            # print("i_opt: {}".format(i_opt))

            diffs = diffs[i_opt:]
            arr_p = arr_p[i_opt:]
            arr_p_inverse = arr_p_inverse[i_opt:]
            sums_p = sums_p[i_opt:]
            sums_inv_p = sums_inv_p[i_opt:]

            # print("diffs:\n{}".format(diffs))
            # print("arr_p:\n{}".format(arr_p))
            # print("sums_p:\n{}".format(sums_p))
            # print("sums_inv_p:\n{}".format(sums_inv_p))

            amount_of_min_diff = np.sum(diffs==diffs[0])
            # print("amount_of_min_diff: {}".format(amount_of_min_diff))

            opt_idx = 0
            arr_p_opt = arr_p[opt_idx]
            if amount_of_min_diff > 1:
                opt_idx = np.argsort(sums_p[:amount_of_min_diff])[0]
                arr_p_opt = arr_p[opt_idx]

            # print("arr_p_opt: {}".format(arr_p_opt))
            # lsts.append((p, arr_p_opt))
            arr_opt[p-1] = arr_p_opt
        lens_max_lengths.append((m, lens_max_lengths_p))
        all_opt_arrs[m] = arr_opt

        # break

    # x = np.array(np.arange(0,10), dtype = [('x', float)]);
    # y = np.array(np.arange(10,20), dtype = [('y', float)])
    # z = append_fields(x, 't', np.array(y, dtype=np.float))
    # print("z:\n{}".format(z))

    # x = np.array(arr[:, 0], dtype=[('a1', object)])
    # y = np.array(arr[:, 1], dtype=[('a2', object)])
    # # z = append_fields(x, 'a2', arr[:, 1])
    # print("x:\n{}".format(x))
    # print("y:\n{}".format(y))
