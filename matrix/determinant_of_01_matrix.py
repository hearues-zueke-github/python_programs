#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

# TODO: need to be python convension confirm! (coding standard as much as possible!)

import datetime
import dill
import gzip
import inspect
import os
import pdb
import shutil
import string
import sys

from copy import deepcopy

import numpy as np
from array2gif import write_gif

from dotmap import DotMap
from PIL import Image, ImageFont, ImageDraw

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

sys.path.append(PATH_ROOT_DIR+"/..")
import utils_all
import utils_serialization


def det_mat_int(A):
    # TODO: test, if A is an integer matrix or not!

    shape = A.shape
    if shape==(1, 1):
        return A[0, 0]
    elif shape==(2, 2):
        return A[0, 0]*A[1, 1]-A[1, 0]*A[0, 1]
    # print("A: {}".format(A))
    # do the general case otherwise!
    s = 0
    sign = (-1)**(shape[0]%2+1)
    choosen_lines = np.arange(0, shape[0]-1)
    A_1 = A[:, 1:]
    A_col_0 = A[:, 0][::-1]
    # print("shape: {}, A_col_0[0]: {}, sign: {}".format(shape, A_col_0[0], sign))
    # print("choosen_lines: {}".format(choosen_lines))
    s += sign*A_col_0[0]*det_mat_int(A_1[choosen_lines])
    for i, v in zip(range(shape[0]-2, -1, -1), A_col_0[1:]):
        # print("shape: {}, i: {}, v: {}, sign: {}".format(shape, i, v, sign))
        # take the other part of A for calculating the det(A_part)!
        sign *= -1
        choosen_lines[i] += 1
        # print("choosen_lines: {}".format(choosen_lines))
        if v==0:
            continue
        s += sign*v*det_mat_int(A_1[choosen_lines])
    # print("s: {}".format(s))
    return s


def test_own_det_func():
    A_1 = np.array([
        [-9, -6, 6, -1],
        [5, 4, -5, -4],
        [0, -6, -7, 6],
        [4, 10, -2, 7]
    ])

    det_a = 3490
    assert int(np.linalg.det(A_1))==det_a
    det_own_a = det_mat_int(A_1)
    # print("det_own_a: {}".format(det_own_a))
    assert det_own_a==det_a

    A_2 = np.array([
        [-2, 6, 5, -10, 6, -9],
        [2, 2, 6, -9, -2, 9],
        [3, 9, 1, -7, 2, 10],
        [0, 5, -1, -2, -9, -9],
        [-10, 6, 9, -10, -3, 1],
        [5, 1, -10, -4, -9, 4],
    ])

    det_A_2 = 2552976
    assert int(np.round(np.linalg.det(A_2)))==det_A_2
    det_own_A_2 = det_mat_int(A_2)
    # print("det_own_A_2: {}".format(det_own_A_2))
    assert det_own_A_2==det_A_2


test_own_det_func()
if __name__ == '__main__':

    # n = 7
    # A = np.zeros((n, n), dtype=np.int64)

    # li = np.array([i for i in range(0, n)])
    # A[(li, li)] = li+1

    # for j in range(0, n):
    #     for i in range(j+1, n):
    #         A[j, i] = j+i+2

    # print("A: {}".format(A))
    # det_A = det_mat_int(A)
    # print("det_A: {}".format(det_A))

    # diag_prod = np.product(li+1)
    # print("diag_prod: {}".format(diag_prod))

    # sys.exit(0)


    # try first random 0 and 1 matrices out with different det!
    obj_path = PATH_ROOT_DIR+'obj_det_max_min_amount.pkl.gz'
    if not os.path.exists(obj_path):
        with gzip.open(obj_path, 'wb') as f:
            dill.dump({}, f)

    assert os.path.exists(obj_path)
    with gzip.open(obj_path, 'rb') as f:
        obj = dill.load(f)

    n = 7
    if not n in obj:
        obj[n] = {}

    d_dets = obj[n]

    if not 'stats' in d_dets:
        d_dets['stats'] = {
            'max_det': 0,
        }
    stats = d_dets['stats']

    if not 'possible_dets' in d_dets:
        d_dets['possible_dets'] = set()
    possible_dets = d_dets['possible_dets']

    for i in range(0, 300000):
        if i%1000==0:
            print("i: {}".format(i))
        A = np.random.randint(-1, 2, (n, n))
        det_A = det_mat_int(A)

        if not det_A in possible_dets:
            possible_dets.add(det_A)
        # print("A:\n{}".format(A))
        # print("- det_A: {}".format(det_A))

        max_det_A = stats['max_det']
        if det_A<max_det_A:
            continue
        elif det_A>max_det_A:
            if max_det_A in d_dets:
                del d_dets[max_det_A]
            stats['max_det'] = det_A

        if not det_A in d_dets:
            d_dets[det_A] = []

        lst_A = d_dets[det_A]

        A_flatten = tuple(A.flatten().tolist())
        # A_T_flatten = tuple(A.T.flatten().tolist())

        if not A_flatten in lst_A:
            lst_A.append(A_flatten)

        # if not A_flatten in lst_A and not A_T_flatten in lst_A:
        #     if A_flatten<A_T_flatten:
        #         lst_A.append(A_flatten)
        #     else:
        #         lst_A.append(A_T_flatten)

    def print_obj(obj):
        keys = sorted(obj.keys())
        print("keys: {}".format(keys))
        for k in keys:
            d = obj[k]
            max_det = d['stats']['max_det']
            l = len(d[max_det])
            print("k: {}, max_det: {}, l: {}".format(k, max_det, l))
    print_obj(obj)

    with gzip.open(obj_path, 'wb') as f:
        dill.dump(obj, f)
