#! /usr/bin/python3.6

import os
import sys

import numpy as np

from PIL import Image

def find_best_approx_array(arr_1, arr_2):
    assert isinstance(arr_1, np.ndarray)
    assert isinstance(arr_2, np.ndarray)

    assert len(arr_1.shape) == 2
    assert len(arr_2.shape) == 2
    assert arr_1.shape == arr_2.shape

    euclid_dist = np.sum((arr_1.reshape((n, 1, m))-arr_2)**2, axis=-1)
    temp_1 = np.vstack((np.argmin(euclid_dist, axis=1), np.arange(0, n)))
    temp_2 = temp_1.T.reshape((-1, )).view("i8,i8")
    best_idx = np.sort(temp_2, order=["f0"]).view("i8").reshape((-1, 2)).T[1]

    return arr_1[best_idx]

def find_best_match_numeric(arr_1, arr_2):
    get_sum = lambda a1, a2: np.sum((a1-a2)**2)

    min_val_numeric = get_sum(arr_1, arr_2)
    best_fit = arr_1
    
    for _ in range(0, 4000):
        arr_1_rnd = np.random.permutation(arr_1)
        val = get_sum(arr_1_rnd, arr_2)
        if min_val_numeric > val:
            min_val_numeric = val
            best_fit = arr_1_rnd

    print("min_val_numeric: {}".format(min_val_numeric))

    return best_fit

def find_best_match(arr_1, arr_2):
    idxs = np.arange(0, arr_1.shape[0])
    arrs_1 = np.vstack((idxs, arr_1)).T.copy().view("i8,i8").reshape((-1, ))
    arrs_2 = np.vstack((idxs, arr_2)).T.copy().view("i8,i8").reshape((-1, ))

    arrs_1 = np.sort(arrs_1, order="f1")
    arrs_2 = np.sort(arrs_2, order="f1")

    # print("arrs_1:\n{}".format(arrs_1))
    # print("arrs_2:\n{}".format(arrs_2))

    arr_1_new = np.zeros((arr_1.shape[0], ), dtype=np.int64)


    arr_1_new[arrs_2["f0"]] = arrs_1["f1"]
    arr_1 = arr_1_new
    # print("arr_1: {}".format(arr_1))
    
    # print("arr_1.dtype: {}".format(arr_1.dtype))
    # print("arr_2.dtype: {}".format(arr_2.dtype))
    min_val = np.sum((arr_1-arr_2)**2)
    print("min_val: {}".format(min_val))

    return arr_1

if __name__ == "__main__":
    n = 30

    arr_1 = np.random.randint(0, 100, (n, ))
    arr_2 = np.random.randint(0, 100, (n, ))

    print("arr_1:\n{}".format(arr_1))
    print("arr_2:\n{}".format(arr_2))

    arr_1_best_numeric = find_best_match_numeric(arr_1, arr_2)
    # print("arr_1_best_numeric: {}".format(arr_1_best_numeric))

    arr_1_best = find_best_match(arr_1, arr_2)
    # print("arr_1_best: {}".format(arr_1_best))

    sys.exit(0)

    euclid_dist = np.sum((arr_1.reshape((n, 1, m))-arr_2)**2, axis=-1)
    print("euclid_dist:\n{}".format(euclid_dist))
    temp_1 = np.vstack((np.argmin(euclid_dist, axis=1), np.arange(0, n)))
    temp_2 = temp_1.T.reshape((-1, )).view("i8,i8")
    best_idx = np.sort(temp_2, order=["f0"]).view("i8").reshape((-1, 2)).T[1]
    print("best_idx: {}".format(best_idx))

    # best_idx_2 = np.sort(np.vstack((np.argmin(euclid_dist, axis=1), np.arange(0, 5))).T.reshape((-1, )).view("i8,i8"), order=["f0"]).view("i8").reshape((-1, 2)).T[1]
    
