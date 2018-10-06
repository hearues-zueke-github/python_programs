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

if __name__ == "__main__":
    n = 100
    m = 3

    arr_1 = np.random.randint(0, 10, (n, m))
    arr_2 = np.random.randint(0, 10, (n, m))

    print("arr_1:\n{}".format(arr_1))
    print("arr_2:\n{}".format(arr_2))

    euclid_dist = np.sum((arr_1.reshape((n, 1, m))-arr_2)**2, axis=-1)
    print("euclid_dist:\n{}".format(euclid_dist))
    temp_1 = np.vstack((np.argmin(euclid_dist, axis=1), np.arange(0, n)))
    temp_2 = temp_1.T.reshape((-1, )).view("i8,i8")
    best_idx = np.sort(temp_2, order=["f0"]).view("i8").reshape((-1, 2)).T[1]
    print("best_idx: {}".format(best_idx))

    # best_idx_2 = np.sort(np.vstack((np.argmin(euclid_dist, axis=1), np.arange(0, 5))).T.reshape((-1, )).view("i8,i8"), order=["f0"]).view("i8").reshape((-1, 2)).T[1]

