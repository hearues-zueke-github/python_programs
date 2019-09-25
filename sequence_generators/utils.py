#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import numpy as np

def num_to_base(num, b):
    lst = []
    while num > 0:
        lst.append(num%b)
        num //= b
    return lst[::-1]

def get_number_pos_1d(m, lst):
    appearances_counter_abs = np.zeros((m, ), dtype=np.int)
    appearances_counter = np.zeros((m, ), dtype=np.int)
    appearances_positions = np.array([object for _ in range(0, m)])
    appearances_positions_abs = np.zeros((m, ), dtype=np.int)
    for i in range(0, m):
        appearances_positions[i] = []

    for num in lst:
        appearances_counter_abs[num] += 1
        appearances_counter += 1
        p = appearances_counter[num]
        appearances_counter[num] = 0
        appearances_positions[num].append(p)
        appearances_positions_abs[num] += p

    return appearances_counter_abs, appearances_counter, appearances_positions, appearances_positions_abs


def get_number_pos_2d(m, lst):
    appearances_counter_abs = np.zeros((m, m), dtype=np.int)
    appearances_counter = np.zeros((m, m), dtype=np.int)
    appearances_positions = np.array([[object for _ in range(0, m)] for _ in range(0, m)])
    appearances_positions_abs = np.zeros((m, m), dtype=np.int)

    for j in range(0, m):
        for i in range(0, m):
            appearances_positions[j, i] = []

    i1 = lst[0]
    for i2 in lst[1:]:
        idx = (i1, i2)
        appearances_counter_abs[idx] += 1
        appearances_counter += 1
        p = appearances_counter[idx]
        appearances_counter[idx] = 0
        appearances_positions[idx].append(p)
        appearances_positions_abs[idx] += p

        i1 = i2

    return appearances_counter_abs, appearances_counter, appearances_positions, appearances_positions_abs


def get_number_pos_3d(m, lst):
    appearances_counter_abs = np.zeros((m, m, m), dtype=np.int)
    appearances_counter = np.zeros((m, m, m), dtype=np.int)
    appearances_positions = np.array([[[object for _ in range(0, m)] for _ in range(0, m)] for _ in range(0, m)])
    appearances_positions_abs = np.zeros((m, m, m), dtype=np.int)

    for k in range(0, m):
        for j in range(0, m):
            for i in range(0, m):
                appearances_positions[k, j, i] = []

    i1 = lst[0]
    i2 = lst[1]
    for i3 in lst[2:]:
        idx = (i1, i2, i3)
        appearances_counter_abs[idx] += 1
        appearances_counter += 1
        p = appearances_counter[idx]
        appearances_counter[idx] = 0
        appearances_positions[idx].append(p)
        appearances_positions_abs[idx] += p

        i1 = i2
        i2 = i3

    return appearances_counter_abs, appearances_counter, appearances_positions, appearances_positions_abs


def get_sequence_randomness_analysis(m, arr):
    assert isinstance(arr, np.ndarray)

    counts_0d = np.zeros((m, ), dtype=np.uint64)
    counts_1d = np.zeros((m, m), dtype=np.uint64)
    counts_2d = np.zeros((m, m, m), dtype=np.uint64)
    # counts_3d = np.zeros((m, ), dtype=np.uint64)
    length = arr.shape[0]
    
    u0, c0 = np.unique(arr, return_counts=True)
    for i, v in zip(u0, c0):
        counts_0d[i] = v

    arr_1d = np.vstack((arr[:-1], arr[1:])).T.reshape((-1, )).astype(np.uint16).view("u2,u2")
    u1, c1 = np.unique(arr_1d, return_counts=True)
    for (i1, i2), v in zip(u1, c1):
        counts_1d[i1, i2] = v

    arr_2d = np.vstack((arr[:-2], arr[1:-1], arr[2:])).T.reshape((-1, )).astype(np.uint16).view("u2,u2,u2")
    u1, c1 = np.unique(arr_2d, return_counts=True)
    for (i1, i2, i3), v in zip(u1, c1):
        counts_2d[i1, i2, i3] = v

    counts_0d = counts_0d/length
    counts_1d = counts_1d*m/length
    counts_2d = counts_2d*m**2/length

    # mn = np.mean
    sd = np.std
    # get all stds of all counts
    std1_0 = sd(counts_0d)
    std1_1 = sd(counts_1d)
    std1_2 = sd(counts_2d)
    # std1 = np.sum([std1_0, std1_1, std1_2])
    std1 = np.sum([std1_1])

    # std2_0 = sd(counts_0d)
    # std2_1 = sd(sd(counts_1d, axis=1))
    # std2_2 = sd(sd(np.std(counts_2d, axis=2), axis=1))
    # std2 = np.sum([std2_0, std2_1, std2_2])
    # print("std1: {}".format(std1))
    # print("std2: {}".format(std2))

    return (counts_0d, counts_1d, counts_2d), (std1_0, std1_1, std1_2)
    # return std1
