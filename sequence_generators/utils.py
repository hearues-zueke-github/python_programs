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
