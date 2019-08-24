#! /usr/bin/python3.7

# -*- coding: utf-8 -*-

import dill
import os
import sys

import numpy as np

import matplotlib.pyplot as plt


# m ... amount of different states
# n ... amount of states
def get_all_combinations_repeat(m, n):
    amount = m**n
    arr = np.zeros((amount, n), dtype=np.uint8)
    # arr = np.zeros((amount, n), dtype=np.int)

    arr[:m, n-1] = np.arange(0, m)

    for i in range(1, n):
        r = m**i
        first_col = arr[:r, -i:]
        for j in range(1, m):
            arr[r*j:r*(j+1), -i-1] = j
            arr[r*j:r*(j+1), -i:] = first_col

    return arr


def get_all_combinations_repeat_generator(m, n):
    arr = np.zeros((n, ), dtype=np.uint8).copy()
    yield arr.copy()

    i_start = arr.shape[0]-1
    while True:
        i = i_start
        while i >= 0:
            arr[i] += 1
            if arr[i] < m:
                break
            arr[i] = 0
            i -= 1
        if i < 0:
            break
        yield arr.copy()


# m ... max num for state
# n ... amount of states
def get_all_combinations_increment(m, n):
    sum_row = np.arange(1, m+1)
    
    sum_table = np.zeros((n, m), dtype=np.int)
    sum_table[0] = sum_row
    idx_table = np.zeros((n, m), dtype=np.int)
    idx_table[0] = sum_row
    for i in range(1, n):
        idx_table[i] = np.cumsum(sum_row[::-1])
        sum_row = np.cumsum(sum_row)
        sum_table[i] = sum_row

    globals()["sum_table"] = sum_table

    arr = np.zeros((sum_row[-1], n), dtype=np.int)

    arr[:m, -1] = np.arange(0, m)
    for col, (idx_row_prev, idx_row) in enumerate(zip(idx_table[:-1], idx_table[1:]), 1):
        for i, (i1, i2) in enumerate(zip(idx_row[:-1], idx_row[1:]), 1):
            arr[i1:i2, -col-1] = i
            arr[i1:i2, -col:] = arr[idx_row_prev[i-1]:idx_row_prev[-1], -col:]

    return arr


# n ... amount of states
def get_permutation_table(n, same_pos=True):
    if n == 1:
        return np.array([[1]], dtype=np.uint8)
    arr = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    
    p = 2
    for i in range(3, n+1):
        arr_new = np.zeros((p*i, i), dtype=np.uint8)

        for j in range(0, i):
            arr_new[p*j:p*(j+1), 0] = (j-1) % i
            arr_new[p*j:p*(j+1), 1:] = (arr+j) % i

        p *= i
        arr = arr_new

    arr = np.sort(arr.reshape((-1, )).view(','.join(['u1']*n))).view('u1').reshape((-1, n))

    if not same_pos:
        arr = arr[~np.any(arr == np.arange(0, n), axis=1)]

    return arr

# n ... amount of numbers
# m ... length of permutations
def get_all_permutations_increment(n, m):
    return get_all_combinations_increment(n-m+1, m)+np.arange(0, m)
    

# n ... amount of numbers
# m ... length of permutations
# instead of e.g. [1, 2, 3] it will also contain [2, 3, 1], [1, 3, 2], etc.
# each permutation of [1, 2, 3] including!
def get_all_permutations_parts(n, m):
    arr = get_all_permutations_increment(n, m)
    size = arr.shape[0]
    perm_tbl = get_permutation_table(m)
    big_arr = np.zeros((arr.shape[0]*perm_tbl.shape[0], m), dtype=np.int)
    for i, row in enumerate(perm_tbl, 0):
        big_arr[size*i:size*(i+1)] = arr[:, row]
    return big_arr


# m ... max num for state
# n ... amount of states
def get_amount_of_increment_combinations(m, n):
    arr = get_all_combinations_increment(m, n)

    amount_arr = np.zeros((arr.shape[0], m), dtype=np.int)

    for i in range(0, m):
        amount_arr[:, i] = np.sum(arr==i, axis=1)

    return amount_arr


def move_all_values_to_left(arr):
    # arr = np.random.randint(0, 10, (10, 4))

    idx = arr!=0
    # print("arr:\n{}".format(arr))
    # print("idx:\n{}".format(idx))

    arr_cpy = np.zeros(arr.shape, dtype=arr.dtype)
    idx_col = np.zeros((arr.shape[0], ), dtype=np.int)

    cols = arr.shape[1]
    for i in range(0, cols):
        vals = arr[:, i]
        vals_not_zero = vals!=0

        arr_cpy[vals_not_zero, idx_col[vals_not_zero]] = vals[vals_not_zero]
        idx_col += vals_not_zero

    return arr_cpy

    # print("arr_cpy: {}".format(arr_cpy))


def get_unique_addition_combos(m, n):
    amount_arr = get_amount_of_increment_combinations(m, n)

    amount_arr_shifted = move_all_values_to_left(amount_arr)
    arr_view = amount_arr_shifted.astype(np.uint8).view("u1"+("" if m == 1 else ",u1"*(m-1))).reshape((-1, ))
    arr_unique = np.unique(arr_view).view("u1").reshape((-1, m))

    idx = arr_unique!=0
    amount_per_row = np.sum(idx, axis=1)
    amount_arr_no_zero = arr_unique[idx].tolist()
    i_idxs = np.hstack(((0, ), np.cumsum(amount_per_row)))

    amount_per_row_lst = [amount_arr_no_zero[i1:i2] for i1, i2 in zip(i_idxs[:-1], i_idxs[1:])]
    amount_per_row_unique_lst = list(set(map(tuple, amount_per_row_lst)))

    unique_lengths = {i: [] for i in range(1, m+1)}

    for unique_addition in amount_per_row_unique_lst:
        unique_lengths[len(unique_addition)].append(unique_addition)

    # return amount_per_row, amount_per_row_unique_lst, unique_lengths
    return unique_lengths


def table_of_unique_numbers():
    m = 3
    k = 10

    for n in range(m, m+k):
        i = n
        unique_lengths = get_unique_addition_combos(m, n)

        lengths = {key: len(value) for key, value in unique_lengths.items()}

        print("\nm: {}, n: {}".format(m, n))
        print("lengths:\n{}".format(lengths))


def print_dict(d):
    for key, value in d.items():
        print("  {}: {}".format(key, value))


def table_of_unique_numbers_2():
    m = 3
    k = 5

    for n in range(m, m+k):
        arr = get_all_combinations_repeat(2, n)
        unique_lengths = get_unique_addition_combos(n, n)
        print("\nn: {}".format(n))
        print("arr.shape: {}".format(arr.shape))

        sums = np.sum(arr, axis=1)
        # print("  sums: {}".format(sums))
        # diff = sums[1:]-sums[:-1]
        # print("  diff: {}".format(diff))
        # print("  diff[np.arange(3, diff.shape[0], 4)]: {}".format(  diff[np.arange(3, diff.shape[0], 4)]))

        # print("  np.sum(diff): {}".format(  np.sum(diff)))

        lr_dict = {}
        for i in range(0, n+1):
            lst = arr[sums==i].tolist()
            lr_lst = list(map(lambda x: "".join(list(map(lambda y: {0: "L", 1: "R"}[y], x))), lst))
            lr_dict[(n-i, i)] = lr_lst

        print("lr_dict:")
        print_dict(lr_dict)
        print("unique_lengths:")
        print_dict(unique_lengths)

    globals()["arr"] = arr
    globals()["lr_dict"] = lr_dict


def simple_example():
    m = 3
    n = 4

    arr_repeat = get_all_combinations_repeat(m, n)
    arr_increment = get_all_combinations_increment(m, n)
    amount_arr_increment = get_amount_of_increment_combinations(m, n)

    print("\nm: {}, n: {}".format(m, n))
    print("arr_repeat:\n{}".format(arr_repeat))
    print("arr_increment:\n{}".format(arr_increment))
    print("amount_arr_increment: {}".format(amount_arr_increment))

    print("\nm: {}, n: {}".format(m, n))
    print("arr_repeat.shape:\n{}".format(arr_repeat.shape))
    print("arr_increment.shape:\n{}".format(arr_increment.shape))
    print("amount_arr_increment.shape: {}".format(amount_arr_increment.shape))


# global simple tests
arr_1_1 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=np.uint8)
arr_1_2 = get_all_combinations_repeat(2, 3)
assert np.all(arr_1_1==arr_1_2)

arr_2_1 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 1], [0, 1, 2], [0, 2, 2], [1, 1, 1], [1, 1, 2], [1, 2, 2], [2, 2, 2]])
arr_2_2 = get_all_combinations_increment(3, 3)
assert np.all(arr_2_1==arr_2_2)


if __name__ == "__main__":
    # move_all_values_to_left()
    
    # simple_example()
    # table_of_unique_numbers()
    table_of_unique_numbers_2()
