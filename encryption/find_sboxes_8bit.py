#! /usr/bin/python2.7
# -*- coding: utf-8 -*- 

import os
import sys

import numpy as np

import matplotlib.pyplot as plt

from Utils import pretty_block_printer, clrs

# np.set_printoptions(formatter={'int': lambda x: "0x{:02X}".format(x)})

def f(a, x):
    # print("a: {}".format(a))
    s = np.zeros(x.shape, dtype=np.uint8)

    for i, c in a:
        s += c*x**i
    
    return s

def swap_4bits(sbox_8bit):
    return (sbox_8bit>>4)|(sbox_8bit<<4)

def apply_f_n_times(a, n):
    x_0 = np.arange(0, 256).astype(np.uint8)
    x_prev = x_0+0
    x_next = np.roll(f(a, x_prev), 1)
    x_next = swap_4bits(x_next)

    # same_idx = np.where(x_next==x_0)[0].astype(np.uint8)

    print("{}x_next{}:".format(clrs.lcb, clrs.rst))
    pretty_block_printer(x_next, 8, 256)

    # print("{}same_idx{}:".format(clrs.lgb, clrs.rst))
    # pretty_block_printer(same_idx, 8, len(same_idx))

    for i in xrange(0, n):
        x_new = x_prev[x_next]
        x_prev = x_next
        x_next = np.roll(f(a, x_new), 1)
        x_next = swap_4bits(x_next)

        print("i: {}{}{}, {}x_next{}:".format(clrs.lwb, i, clrs.rst, clrs.lcb, clrs.rst))
        pretty_block_printer(x_next, 8, 256)

    if check_if_sbox_good(x_next) == False:
        return None

    return x_next

def apply_f_n_times_roll(a):
    x_0 = np.arange(0, 256).astype(np.uint8)

    x_prev = x_0+0
    x_next = x_0+0

    for i in a:
        x_new = x_prev[np.roll(x_next, i)]
        x_prev = x_next
        x_next = swap_4bits(x_new)
        x_next += i

    if check_if_sbox_good(x_next) == False:
        return None

    return x_next

def apply_f_az(az, x):
    for a in az:
        x = f(a, x)
        if check_if_sbox_good(x) == False:
            return None
    return x

def check_if_sbox_good(sbox_8bit):
    arr_sorted = np.sort(sbox_8bit)
    diff = arr_sorted[1:]-arr_sorted[:-1]
    first_test = np.sum(diff!=1)==0
    # arr_0 = np.arange(0, 256).astype(np.uint8)
    # second_test = np.sum(sbox_8bit==arr_0)==0
    return first_test
    # return first_test and second_test

def find_acceptable_constants():
    acceptable_constants = []
    x = np.arange(0, 256).astype(np.uint8)

    for i1 in xrange(0, 8):
     for i2 in xrange(0, 8):
      for i3 in xrange(0, 8):
       for i4 in xrange(0, 8):
        for i5 in xrange(0, 8):
            a = [i1, i2, i3, i4, i5]

            sbox_8bit = f(a, np.arange(0, 256).astype(np.uint8))

            is_equal_poss = np.sum(sbox_8bit==x) > 0
            if is_equal_poss:
                # print("for a: {} some values are same like the index".format(a))
                continue

            arr_sorted = np.sort(sbox_8bit)
            diff = arr_sorted[1:]-arr_sorted[:-1]
            is_good_sbox = np.sum(diff!=1)==0

            if is_good_sbox:
                acceptable_constants.append(a)

    return acceptable_constants

def get_sbox_cycles(sbox_8bit):
    cycles = []
    poss_idx = np.arange(0, 256).tolist()

    while len(poss_idx) > 0:
        idx_start = poss_idx.pop(0)
        idx = sbox_8bit[idx_start]
        cycle = [idx_start]

        while idx != idx_start:
            poss_idx.remove(idx)
            idx_prev = idx
            idx = sbox_8bit[idx]
            cycle.append(idx_prev)

        cycles.append(np.array(cycle))

    return cycles

if __name__ == "__main__":
    # acceptable_constants = find_acceptable_constants()
    # for i, accept_consts in enumerate(acceptable_constants):
    #     print("i: {}, accept_consts: {}".format(i, accept_consts))

    # sys.exit(0)

    x_0 = np.arange(0, 256).astype(np.uint8)
    a = [2, 10, 12, 13, 17, 33, 22]
    # a = [(0, 5), (1, 7), (2, 10), (4, 14)]

    # az = [[(0, 5), (1, 5), (2, 4), (4, 14)],
    #       [(0, 5), (1, 7), (2, 10), (4, 14)]]
    # sbox_8bit = f(a, x)
    # sbox_8bit = apply_f_az(az, x_0)

    sbox_8bit = apply_f_n_times_roll(a)
    # sbox_8bit = apply_f_n_times(a, 2)

    if sbox_8bit is None:
        print("NOOO!!")
        sys.exit(0)

    print("a: {}".format(a))
    print("sbox_8bit:")
    pretty_block_printer(sbox_8bit, 8, 256)
    
    sbox_int = sbox_8bit.astype(np.int)

    x_0 = np.arange(0, 256).astype(np.int)

    diff_1 = sbox_int-x_0
    diff_1_abs = np.abs(diff_1)

    diff_2 = np.roll(sbox_int, 1)-sbox_int
    diff_2_abs = np.abs(diff_2)

    # print("diff_1: {}".format(diff_1))
    print("diff_1_abs: {}".format(diff_1_abs))
    # print("diff_2: {}".format(diff_2))
    print("diff_2_abs: {}".format(diff_2_abs))

    amount_diff_1_abs = np.zeros(256)
    for i in diff_1_abs:
        amount_diff_1_abs[i] += 1

    x = np.arange(0, 256)
    fig, ax = plt.subplots()

    width = 1.
    rects1 = ax.bar(x-width/2, amount_diff_1_abs, width, color="b")

    plt.title("Diff 1 abs")
    plt.show()

    amount_diff_2_abs = np.zeros(256)
    for i in diff_2_abs:
        amount_diff_2_abs[i] += 1

    x = np.arange(0, 256)
    fig, ax = plt.subplots()

    width = 1.
    rects1 = ax.bar(x-width/2, amount_diff_2_abs, width, color="b")

    plt.title("Diff 2 abs")
    plt.show()

    # sbox_8bit_swapped_4bits = sbox_8bit>>4|sbox_8bit<<4
    # print("sbox_8bit_swapped_4bits:")
    # pretty_block_printer(sbox_8bit_swapped_4bits, 8, 256)

    # arr_sorted = np.sort(sbox_8bit)
    # diff = arr_sorted[1:]-arr_sorted[:-1]
    # is_good_sbox = np.sum(diff!=1)==0
    # print("is_good_sbox: {}".format(is_good_sbox))

    # if is_good_sbox:

    # TODO: add a measurment for randomnet of a sbox!

    # cycles = get_sbox_cycles(sbox_8bit)
    # # print("cycles:\n{}".format(cycles))

    # for i, cycle in enumerate(cycles):
    #     print("i: {}, len(cycle): {}".format(i, len(cycle)))
    #     # print("cycle:\n{}".format(cycle))
    #     print("cycle:")
    #     pretty_block_printer(cycle, 8, len(cycle))
    
    sbox_inv_8bit = sbox_8bit+0
    sbox_inv_8bit[sbox_8bit] = np.arange(0, 256).astype(np.uint8)
