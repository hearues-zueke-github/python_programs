#! /usr/bin/python2.7

import os
import sys

import numpy as np

from Utils import pretty_block_printer

np.set_printoptions(formatter={'int': lambda x: "{:02X}".format(x)})

def f(a, x):
    print("a: {}".format(a))
    s = np.zeros(x.shape, dtype=np.uint8)

    for i, c in enumerate(a):
        s += c*x**i
    
    return s

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

def get_sbox_cycles(s_8b):
    cycles = []
    poss_idx = np.arange(0, 256).tolist()

    while len(poss_idx) > 0:
        idx_start = poss_idx.pop(0)
        idx = s_8b[idx_start]
        cycle = [idx_start]

        while idx != idx_start:
            poss_idx.remove(idx)
            idx_prev = idx
            idx = s_8b[idx]
            cycle.append(idx_prev)

        cycles.append(np.array(cycle))

    return cycles

if __name__ == "__main__":
    # acceptable_constants = find_acceptable_constants()
    # for i, accept_consts in enumerate(acceptable_constants):
    #     print("i: {}, accept_consts: {}".format(i, accept_consts))

    # sys.exit(0)

    sbox_8bit = f([5, 5, 4, 2, 2, 6, 8, 6], np.arange(0, 256).astype(np.uint8))

    print("sbox_8bit:")
    pretty_block_printer(sbox_8bit, 8, 256)

    arr_sorted = np.sort(sbox_8bit)
    diff = arr_sorted[1:]-arr_sorted[:-1]
    is_good_sbox = np.sum(diff!=1)==0
    print("is_good_sbox: {}".format(is_good_sbox))

    if is_good_sbox:
        cycles = get_sbox_cycles(sbox_8bit)
        # print("cycles:\n{}".format(cycles))

        for i, cycle in enumerate(cycles):
            print("i: {}, len(cycle): {}".format(i, len(cycle)))
            # print("cycle:\n{}".format(cycle))
            print("cycle:")
            pretty_block_printer(cycle, 8, len(cycle))
    
    sbox_inv_8bit = sbox_8bit+0
    sbox_inv_8bit[sbox_8bit] = np.arange(0, 256).astype(np.uint8)
