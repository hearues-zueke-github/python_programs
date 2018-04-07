#! /usr/bin/python2.7

import os
import sys

import numpy as np

from Utils import pretty_block_printer

np.set_printoptions(formatter={'int': lambda x: "{:02X}".format(x)})

def f(a, x):
    print("a: {}".format(a))
    s = np.zeros(x.shape, dtype=np.uint8)

    for i, c in a:
        s += (c*x**i) % 16
        s %= 16
    
    return s

def get_sbox_cycles(s_4b):
    cycles = []
    poss_idx = np.arange(0, 16).tolist()

    while len(poss_idx) > 0:
        idx_start = poss_idx.pop(0)
        idx = s_4b[idx_start]
        cycle = [idx_start]

        while idx != idx_start:
            poss_idx.remove(idx)
            idx_prev = idx
            idx = s_4b[idx]
            cycle.append(idx_prev)

        cycles.append(np.array(cycle))

    return cycles

if __name__ == "__main__":
    # acceptable_constants = find_acceptable_constants()
    # for i, accept_consts in enumerate(acceptable_constants):
    #     print("i: {}, accept_consts: {}".format(i, accept_consts))

    # sys.exit(0)

    # a = [5, 5, 4, 2, 2, 6, 8, 6]
    a = [(0, 5), (1, 5)]
    sbox_8bit = f(a, np.arange(0, 16).astype(np.uint8))

    print("sbox_8bit:")
    pretty_block_printer(sbox_8bit, 8, 16, 4)

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
            pretty_block_printer(cycle, 8, len(cycle), 4)
    
    sbox_inv_8bit = sbox_8bit+0
    sbox_inv_8bit[sbox_8bit] = np.arange(0, 16).astype(np.uint8)
