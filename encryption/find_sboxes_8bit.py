#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

import matplotlib.pyplot as plt

from Utils import pretty_block_printer, clrs
from sboxes import get_basic_sboxes 

from os.path import expanduser
np.set_printoptions(formatter={'int': lambda x: "0x{:02X}".format(x)})

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

def apply_f_n_times_roll(a, sboxes):
    x_0 = np.arange(0, 256).astype(np.uint8)

    x_prev = x_0+0
    x_next = x_0+0

    for i in a:
        x_new = x_prev[np.roll(x_next, i)]
        x_prev = x_next
        for sbox in sboxes:
            x_next = sbox[x_next]
        # x_next = swap_4bits(x_new)
        x_next += i

    if check_if_sbox_good(x_next) == False:
        return None

    return x_next

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
    # a = [2, 10, 12, 13, 17, 33, 22, 6, 3, 8, 7, 4, 11, 64, 7, 5, 3, 8,
    #      10, 12, 13, 17, 33, 22, 6, 3, 8, 7, 4, 11, 64, 7, 5, 3, 8, 7, 4, 11, 64, 7, 5, 3]
    # a = [3, 5, 6, 2, 4, 7, 5, 10, 5, 6, 23, 44, 23, 14]
    a = np.random.randint(1, 256, (128, ))

    different_sboxes = get_basic_sboxes(8)

    sbox_8bit = apply_f_n_times_roll(a, different_sboxes)
    # sbox_8bit = apply_f_n_times(a, 2)

    if sbox_8bit is None:
        print("NOOO!!")
        sys.exit(0)

    print("{}a:{}".format(clrs.lgb, clrs.rst))
    pretty_block_printer(a, 8, a.shape[0])
    print("{}sbox_8bit:{}".format(clrs.lgb, clrs.rst))
    pretty_block_printer(sbox_8bit, 8, 256)
    
    sbox_int = sbox_8bit.astype(np.int)

    x_0 = np.arange(0, 256).astype(np.int)

    diff_1 = sbox_int-x_0
    diff_1_abs = np.abs(diff_1)

    diff_2 = np.roll(sbox_int, 1)-sbox_int
    diff_2_abs = np.abs(diff_2)

    sbox1, sbox2, sbox3, sbox4 = list(map(lambda x: x.astype(np.int), different_sboxes))

    diff_3 = sbox_int-sbox1
    diff_3_abs = np.abs(diff_3)
    diff_4 = sbox_int-sbox2
    diff_4_abs = np.abs(diff_4)
    diff_5 = sbox_int-sbox3
    diff_5_abs = np.abs(diff_5)
    diff_6 = sbox_int-sbox4
    diff_6_abs = np.abs(diff_6)

    def get_amount_diff_abs(diff):
        amount_diff_abs = np.zeros(256)
        for i in diff:
            amount_diff_abs[i] += 1

        return amount_diff_abs

    def plot_amount_diff(amount_diffs, title):
        x = np.arange(0, 256)
        fig, axarr = plt.subplots(len(amount_diffs), sharex=True, figsize=(6, 18))
        
        rects = []
        width = 1.

        for i in xrange(0, len(amount_diffs)):
            rects.append(axarr[i].bar(x-width/2, amount_diffs[i], width, color="b"))

            axarr[i].set_title(title.format(i+1))

        plt.subplots_adjust(left=None, bottom=0.05,
                            right=None, top=0.95,
                            wspace=None, hspace=0.4)

    def plot_amount_sum(diff_sum, title):
        x = np.arange(0, 256)
        fig, ax = plt.subplots(figsize=(6, 3))
        
        width = 1.
        rect = ax.bar(x-width/2, diff_sum, width, color="b")

        ax.set_title(title)

        plt.subplots_adjust(left=None, bottom=0.1,
                            right=None, top=0.9,
                            wspace=None, hspace=0.4)

    diffs_abs = [diff_1_abs,
                 diff_2_abs,
                 diff_3_abs,
                 diff_4_abs,
                 diff_5_abs,
                 diff_6_abs]

    amount_diffs_abs = list(map(get_amount_diff_abs, diffs_abs))

    home_path = expanduser("~")
    full_folder_path = home_path+"/Pictures/encryption_sbox_testing"
    if not os.path.exists(full_folder_path):
        os.makedirs(full_folder_path)

    plot_amount_diff(amount_diffs_abs, "Diff nr. {} abs")
    print("Saving plot of diffs amount")
    plt.savefig(full_folder_path+"/diff_amount_plot.png")
    
    diff_sum = np.sum(amount_diffs_abs, axis=0)
    plot_amount_sum(diff_sum, "Sum of diffs abs")
    print("Saving plot of diffs sum")
    plt.savefig(full_folder_path+"/diff_amount_sum_plot.png")
