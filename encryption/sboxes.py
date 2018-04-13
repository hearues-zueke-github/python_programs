#! /usr/bin/python2.7
# -*- coding: utf-8 -*- 

import os
import sys

import numpy as np

import matplotlib.pyplot as plt

from Utils import pretty_block_printer, clrs

def invert_bits_func(bits):
    assert bits%2 == 0
    
    arr = np.arange(0, 2**bits).astype(np.uint8)

    return (~arr)&((1<<bits)-1)

def invert_pos_func(bits):
    assert bits%2 == 0
    
    arr = np.arange(0, 2**bits).astype(np.uint8)
    f_table = np.zeros(arr.shape).astype(np.uint8)
    
    for i in xrange(0, bits):
        shift_arr = ((arr<<(bits-1-i*2) if i < bits//2 else arr>>(1+i*2-bits))&(1<<(bits-1-i)))
        f_table |= shift_arr

    return f_table

def swap_pos_func(bits):
    assert bits%2 == 0
    half_bits = bits//2
    arr = np.arange(0, 2**bits).astype(np.uint8)
    bit_mask_r = (1<<half_bits)-1
    bit_mask_l = bit_mask_r<<half_bits

    return ((arr>>(half_bits))&bit_mask_r)|\
           ((arr<<(half_bits))&bit_mask_l)

def roll_shift_func(bits):
    assert bits%2 == 0
    arr = np.arange(0, 2**bits).astype(np.uint8)

    return ((arr>>1)|((arr&1)<<(bits-1)))[np.roll(arr, len(arr)-1)]
    # return ((arr>>1)|((arr&1)<<(bits-1)))^\
    # return arr|\
           # np.roll(arr, len(arr)-1)

def get_basic_sboxes(bits):
    assert bits%2 == 0

    sbox_1 = invert_bits_func(bits)
    sbox_2 = invert_pos_func(bits)
    sbox_3 = swap_pos_func(bits)
    sbox_4 = roll_shift_func(bits)

    return [sbox_1, sbox_2, sbox_3, sbox_4]

if __name__ == "__main__":
    print("different simple sbox functions")

    bits = 8

    ib_table = invert_bits_func(bits)
    print("{}ib_table:{}".format(clrs.lgb, clrs.rst))
    pretty_block_printer(ib_table, 8, len(ib_table))

    ip_table = invert_pos_func(bits)
    print("{}ip_table:{}".format(clrs.lgb, clrs.rst))
    pretty_block_printer(ip_table, 8, len(ip_table))

    sp_table = swap_pos_func(bits)
    print("{}sp_table:{}".format(clrs.lgb, clrs.rst))
    pretty_block_printer(sp_table, 8, len(sp_table))

    rs_table = roll_shift_func(bits)
    print("{}rs_table:{}".format(clrs.lgb, clrs.rst))
    pretty_block_printer(rs_table, 8, len(rs_table))

    # rx_table_sort = np.sort(rs_table)
    # diff = rx_table_sort[1:]-rx_table_sort[:-1]
    # is_table_good = np.sum(diff!=1)==0
    # print("is_table_good: {}".format(is_table_good))
