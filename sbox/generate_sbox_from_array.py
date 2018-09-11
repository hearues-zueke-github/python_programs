#! /usr/bin/python3.5

# -*- coding: utf-8 -*-

import numpy as np

import sys
sys.path.append("../encryption")

import Utils

def convert_to_4_bit_code(arr_8_bits):
    return np.vstack((arr_8_bits>>4, arr_8_bits&0xF)).T.reshape((-1, ))

def convert_to_8_bit_code(arr_4_bits):
    a, b = arr_4_bits.reshape((2, -1))
    return (a<<4) ^ b
    # return (lambda a, b: (a<<4) & b)((lambda x: x[0], x[1])(arr_4_bits.reshape((2, -1))))

def apply_4_bit_code(arr_4_bits):
    len_arr = len(arr_4_bits)

    arr_2 = np.zeros((512, ), dtype=np.uint8)

    idx = 0
    for i in range(0, len_arr//2):
        rang, val = arr_4_bits[2*i:2*(i+1)]

        for _ in range(0, rang+1):
            arr_2[idx] ^= val
            idx = (idx+1) % 512
    
    return arr_2

if __name__ == "__main__":
    len_1 = 1024
    arr = np.random.randint(0, 256, (len_1, ), dtype=np.uint8)
    Utils.pretty_block_printer(arr, 8, len_1)

    arr_4_bits = convert_to_4_bit_code(arr)
    Utils.pretty_block_printer(arr_4_bits, 4, len(arr_4_bits))

    for i in range(0, 4):
        print("roll: i: {}".format(i))
        arr_2 = apply_4_bit_code(np.roll(arr_4_bits, i))
        arr_8_bits = convert_to_8_bit_code(arr_2)

        print("arr_8_bits")
        Utils.pretty_block_printer(arr_8_bits, 8, len(arr_8_bits))
