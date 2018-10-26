#! /usr/bin/python3.5

# -*- coding: utf-8 -*-

import operator

import numpy as np

from copy import deepcopy
from math import gcd, factorial as fac

import sys
sys.path.append("../encryption")

import Utils

def get_new_1_loop_sbox(n, xs=None, unique=False):
    assert isinstance(xs, list) or isinstance(xs, type(None))

    if xs == None:
        xs = np.random.permutation(np.arange(0, n)).tolist()
    x = xs[0]

    f_x = np.zeros((n, ), dtype=np.int)
    while len(xs) > 0:
        if xs[0] == x:
            xs = xs[1:]+xs[:1]
        x_new = xs.pop(0)
        f_x[x_new] = x
        x = x_new

    if unique == True:
        idx = np.where(f_x==0)[0][0]
        if idx > 1:
            return np.hstack((f_x[idx-1:], f_x[:idx-1])).astype(np.uint8)
    
    return f_x.astype(np.uint8)

def move_to_2nd_pos(f_x):
    idx = np.where(f_x==0)[0][0]
    if idx == 1:
        return f_x
    return np.hstack((f_x[idx-1:], f_x[:idx-1]))

def get_linear_mixed_array(n, a_0, a_1, find_next_a_1=False):
    # assert gcd(n, a_1) == 1
    if find_next_a_1:
        while gcd(n, a_1) != 1:
            a_1 += 1
        # print("a_1: {}".format(a_1))
    return ((a_0+np.arange(0, n)*a_1) % n).astype(np.uint8)

# returns a 4 byte value e.g.
def prng(seed=None):
    if seed == None:
        seed = np.random.randint(0, 2**32)

    def get_lines_mixed(seed):
        max_bits = 32
        max_num = 2**max_bits
        mask = max_num-1

        while seed > max_num:
            seed = (seed>>max_bits)|(seed&mask)

        seed = (23+11*seed) % max_num
        a_1_3 = seed % 256
        for _ in range(0, 2):
            seed = (23+11*seed) % max_num
            a_0_0 = (seed+3*a_1_3) % 256
            seed = (24+11*seed) % max_num
            a_0_1 = (seed+5*a_0_0) % 256
            if a_0_1 == 0:
                a_0_1 = 1

            seed = (23+11*seed) % max_num
            a_1_0 = (seed+7*a_0_1) % 256
            seed = (23+11*seed) % max_num
            a_1_1 = (seed+9*a_1_0) % 256
            if a_1_1 == 0:
                a_1_1 = 1

            seed = (23+11*seed) % max_num
            a_0_2 = (seed+11*a_1_1) % 256
            seed = (23+11*seed) % max_num
            a_0_3 = (seed+13*a_0_2) % 256
            if a_0_3 == 0:
                a_0_3 = 1

            seed = (23+11*seed) % max_num
            a_1_2 = (seed+17*a_0_3) % 256
            seed = (23+11*seed) % max_num
            a_1_3 = (seed+19*a_1_2) % 256
            if a_1_3 == 0:
                a_1_3 = 1

        xs0 = get_linear_mixed_array(256, a_0_0, a_1_0, find_next_a_1=True)
        xs1 = get_linear_mixed_array(256, a_0_1, a_1_1, find_next_a_1=True)
        xs2 = get_linear_mixed_array(256, a_0_2, a_1_2, find_next_a_1=True)
        xs3 = get_linear_mixed_array(256, a_0_3, a_1_3, find_next_a_1=True)

        seed = (26+13*seed) % max_num
        idx = seed % 256
        # seed = (26+13*seed) % max_num
        for i in range(0, 256):
            i1, i2= (idx+i)%256, (idx+3*i+1)%256
            if i1 == i2:
                i2 += 1
            xs0[i1], xs0[i2] = xs0[i2], xs0[i1]
            xs1[i1], xs1[i2] = xs1[i2], xs1[i1]
            xs2[i1], xs2[i2] = xs2[i2], xs2[i1]
            xs3[i1], xs3[i2] = xs3[i2], xs3[i1]

        return seed, xs0, xs1, xs2, xs3

    # seed, xs0, xs1, xs2, xs3 = get_lines_mixed(seed)

    # xsl0 = get_new_1_loop_sbox(256, xs0.tolist())
    # xsl1 = get_new_1_loop_sbox(256, xs1.tolist())
    # xsl2 = get_new_1_loop_sbox(256, xs2.tolist())
    # xsl3 = get_new_1_loop_sbox(256, xs3.tolist())

    # nxs0 = xs3[xs2[xs0]]
    # nxs1 = xs2[xs1[xs3]]
    # nxs2 = xs0[xs3[xs1]]
    # nxs3 = xs3[xs1[xs2]]
        
    # xs0 = nxs0
    # xs1 = nxs1
    # xs2 = nxs2
    # xs3 = nxs3

    # x0 = xs0[0]
    # x1 = xs0[1]
    # x2 = xs1[0]
    # x3 = xs1[1]
    # x4 = xs2[0]
    # x5 = xs2[1]
    # x6 = xs3[0]
    # x7 = xs3[1]
    while True:
        print("old seed: {}".format(seed))
        seed, xs0, xs1, xs2, xs3 = get_lines_mixed(seed+1)
        print("new seed: {}".format(seed))

        globals()["xs0"] = xs0
        globals()["xs1"] = xs1
        globals()["xs2"] = xs2
        globals()["xs3"] = xs3
        
        xsl0 = get_new_1_loop_sbox(256, xs0.tolist())
        xsl1 = get_new_1_loop_sbox(256, xs1.tolist())
        xsl2 = get_new_1_loop_sbox(256, xs2.tolist())
        xsl3 = get_new_1_loop_sbox(256, xs3.tolist())

        # xsl0 = xs0
        # xsl1 = xs1
        # xsl2 = xs2
        # xsl3 = xs3

        # globals()["xsl0"] = xsl0
        # globals()["xsl1"] = xsl1
        # globals()["xsl2"] = xsl2
        # globals()["xsl3"] = xsl3
        # sys.exit(-2)

        # xsl0_0 = xsl0[0]
        # xsl1_0 = xsl1[0]
        # xsl0_copy = xsl0.copy()
        # xsl1_copy = xsl1.copy()

        seed = (13+seed*17)%0x10000
        num1 = (xsl0[0]<<16|xsl1[0])^seed
        seed = (13+seed*17)%0x10000
        num2 = (xsl2[0]<<16|xsl3[0])^seed

        num1_orig = num1
        num2_orig = num2
        num1 = (13+num1*19)%0x100000000
        num2 = (43+num2*23)%0x100000000    
        while num1_orig != num1 or num2_orig != num2:
            num1 = (13+num1*19)%0x100000000
            num2 = (43+num2*23)%0x100000000
            yield (num1^num2)&0xFFFF

        # for i0 in range(0, 256):
        #     x0 = int(xsl0[i0])
        #     x2 = int(xsl2[i0])
        #     for i1 in range(0, 256):
        #         x1 = int(xsl1[i1])
        #         x3 = int(xsl3[i1])
        #         num = (x2<<16)|(x0<<8)|(x1)
        #         # num = (x3<<24)|(x2<<16)|(x0<<8)|(x1)
                
        #         # num1 = (x3<<24)|(x2<<16)|(x0<<8)|(x1)
        #         # num2 = (x3<<24)|(x2<<16)|(x0<<8)|(x1)
                
        #         # shift = i1%16
        #         # num_shift = ((num>>shift)&(2**(16-shift)-1))|(num&((2**(16-shift)-1)<<shift))
                
        #         # print("shift: {}, num: 0x{:02X}, bin: {}, num_shift: 0x{:02X}, bin: {}".format(shift, num, bin(num), num_shift, bin(num_shift)))
        #         # yield num_shift
        #         yield num
            
            # sys.exit(-2)
            # x2 = xsl2[i]
            # x3 = xsl3[i]
                
            # yield (x0<<8)|(x1)
            # yield (x2<<16)|(x0<<8)|(x1)
            
            # yield (x3<<8)|(x2)
            # yield (x2<<8)|(x0)
            # yield (x0<<8)|(x3)

        # for j in range(0, 3):
        # # while True:
        #     for i in range(0, 256):
        #     # for i in range(0, 256):
        #         x0 = xsl0[i]
        #         x1 = xsl1[i]
                
        #         yield (x0<<8)|(x1)

        #     nxs0 = xsl3[xsl2[xsl0^0x34]]+1
        #     nxs1 = xsl2[xsl1[xsl3]^0x45]+2
        #     nxs2 = xsl0[xsl3[xsl1]]^0x8A+3
        #     nxs3 = xsl3[xsl1[xsl2^0xE2]^0x19]^0x23+4

        #     xsl0 = nxs0
        #     xsl1 = nxs1
        #     xsl2 = nxs2
        #     xsl3 = nxs3


def test_prng(seed=None):
    # if seed == None:
    #     seed = 123
    my_prng = prng(seed=seed)

    # print("seed: {}".format(seed))
    nums = []
    # first_num = next(my_prng)
    for i in range(0, 10000000):
        num = next(my_prng)
        print("  i: {}, next(my_prng): 0x{:08X}".format(i, num))
        if num in nums:
        # if first_num == num:
            print("EQUAL TO ONE PREVIOUS GENERATED NUMBER!!!")
            nums.append(num)
            break
        nums.append(num)
    print("nums.index(num): {}".format(nums.index(num)))
    num_repeats = len(nums)-nums.index(num)
    print("length of repeat: {}".format(num_repeats))
    
    return num_repeats

if __name__ == "__main__":
    # print("Hello World!")
    n = 6

    # # build a 1 loop sbox!
    # sbox_1_loop = get_new_1_loop_sbox(n, unique=True)
    # print("sbox_1_loop: {}".format(sbox_1_loop))   

    num_repeats_lst = []
    for i in range(0, 1):
        num_repeats = test_prng()
        num_repeats_lst.append(num_repeats)

    print("np.mean(num_repeats_lst): {}".format(np.mean(num_repeats_lst)))

    unique_1_loops = []

    for i in range(0, 5000):
        sbox_1_loop = tuple(get_new_1_loop_sbox(n, unique=False))
        # sbox_1_loop = tuple(get_new_1_loop_sbox(n, unique=True))
        if not sbox_1_loop in unique_1_loops:
            unique_1_loops.append(sbox_1_loop)

    print("len(unique_1_loops): {}".format(len(unique_1_loops)))
