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
    # print("seed: {}".format(seed))
    if seed == None:
        seed = np.random.randint(0, 2**32)
        # a_0_0 = np.random.randint(0, 256)
        # a_0_1 = np.random.randint(1, 256)
        # a_1_0 = np.random.randint(0, 256)
        # a_1_1 = np.random.randint(1, 256)
        # a_0_2 = np.random.randint(0, 256)
        # a_0_3 = np.random.randint(1, 256)
        # a_1_2 = np.random.randint(0, 256)
        # a_1_3 = np.random.randint(1, 256)
    # else:
    def get_lines_mixed(seed):
        max_bits = 16
        max_num = 2**max_bits
        mask = max_num-1

        # print("seed before: 0x{:04X}".format(seed))
        while seed > max_num:
            seed = (seed>>max_bits)|(seed&mask)
        # print("seed after:  0x{:04X}".format(seed))

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

        # print("a_0_0: 0x{:02X}, a_1_0: 0x{:02X}".format(a_0_0, a_1_0))
        # print("a_0_1: 0x{:02X}, a_1_1: 0x{:02X}".format(a_0_1, a_1_1))
        # print("a_0_2: 0x{:02X}, a_1_2: 0x{:02X}".format(a_0_2, a_1_2))
        # print("a_0_3: 0x{:02X}, a_1_3: 0x{:02X}".format(a_0_3, a_1_3))

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

        for j in range(0, 3):
        # while True:
            for i in range(0, 256):
            # for i in range(0, 256):
                x0 = xsl0[i]
                x1 = xsl1[i]
                # x2 = xsl2[i]
                # x3 = xsl3[i]
                
                # print("x: {}".format((x0, x1)))
                
                # print("x: {}".format((x0, x1, x2, x3)))
                # print("x: {}".format((x0, x1, x2, x3, x4, x5, x6, x7)))
                # yield (x0<<24)|(x1<<16)|(x2<<8)|(x3)
                # yield ((x0<<8)&0x0F)|(x1)
                yield (x0<<8)|(x1)

            # nxs0 = xsl3[xsl2[xsl0]]
            # nxs1 = xsl2[xsl1[xsl3]]
            
            nxs0 = xsl3[xsl2[xsl0^0x34]]+1
            nxs1 = xsl2[xsl1[xsl3]^0x45]+2
            nxs2 = xsl0[xsl3[xsl1]]^0x8A+3
            nxs3 = xsl3[xsl1[xsl2^0xE2]^0x19]^0x23+4

            xsl0 = nxs0
            xsl1 = nxs1
            xsl2 = nxs2
            xsl3 = nxs3

            # if np.sum(xsl0_copy!=xsl0)==0 and np.sum(xsl1_copy!=xsl1)==0:
            # if xsl0_0 == xsl0[0] and xsl1_0 == xsl1[0]:
            #     print("BREAK!!!")
            #     break
        
        # xs0 = nxs0
        # xs1 = nxs1
        # xs2 = nxs2
        # xs3 = nxs3
        
        # xn0 = (xs0[x2]^xs1[x1]^xs2[x5]+0x23) % 256
        # xn1 = (xs0[x3]^xs1[x2]^xs2[x4]+0x21) % 256
        # xn2 = (xs0[x1]^xs1[x4]^xs2[x5]+0x23) % 256
        # xn3 = (xs0[x7]^xs1[x3]^xs2[x1]+0x22) % 256
        # xn4 = (xs0[x3]^xs1[x1]^xs2[x4]+0x21) % 256
        # xn5 = (xs0[x2]^xs1[x3]^xs2[x6]+0x52) % 256
        # xn6 = (xs0[x1]^xs1[x2]^xs2[x5]+0x2F) % 256
        # xn7 = (xs0[x7]^xs1[x6]^xs2[x4]+0x53) % 256

        # x0, x1, x2, x3, x4, x5, x6, x7 = xn0, xn1, xn2, xn3, xn4, xn5, xn6, xn7
        # # yield (x2<<8)|(x3)
        # # yield (x6<<48)|(x5<<40)|(x4<<32)|(x3<<24)|(x2<<16)|(x1<<8)|(x0)
        # print("x: {}".format((x0, x1, x2, x3, x4, x5, x6, x7)))
        # # yield int((x6<<56)|(x7<<48)|(x5<<40)|(x4<<32)|(x3<<24)|(x2<<16)|(x1<<8)|(x0))
        # yield (x0<<24)|(x1<<16)|(x2<<8)|(x3)

def test_prng(seed=None):
    # if seed == None:
    #     seed = 123
    my_prng = prng(seed=seed)

    # print("seed: {}".format(seed))
    nums = []
    # first_num = next(my_prng)
    for i in range(0, 100000):
        num = next(my_prng)
        # print("  i: {}, next(my_prng): 0x{:08X}".format(i, num))
        if num in nums:
        # if first_num == num:
            print("EQUAL TO ONE PREVIOUS GENERATED NUMBER!!!")
            nums.append(num)
            break
        nums.append(num)
    print("nums.index(num): {}".format(nums.index(num)))
    print("length of repeat: {}".format(len(nums)-nums.index(num)))

if __name__ == "__main__":
    # print("Hello World!")
    n = 6

    # # build a 1 loop sbox!
    # sbox_1_loop = get_new_1_loop_sbox(n, unique=True)
    # print("sbox_1_loop: {}".format(sbox_1_loop))   

    test_prng()

    unique_1_loops = []

    for i in range(0, 5000):
        sbox_1_loop = tuple(get_new_1_loop_sbox(n, unique=False))
        # sbox_1_loop = tuple(get_new_1_loop_sbox(n, unique=True))
        if not sbox_1_loop in unique_1_loops:
            unique_1_loops.append(sbox_1_loop)

    print("len(unique_1_loops): {}".format(len(unique_1_loops)))
