#! /usr/bin/python3.5

# -*- coding: utf-8 -*-

import operator

import numpy as np

from copy import deepcopy
from datetime import datetime
from math import gcd, factorial as fac

import sys
sys.path.append("../encryption")

import Utils

def prng(seed=None):
    # # True random!
    # while True:
    #     yield np.random.randint(0, 1<<16)

    max_bits = 32
    max_num = 1<<max_bits
    half_num = 1<<(max_bits//2)
    mask = max_num-1

    if seed == None:
        seed = int((datetime.now()-datetime.utcfromtimestamp(0)).total_seconds()*1000000)
    else:
        while seed < half_num:
            seed = (17+seed*31)%max_num

    while seed >= max_num:
        seed = (seed>>max_bits)^(seed&mask)
    start_num = seed%2
    print("seed: {}".format(seed))
    while True:
        seed1 = (21+23*seed)%max_num
        seed2 = (29+27*seed)%max_num
        seed3 = (31+29*seed)%max_num

        loops = (seed1+seed2+seed3)%10+1
        print("loops: {}".format(loops))
        for i in range(0, loops):
        # while not(seed1_orig == seed1 or seed2_orig == seed2):
            seed1 = (13+seed1*23)%max_num
            seed2 = (43+seed2*23)%max_num
            seed3 = (53+seed3*13)%max_num
            # yield (seed1^seed2+seed3^start_num^i)&0xFFFF
            yield (seed1^seed2+seed3)&0xFFFF

        start_num = (start_num+1)%2

        seed = (17+seed*31)%max_num
        while seed < half_num:
            seed = (19+seed*37)%max_num

def quality_of_prng(seed=None):
    rng = prng(seed=seed)

    n = 1024
    arr = np.array([next(rng)&0xFFFF for _ in zip(range(0, n))])

    bits_changes = [(arr>>i)&0x1 for i in range(0, 16)]

    get_diffs_amount = lambda bc: np.unique((lambda x: x[1:]-x[:-1])(np.hstack(((0, ), np.where(bc[1:]!=bc[:-1])[0]+1))), return_counts=True)
    diffs_amount = [get_diffs_amount(bc) for bc in bits_changes]
    expected_mean = [np.sum(diff_a[0]*diff_a[1])/ np.sum(diff_a[1]) for diff_a in diffs_amount]
    # print("arr:\n{}".format(arr))
    # print("bits_changes:\n{}".format(bits_changes))
    for bit, diff_a in enumerate(diffs_amount):
        print("bit: {}, diff_a: {}".format(bit, diff_a))
    for bit, expect_mean in enumerate(expected_mean):
        print("bit: {}, expect_mean: {}".format(bit, expect_mean))

    mse_bits_randomness = np.sum((np.array(expected_mean)-2)**2)/len(expected_mean)
    print("mse_bits_randomness: {}".format(mse_bits_randomness))
    # globals()["bits_changes"] = bits_changes
    # return arr

if __name__ == "__main__":
    # print("Hello World!")
    seed = 345
        
    # quality_of_prng()
    quality_of_prng(seed=0)
