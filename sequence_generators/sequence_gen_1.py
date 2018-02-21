#! /usr/bin/python2.7

import os
import sys

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

mod = 256
n = 5

f = lambda c, mod: lambda params: lambda x: (np.sum(params*x)+c) % mod

# print("params: {}".format(params))
# print("c: {}".format(c))


# # test a random x
# x = np.random.randint(0, mod, (n, ))
# print("x: {}".format(x))
# y = f_1(x)
# print("y: {}".format(y))

def get_sequence_with_all_nums(mod, n, length):
    c = np.random.randint(1, mod)
    f_prepared = f(c, mod)
    for tries in xrange(0, 10000):
        params = np.random.randint(0, mod, (n, ))
        f_1 = f_prepared(params)

        nums = np.random.randint(0, mod, (n, )).tolist()
        # nums = [0]*(n-1)+[1]

        for i in xrange(0, length-n):
            nums.append(f_1(nums[-n:]))

        # check, if all numbers can be found in the sequence
        poss_nums_pos = np.zeros((mod, )).astype(np.int)
        poss_nums_pos[nums] = 1
        # print("poss_nums_pos: {}".format(poss_nums_pos))
        not_included_nums = np.arange(0, mod)[poss_nums_pos==0]
        # print("not_included_nums: {}".format(not_included_nums))

        if not_included_nums.shape[0] == 0:
            break

    print("c: {}".format(c))
    print("params: {}".format(params))

    return nums

def get_random_sequence_1d(mod):
    nums = get_sequence_with_all_nums(mod, 1, mod)
    print("nums: {}".format(nums))

def get_random_picture():
    amount = 3*mod
    length = 2*mod
    pix = np.zeros((amount, length, 3)).astype(np.uint8)

    for i in xrange(0, amount):
        print("i: {}".format(i))
        nums = get_sequence_with_all_nums(mod, n, length)
        pix[i] = np.array(nums).reshape((-1, 1))*4

    img = Image.fromarray(pix)
    img.show()

# print("mod: {}".format(mod))
# print("n: {}".format(n))
# print("nums: {}".format(nums))

# print("len(nums: {}".format(len(nums)))

get_random_sequence_1d(10)
