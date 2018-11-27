#! /usr/bin/python3.5

import decimal
import math

from decimal import Decimal as Dec

import matplotlib.pyplot as plt

decimal.getcontext().prec = 1000

# s = Dec("3.")
dec_1 = Dec("1.")
# x_prev = x
# dec_2 = Dec("2.")

str_d1 = "0."

n = 1000

for i in range(1, n+1):
    print("i: {}".format(i))
    str_d1 += str(i)
    d1 = Dec(str_d1)
    # print("  d1: {}".format(d1))

    D = dec_1/d1
    # print("  D: {}".format(D))

    d2 = dec_1/D
    # print("  d2: {}".format(d2))

    # diff = d1-d2
    # print("diff: {}".format(diff))

    d2_str = str(d2)
    d2_str = d2_str[2:]
    # print("d2_str: {}".format(d2_str))

    for j in range(1, n+1):
        j_str = str(j)
        l = len(j_str)

        part_str = d2_str[:l]

        if part_str != j_str:
            j -= 1
            break
        d2_str = d2_str[l:]
    print("good_length: {}".format(j))
