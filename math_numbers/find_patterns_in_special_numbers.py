#! /usr/bin/python3.5

import decimal
import math

import numpy as np

import matplotlib.pyplot as plt

from decimal import Decimal as D
from functools import reduce

decimal.getcontext().prec = 2000

# TODO: maybe interesting for Numberphile?!

def remove_leading_zeros(l):
    l = np.array(l).copy()
    i = 0
    while l[i] == 0:
        i += 1
    l = l[i:]
    return l

def find_pattern(l):
    l = remove_leading_zeros(l)

    length = l.shape[0]

    max_idx = length//2
    n = l[0]
    start_idx = 1
    while n == l[start_idx]:
        start_idx += 1
        if start_idx >= max_idx:
            return np.array([n])

    for i in range(start_idx, max_idx):
        if np.sum(l[:i] != l[i:i*2]) == 0:
            return l[:i]

    return np.array([])

if __name__ == "__main__":
    # print("Hello World!")
    print("Calculating numbers")
    num = 9
    p = 1
    nums = []
    max_power = 7000
    for i in range(1000, max_power):
        p *= num
        nums.append(list(map(int, str(p))))

    print("Getting them into the matrix")
    matrix = np.zeros((len(nums), len(nums[-1])), dtype=np.int)
    for i in range(0, len(nums)):
        l = nums[i]
        matrix[i, :len(l)] = l[::-1]

    print("Finding the pattern")
    patterns = [(lambda x: (i, len(x), x))(find_pattern(l)) for i, l in enumerate(matrix.T[:int(np.sqrt(max_power))])]
