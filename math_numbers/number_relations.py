#! /usr/bin/python3.5

import decimal
import math

import numpy as np

import matplotlib.pyplot as plt

from decimal import Decimal as D
from functools import reduce

decimal.getcontext().prec = 2000

def to_digit_list(n):
    return list(map(int, list(str(n))))

''' Number manipulation functions '''
def digit_sum(n):
    return np.sum(to_digit_list(n))

def digit_diff(n):
    l = list(map(int, list(str(n))))
    return int("".join(list(map(lambda a, b: str((a-b)%10), l[1:], l[:-1]))))

def digit_prod(n):
    l = np.array(to_digit_list(n))
    l[l==0] = 1
    return np.prod(l)

a = 1234568
print("a: {}".format(a))
print("digit_sum(a): {}".format(digit_sum(a)))
print("digit_diff(a): {}".format(digit_diff(a)))
print("digit_prod(a): {}".format(digit_prod(a)))

n_max = 100000
l_dig_sums = [digit_sum(i) for i in range(0, n_max+1)]
l_dig_diffs = [0 for _ in range(0, 10)]+[digit_diff(i) for i in range(10, n_max+1)]
l_dig_prods = [digit_prod(i) for i in range(0, n_max+1)]

print("np.min(l_dig_sums): {}".format(np.min(l_dig_sums)))
print("np.min(l_dig_diffs): {}".format(np.min(l_dig_diffs)))
print("np.min(l_dig_prods): {}".format(np.min(l_dig_prods)))

print("np.max(l_dig_sums): {}".format(np.max(l_dig_sums)))
print("np.max(l_dig_diffs): {}".format(np.max(l_dig_diffs)))
print("np.max(l_dig_prods): {}".format(np.max(l_dig_prods)))

ls = []
for n in range(1, 1000):
    l = [n]
    j = n
    for i in range(0, 10):
        j = l_dig_sums[j]
        l.append(j)
        # j = l_dig_diffs[j]
        # l.append(j)
        j = l_dig_prods[j]
        l.append(j)
    ls.append((n, l))

for n, l in ls:
    print("n: {}, l:\n{}".format(n, l))
