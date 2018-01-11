#! /usr/bin/python3

import math

import numpy as np

import matplotlib.pyplot as plt

from functools import reduce

# approx numerically following definiton:
# given numbers 1,2,3. How big is the probability that a sequence of
# length l wiht contains all numbers min 1 times?
# e.g.:
"""
    1123
    1213
    2113
    1231
    1321
    3121
    etc.
    are the possible combinations, where every number shows min 1 time
"""
# P(all number times >= 1) = #found combinations / #all combinations (which is m**l)

def calc_propability_all_element_more_1_time_numerically(m, l):
    n = 100000
    arr = np.random.randint(0, m, (n, l))

    print("arr:\n{}".format(arr))
    sums = 0
    for i in range(0, m):
        amount = np.sum(arr==i, axis=1)>0
        sums += amount

    sums_all = (sums==m)

    total_amount = np.sum(sums_all)
    probability = total_amount/n

    print("total_amount: {}".format(total_amount))
    print("P(all numbers times >= 1) = {} / {} = {:.5f}".format(total_amount, n, probability))
    return probability

def get_all_valid_m_l_sums(m, l):
    def valid_m_l_sum(array, i):
        if not array[0] > array[1]+1:
            return
        if array[0] > array[1]+1 and array[i-1] > array[i]:
            new_array = list(array)
            new_array[0] -= 1
            new_array[i] += 1
            all_valid_m_l_sums.append(new_array)
            valid_m_l_sum(new_array, i)
        if i+1 < m and array[i] > array[i+1]:
            valid_m_l_sum(array, i+1)
    array = [1 for _ in range(0, m)]
    array[0] = l-m+1
    all_valid_m_l_sums = [array]
    valid_m_l_sum(array, 1)
    return all_valid_m_l_sums

def calc_propability_all_element_more_1_time(m, l):
    all_valid_m_l_sums = get_all_valid_m_l_sums(m, l)
    fac = math.factorial
    def get_binomials_multiplier(v):
        a = list(set(v))
        p = fac(m)
        for i in a:
            p //= fac(v.count(i))
        return p
    def get_binomials(v):
        p = fac(l)
        for i in v:
            p //= fac(i)
        return p
    coefficients = list(map(lambda v: get_binomials_multiplier(v)*get_binomials(v), all_valid_m_l_sums))
    all_valid_combinations = 0
    for c in coefficients:
        all_valid_combinations += c
    # all_valid_combinations = reduce(lambda a, b: a+b, coefficients, 0)
    all_combos = m**l
    print("len(all_valid_m_l_sums): {}".format(len(all_valid_m_l_sums)))
    print("all_valid_combinations: {}".format(all_valid_combinations))
    print("all_combos: {}".format(all_combos))
    probability = all_valid_combinations/all_combos

    return probability

m = 12
l = 12
all_valid_m_l_sums = get_all_valid_m_l_sums(m, l)
print("all_valid_m_l_sums: m: {}, l: {}".format(m, l))
for valid_m_l_sum in all_valid_m_l_sums:
    print("{}".format(valid_m_l_sum))
prob = calc_propability_all_element_more_1_time(m, l)
print("probability for m: {} and l: {}: {}".format(m, l, prob))

labels = []
xs = []
ys = []
probabilities = []
for m in range(2, 16):
    labels.append("m: {}".format(m))
    x = []
    y = []
    for l in range(m, m*7, m):
        x.append(l//m)
        y.append(calc_propability_all_element_more_1_time(m, l))
    xs.append(x)
    ys.append(y)

plt.figure()
plts = []
for x, y in zip(xs, ys):
    plts.append(plt.plot(x, y, "-")[0])
plt.legend(plts, labels, loc="upper right")
plt.show()
