#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import sys

import numpy as np

import matplotlib.pyplot as plt

import decimal
from decimal import Decimal as Dec

decimal.getcontext().prec = 100

def approach_1(n):
    vals = np.random.randint(0, n, (100000, ))

    amount_tries = []

    vs = np.zeros((n, ), dtype=np.int)
    tries = 0
    for v in vals:
        tries += 1
        vs[v] = 1
        if np.sum(vs) == n:
            amount_tries.append(tries)
            tries = 0
            vs[:] = 0

    print("amount_tries: {}".format(amount_tries))

    unique, counts = np.unique(amount_tries, return_counts=True)
    print("unique: {}".format(unique))
    print("counts: {}".format(counts))

    plt.figure()

    plt.plot(unique, counts)
    plt.ylim([-1, np.max(counts)+5])

    plt.show()


def approach_2(n):
    l = 8000000
    # vals = np.random.randint(0, n, (l, 100))
    counts = np.zeros((l, n), dtype=np.int)
    xs = np.arange(0, l)

    amount_tries = []
    for tries in range(1, 150+1):
        l = counts.shape[0]
        if l == 0:
            break
        vals = np.random.randint(0, n, (l, ))
        # print("vals.shape[0]: {}".format(vals.shape[0]))
        counts[xs, vals] = 1
        idx = np.sum(counts, axis=1)==n
        if np.any(idx):
            # vals = vals[~idx]
            counts = counts[~idx]
            xs = np.arange(0, counts.shape[0])

            amount_tries.append((tries, np.sum(idx)))
    
    amount_tries = np.array(amount_tries).T
    expected_tries = np.sum(amount_tries[0]*amount_tries[1]/np.sum(amount_tries[1]))

    return amount_tries, expected_tries

def calculate_estimation_for_n_3():
    n = 3
    ii = np.arange(n, 1000)

    dec_n = Dec(n)
    s = Dec(0)
    for i in ii:
        i = int(i)
        print("\ni: {}".format(i))
        s1 = 0
        k = i-n+1
        for j in range(1, k+1):
            s1 += 1**j*2**(k-j+1)
        s += Dec(i)*Dec(s1)/dec_n**(k+1)
        print("s1: {}".format(s1))
        print("s: {}".format(s))

    return s

def 

def calculate_estimation_for_n_4():
    tries = 7
    n = 4

    lefts = n - 2
    rights = tries-n
    combos = []

    idx = np.zeros((rights**lefts, lefts), dtype=np.int)
    idx_numbers = np.zeros((rights+1, ), dtype=np.int)
    i = lefts-1
    while i >= 0:



if __name__ == "__main__":
    # n = 5

    expected_tries_n_3 = calculate_estimation_for_n_3()
    # sys.exit(0)

    n_max = 22
    ns = []
    expected_tries_lst = []
    for n in range(2, n_max+1):
        amount_tries, expected_tries = approach_2(n)
        # print("amount_tries:\n{}".format(amount_tries))
        print("n: {}, expected_tries: {}".format(n, expected_tries))
        ns.append(n)
        expected_tries_lst.append(expected_tries)

    plt.figure()

    plt.title("Expected tries per n")
    plt.xlabel("n")
    plt.ylabel("Expected tries")
    
    plt.plot(ns, expected_tries_lst, "b-.")

    plt.savefig("exptected_values_from_2_to_{}_a.png".format(n_max))
    # plt.show()
