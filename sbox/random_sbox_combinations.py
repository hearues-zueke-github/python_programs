#! /usr/bin/python3.5

# -*- coding: utf-8 -*-

import operator

import numpy as np

from copy import deepcopy

import sys
sys.path.append("../encryption")

import Utils

if __name__ == "__main__":
    n = 12 # amount of elements = modulo = number base

    x = np.arange(0, n, dtype=np.int)
    print("x: {}".format(x))

    f_x_set = set()
    factors_f_x = {}
    f_x_factors = {}
    func_amount = 3
    func_i = 0
    while func_i < func_amount:
        f_x = np.random.permutation(x)
        f_x_tpl = tuple(f_x.tolist())
        factors = (func_i, )
        if f_x_tpl in f_x_set:
            continue

        f_x_set.add(f_x_tpl)
        factors_f_x[factors] = f_x
        f_x_factors[f_x_tpl] = factors

        func_i += 1

    print("len(f_x_set): {}".format(len(f_x_set)))
    print("len(factors_f_x): {}".format(len(factors_f_x)))

    # now combine with one combination to another combination
    def get_new_combinations_random(f_x_set, factors_f_x, f_x_factors, n=500000):
        factors_f_x_2 = {}
        keys = list(factors_f_x)
        len_keys = len(keys)
        get_random_key = lambda: keys[np.random.randint(0, len_keys)]

        lowered_size = 0
        size_equal = 0
        for i in range(0, n):
            k1 = get_random_key()
            k2 = get_random_key()

            arr1 = factors_f_x[k1]
            arr2 = factors_f_x[k2]
            f_x = arr2[arr1]
            f_x_tpl = tuple(f_x.tolist())
            
            new_factors = k1+k2
            if not f_x_tpl in f_x_set:
                f_x_set.add(f_x_tpl)
                factors_f_x_2[new_factors] = f_x
                f_x_factors[f_x_tpl] = new_factors
            elif len(new_factors) < len(f_x_factors[f_x_tpl]):
                f_x_factors[f_x_tpl] = new_factors
                lowered_size += 1
            else:
                size_equal += 1

        print("len(f_x_set): {}".format(len(f_x_set)))
        print("len(factors_f_x_2): {}".format(len(factors_f_x_2)))
        print("lowered_size: {}, size_equal: {}".format(lowered_size, size_equal))

        return factors_f_x_2

    def get_only_true_sbox(n, factors_f_x):
        factors_f_x = deepcopy(factors_f_x)

        x = np.arange(0, n, dtype=np.int)
        keys = list(factors_f_x)
        for key in keys:
            arr = factors_f_x[key]

            if np.any(arr == x):
                factors_f_x.pop(key, None)

        return factors_f_x

    for _ in range(0, 30):
        print("Calc new combinations.")
        factors_f_x_2 = get_new_combinations_random(f_x_set, factors_f_x, f_x_factors)
        print("Combine previous combinations.")
        factors_f_x = {**factors_f_x, **factors_f_x_2}

        max_factor_len = np.max(list(map(len, list(factors_f_x.keys()))))
        print("max_factor_len: {}".format(max_factor_len))

    # factors_f_x_sboxes = get_only_true_sbox(n, factors_f_x)
    # print("len(factors_f_x_sboxes): {}".format(len(factors_f_x_sboxes)))

    initial_f_xs = operator.itemgetter(*[(i,) for i in range(0, func_amount)])(factors_f_x)
    # for i, f_x in enumerate(initial_f_xs):
    #     print("i: {:2}, initial_f_xs: {}".format(i, f_x))
