#! /usr/bin/python3.5

# -*- coding: utf-8 -*-

import numpy as np

from copy import deepcopy

import sys
sys.path.append("../encryption")

import Utils

if __name__ == "__main__":
    n = 10 # amount of elements = modulo = number base

    x = np.arange(0, n, dtype=np.int)

    # idx_change = np.hstack((np.arange(0, n, 2), np.arange(1, n, 2)))
    idx_change = np.hstack(((1, 0), np.arange(2, n)))
    # idx_change = np.arange(n-1, -1, -1)

    print("x: {}".format(x))

    # TODO: make it much more efficient!
    f_x_set = set()
    
    # get all a_0 and a_1 factors, which will create different outputs
    factors_f_x = {}
    f_x_factors = {}
    for a_0 in range(0, n):
      for a_1 in range(0, n):
       for a_2 in range(0, n):
        for a_3 in range(0, n):
            f_x = (a_0+a_1*x+a_2*x**2+a_3*x**3) % n
            # f_x = (a_0+a_1*x+a_2*x**2) % n
            if np.unique(f_x).shape[0] == n:
                f_x_tpl = tuple(f_x.tolist())
                factors = ((a_0, a_1, a_2, a_3), )
                # factors = ((a_0, a_1, a_2), )
                if not f_x_tpl in f_x_set:
                    f_x_set.add(f_x_tpl)
                    factors_f_x[factors] = f_x
                    f_x_factors[f_x_tpl] = factors

                f_x_2 = f_x[idx_change]
                f_x_tpl_2 = tuple(f_x_2.tolist())
                factors = (factors[0]+(-1, ), )
                if not f_x_tpl_2 in f_x_set:
                    f_x_set.add(f_x_tpl_2)
                    factors_f_x[factors] = f_x_2
                    f_x_factors[f_x_tpl_2] = factors

    print("len(f_x_set): {}".format(len(f_x_set)))
    print("len(factors_f_x): {}".format(len(factors_f_x)))

    # # now combine with one combination to another combination
    # def get_new_combinations(f_x_set, factors_f_x, f_x_factors):
    #     factors_f_x_2 = {}

    #     for k1 in factors_f_x:
    #         for k2 in factors_f_x:
    #             arr1 = factors_f_x[k1]
    #             arr2 = factors_f_x[k2]
    #             f_x = arr2[arr1]
    #             f_x_tpl = tuple(f_x.tolist())
                
    #             if not f_x_tpl in f_x_set:
    #                 f_x_set.add(f_x_tpl)
    #                 factors_f_x_2[k1+k2] = f_x
    #             else:


    #     print("len(f_x_set): {}".format(len(f_x_set)))
    #     print("len(factors_f_x_2): {}".format(len(factors_f_x_2)))

    #     return factors_f_x_2

    # now combine with one combination to another combination
    def get_new_combinations_random(f_x_set, factors_f_x, f_x_factors, n=100000):
        factors_f_x_2 = {}
        keys = list(factors_f_x)
        len_keys = len(keys)
        get_random_key = lambda: keys[np.random.randint(0, len_keys)]

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

        print("len(f_x_set): {}".format(len(f_x_set)))
        print("len(factors_f_x_2): {}".format(len(factors_f_x_2)))

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

    for _ in range(0, 5):
        print("Calc new combinations.")
        factors_f_x_2 = get_new_combinations_random(f_x_set, factors_f_x, f_x_factors)
        print("Combine previous combinations.")
        factors_f_x = {**factors_f_x, **factors_f_x_2}

    # factors_f_x_sboxes = get_only_true_sbox(n, factors_f_x)
    # print("len(factors_f_x_sboxes): {}".format(len(factors_f_x_sboxes)))
