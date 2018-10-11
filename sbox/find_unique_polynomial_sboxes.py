#! /usr/bin/python3.5

# -*- coding: utf-8 -*-

import numpy as np

import sys
sys.path.append("../encryption")

import Utils

# @param n: amount of elements = modulo
# @param p: the max potence of the polynomial function
# @param factors: all factors of a polynomial function
#                 factors[0] is equal to a_0
#                 f(x) = a_0+a_1*x+a_2*x**2+...+a_p*x**p
# @return: an array with the size n, f(x) values mod n
def apply_polynomial_function(n, p, factors):
    assert isinstance(n, int)
    assert n > 0
    assert isinstance(p, int)
    # assert isinstance(factors, list)
    # assert len(factors) == p+1
    assert factors.shape[1] == p+1

    x = np.arange(0, n, dtype=object)

    # TODO: need to finish this up!
    f_x = np.zeros((factors.shape[0], n), dtype=object)+factors[:, 0].reshape((-1, 1))
    for i, a_i in zip(range(1, p+1), factors.T[1:]):
        f_x += a_i.reshape((-1, 1))*x**i
    
    return f_x % n

if __name__ == "__main__":
    # ns = 40 # amount of elements = modulo
    ps = 2 # max power
    # max_p = 1
    # arr = np.arange(0, n, dtype=object) # object is used bcs of BigInt of python

    lens_f_x = {}
    # rows_n = list(range(3, ns+1))
    ns = list(2**i for i in range(0, 8))
    
    lens_arr = np.zeros((len(ns)+1, ps+1), dtype=np.int)
    factors_arr = np.zeros((len(ns)+1, ps+1), dtype=object)
    
    lens_arr[1:, 0] = ns
    lens_arr[0] = np.arange(0, ps+1)
    factors_arr[1:, 0] = ns
    factors_arr[0] = np.arange(0, ps+1)
    
    for row_n, n in zip(range(1, len(ns)+1), ns):
     # print("row_n: {}".format(row_n))
     # print("n: {}".format(n))
     for p in range(1, ps+1):
        lens = set()
        unique_factors = set()
        unique_f_x = {}
        unique_f_x_only = set()
        
        # if p == 1:
        #     factors = np.zeros((n, n, 2), dtype=np.uint8)
        #     factors[:, :, 0] = np.arange(0, n)
        #     factors[:, :, 1] = np.arange(0, n).reshape((-1, 1))
        # elif p == 2:
        #     factors = np.zeros((n, n, n, 3), dtype=np.uint8)
        #     factors[:, :, :, 0] = np.arange(0, n)
        #     factors[:, :, :, 1] = np.arange(0, n).reshape((-1, 1))
        #     factors[:, :, :, 2] = np.arange(0, n).reshape((-1, 1, 1))
        # elif p == 3:
        #     factors = np.zeros((n, n, n, n, 4), dtype=np.uint8)
        #     factors[:, :, :, :, 0] = np.arange(0, n)
        #     factors[:, :, :, :, 1] = np.arange(0, n).reshape((-1, 1))
        #     factors[:, :, :, :, 2] = np.arange(0, n).reshape((-1, 1, 1))
        #     factors[:, :, :, :, 3] = np.arange(0, n).reshape((-1, 1, 1, 1))
        # factors = factors.reshape((-1, p+1))

        hit = 0
        miss = 0
        try:
            fs_x = apply_polynomial_function(n, p, factors)
            print("\nfactors.shape: {}".format(factors.shape))

            factors_sbox = []
            for facts, row in zip(factors, fs_x):
                unique_len = len(np.unique(row))
                lens.add(unique_len)

                if unique_len == n:
                    hit += 1
                    f_x_tpl = tuple(row)
                    if not f_x_tpl in unique_f_x_only:
                        unique_f_x_only.add(f_x_tpl)
                        factors_sbox.append(facts)
                        # print("len(unique_f_x_only): {}".format(len(unique_f_x_only)))
                    # if not factors_tpl in unique_factors:
                    #     unique_factors.add(factors_tpl)
                    #     print("factors: {}, len(unique_factors): {}".format(factors, len(unique_factors)))
                    #     # print("unique_factors: {}".format(unique_factors))
                    #     f_x_tpl = tuple(f_x)
                    #     if f_x_tpl in unique_f_x:
                    #         unique_f_x[f_x_tpl].add(factors_tpl)
                    #     else:
                    #         unique_f_x[f_x_tpl] = set([factors_tpl])
                else:
                    miss += 1

        except KeyboardInterrupt:
            pass

        # for f_x_num, f_x_tpl in enumerate(unique_f_x):
        #     print("f_x_num: {}, f_x_tpl: {}".format(f_x_num, f_x_tpl))
        #     all_factors = list(unique_f_x[f_x_tpl])
        #     for i, factors in enumerate(all_factors):
        #         print("  i: {}, factors: {}".format(i, factors))
        # arr_factors = np.array(list(unique_factors), dtype=np.int64)
        # sorted_factors  = np.sort(arr_factors.reshape((-1, )).view("u8"+(",u8"*p if p > 1 else ""))).view("u8").reshape(arr_factors.shape)
        
        print("n: {}, p: {}, len(unique_f_x_only): {}".format(n, p, len(unique_f_x_only)))
        print("hit: {}, miss: {}".format(hit, miss))
        print("Chance of getting a unique f_x by random: {:02.3f}%".format(hit/float(hit+miss)*100))
        
        # print("sorted_factors:\n{}".format(sorted_factors))

        lens_f_x[(row_n, p)] = [len(unique_f_x_only), hit, miss]
        lens_arr[row_n, p] = len(unique_f_x_only)
        factors_arr[row_n, p] = factors_sbox

    keys = sorted(list(lens_f_x.keys()))

