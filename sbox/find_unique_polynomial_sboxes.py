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
    assert factors.shape[1] == p+1

    factors = factors.T
    x = np.arange(0, n, dtype=object)
    f_x = np.zeros((factors.shape[1], n), dtype=object)+factors[0].reshape((-1, 1))
    for i, a_i in zip(range(1, p+1), factors[1:]):
        f_x += a_i.reshape((-1, 1))*x**i
    
    return f_x % n

if __name__ == "__main__":
    ns = list(2**i for i in range(0, 5))
    ps = 3 # max power
    # max_p = 1
    # arr = np.arange(0, n, dtype=object) # object is used bcs of BigInt of python
    
    lens_max_arr = np.zeros((len(ns)+1, ps+1), dtype=np.int)
    factors_arr = np.zeros((len(ns)+1, ps+1), dtype=object)
    
    print("lens_max_arr.shape: {}".format(lens_max_arr.shape))
    print("factors_arr.shape: {}".format(factors_arr.shape))

    lens_max_arr[1:, 0] = ns
    lens_max_arr[0] = np.arange(0, ps+1)
    factors_arr[1:, 0] = ns
    factors_arr[0] = np.arange(0, ps+1)
    
    for row_n, n in zip(range(1, len(ns)+1), ns):
     # print("row_n: {}".format(row_n))
     # print("n: {}".format(n))
     for p in range(1, ps+1):
        lens = set()
        unique_f_x_only = set()

        hit = 0
        miss = 0

        factors = np.zeros((n, )*p+(p+1, ), dtype=np.int)
        
        exec_str_temp_before = "factors["+":"+(", :"*(p-1) if p > 1 else "")+", {}] = np.arange(0, n).reshape((-1{}))"
        for i in range(1, p+1):
            exec(exec_str_temp_before.format(i, (", 1"*(i-1) if i > 1 else ", ")))
        
        exec_str_temp = "factors["+":"+(", :"*(p-1) if p > 1 else "")+", 0] = {}"
        
        factors_sbox = []

        # exec(exec_str_temp.format(0))
        # facts_arr = factors.copy()
        # fss_x = apply_polynomial_function(n, p, factors.reshape((-1, p+1)))

        # for v in range(1, n):
        #     exec(exec_str_temp.format(v))
        #     facts_arr = np.vstack((facts_arr, factors))
        #     fs_x = apply_polynomial_function(n, p, factors.reshape((-1, p+1)))
        #     fss_x = np.vstack((fss_x, fs_x))

        # facts_arr = facts_arr.reshape((-1, p+1))
        # fss_x = fss_x.reshape((-1, n))

        # print("facts_arr:\n{}".format(facts_arr))
        # print("fss_x:\n{}".format(fss_x))
        # input()
            
        # if True:
        #     for facts, row in zip(facts_arr, fss_x):

        for v in range(0, n):
            exec(exec_str_temp.format(v))
            fs_x = apply_polynomial_function(n, p, factors.reshape((-1, p+1)))

            for facts, row in zip(factors.reshape((-1, p+1)), fs_x):
            
                unique_len = len(np.unique(row))
                lens.add(unique_len)

                if unique_len == n:
                    hit += 1
                    f_x_tpl = tuple(row)
                    if not f_x_tpl in unique_f_x_only:
                        unique_f_x_only.add(f_x_tpl)
                        factors_sbox.append(facts.tolist())
                else:
                    miss += 1

        print("\nn: {}, p: {}, len(unique_f_x_only): {}".format(n, p, len(unique_f_x_only)))
        print("hit: {}, miss: {}".format(hit, miss))
        print("Chance of getting a unique f_x by random: {:02.3f}%".format(hit/float(hit+miss)*100))
        
        lens_max_arr[row_n, p] = len(unique_f_x_only)
        # print("factors_sbox: {}".format(factors_sbox))
        factors_arr[row_n, p] = factors_sbox
