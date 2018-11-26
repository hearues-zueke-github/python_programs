#! /usr/bin/python3.5

# -*- coding: utf-8 -*-

import numpy as np
np.set_printoptions(threshold=np.nan)

from dotmap import DotMap
from functools import reduce

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

def find_unique_f_x(row_n, n, p, lens_max_arr, factors_arr, factors_unique_arr, f_x_factors_arr):
    lens = set()
    unique_f_x_only = set()

    hit = 0
    miss = 0

    factors_sbox = []
    f_x_factors = {}

    full_factors = np.zeros((0, p+1), dtype=np.int)
    
    # # old
    # factors = np.zeros((n, )*p+(p+1, ), dtype=np.int)
    
    # exec_str_temp_before = "factors["+":"+(", :"*(p-1) if p > 1 else "")+", {}] = np.arange(0, n).reshape((-1{}))"
    # for i in range(1, p+1):
    #     exec(exec_str_temp_before.format(i, (", 1"*(i-1) if i > 1 else ", ")))
    
    # exec_str_temp = "factors["+":"+(", :"*(p-1) if p > 1 else "")+", 0] = {}"
    
    # for v in range(0, 1): # n):
    #     exec(exec_str_temp.format(v))
    #     print("row_n: {}, p: {}, v: {}".format(row_n, p, v))

    
    # new
    factors = np.zeros((n, )*(p-1 if p > 1 else 0)+(p+1, ), dtype=np.int)
    
    exec_str_temp_before = "factors["+":"+(", :"*(p-2) if p > 2 else "")+", {}] = np.arange(0, n).reshape((-1{}))"
    for i in range(0, p-1):
        exec_finish_str = exec_str_temp_before.format(i+2, (", 1"*i if i > 0 else ", "))
        # print("exec_finish_str: {}".format(exec_finish_str))
        exec(exec_finish_str)
    
    exec_str_temp = "factors["+":"+(", :"*(p-2) if p > 2 else "")+", :2] = {}" # .format(p-1)
    for v1 in range(0, 1): # n):
      for v2 in range(1, 4, 2):
        print("row_n: {}, p: {}, v: {}".format(row_n, p, (v1, v2)))
        exec(exec_str_temp.format((v1, v2)))

    
    # # random factors!
    # for v in range(0, 1):
    #     factors = np.random.randint(0, n, (3000, p+1), dtype=np.int)
    #     fs_x = apply_polynomial_function(n, p, factors.reshape((-1, p+1)))
    

        fs_x = apply_polynomial_function(n, p, factors.reshape((-1, p+1)))
        full_factors = np.vstack((full_factors, factors.reshape((-1, p+1))))

        for facts, row in zip(factors.reshape((-1, p+1)), fs_x):
        
            unique_len = len(np.unique(row))
            lens.add(unique_len)

            if unique_len == n:
                hit += 1
                f_x_tpl = tuple(row)
                facts_lst = facts.tolist()
                if not f_x_tpl in unique_f_x_only:
                    unique_f_x_only.add(f_x_tpl)
                    factors_sbox.append(facts_lst)
                    f_x_factors[f_x_tpl] = [facts_lst]
                else:
                    lst = f_x_factors[f_x_tpl]
                    if not facts_lst in lst:
                        lst.append(facts_lst)

            else:
                miss += 1

    print("\nn: {}, p: {}, len(unique_f_x_only): {}".format(n, p, len(unique_f_x_only)))
    print("hit: {}, miss: {}".format(hit, miss))
    print("Chance of getting a unique f_x by random: {:02.3f}%".format(hit/float(hit+miss)*100))
    
    lens_max_arr[row_n, p] = len(unique_f_x_only)
    factors_arr[row_n, p] = np.vstack((factors_arr[row_n, p], np.array(factors_sbox)))
    factors_unique_arr[row_n, p] = [np.unique(col) for col in factors_arr[row_n, p].T]
    f_x_factors_arr[row_n, p] = f_x_factors
    globals()["full_factors"] = full_factors
    # print("full_factors: {}".format(full_factors))

if __name__ == "__main__":
    ns = list(2**i for i in range(1, 6))
    ps = 7 # max power
    # max_p = 1
    # arr = np.arange(0, n, dtype=object) # object is used bcs of BigInt of python
    
    lens_max_arr = np.zeros((len(ns)+1, ps+1), dtype=np.int)
    factors_arr = np.zeros((len(ns)+1, ps+1), dtype=object)
    factors_unique_arr = np.zeros((len(ns)+1, ps+1), dtype=object)
    f_x_factors_arr = np.zeros((len(ns)+1, ps+1), dtype=object)
    
    print("lens_max_arr.shape: {}".format(lens_max_arr.shape))
    print("factors_arr.shape: {}".format(factors_arr.shape))

    lens_max_arr[1:, 0] = ns
    lens_max_arr[0] = np.arange(0, ps+1)
    factors_arr[1:, 0] = ns
    factors_arr[0] = np.arange(0, ps+1)
    factors_unique_arr[1:, 0] = ns
    factors_unique_arr[0] = np.arange(0, ps+1)
    f_x_factors_arr[1:, 0] = ns
    f_x_factors_arr[0] = np.arange(0, ps+1)

    factors_arr[1:, 1:] = [[np.array([], dtype=np.int).reshape((0, p+1)) for p in range(1, ps+1)] for n in range(0, len(ns))]
    f_x_factors_arr[1:, 1:] = [[{} for p in range(1, ps+1)] for n in range(0, len(ns))]
    
    row_n = 5
    n = ns[row_n-1]
    # for p in range(1, ps+1):
    choosen_p = 4
    # for p in range(4, 5):
    p = choosen_p
    find_unique_f_x(row_n, n, p, lens_max_arr, factors_arr, factors_unique_arr, f_x_factors_arr)

    factors_only = reduce(lambda a, b: a+b, f_x_factors_arr[row_n, choosen_p].values(), [])
    # factors_only_sorted = 

    factors_only_arr = (lambda x: x.reshape((x.shape[0], int(np.multiply.reduce(x.shape[1:])))))(np.array(list(f_x_factors_arr[row_n, choosen_p].values())))
    # for row_n, n in zip(range(1, len(ns)+1), ns):
    #  # print("row_n: {}".format(row_n))
    #  # print("n: {}".format(n))
    #  for p in range(1, ps+1):
    #     find_unique_f_x(row_n, n, p, lens_max_arr, factors_arr, factors_unique_arr)

    arr = factors_arr[row_n, choosen_p]
    # arr = factors_only_arr[0].reshape((-1, choosen_p+1))
    arr_sorted = np.sort(arr.reshape((-1, )).view("u8"+",u8"*choosen_p)).view("u8").reshape(arr.shape)
