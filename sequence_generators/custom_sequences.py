#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import functools
import sys
import time

import numpy as np

from indexed import IndexedOrderedDict

from PIL import Image, ImageTk

import multiprocessing as mp
from multiprocessing import Process, Pipe, Lock

import utils_sequence


def get_arr_shifts(arr, length):
    arr_length = arr.shape[0]
    arr_shifts = np.zeros((length, arr_length+length-1), dtype=np.uint8)

    for i in range(0, length):
        arr_shifts[i, i:arr_length+i] = arr
 
    arr_shifts = arr_shifts.T
    arr_shifts = arr_shifts[length-1:-length]

    return arr_shifts


def f_mult_1(n):
    sqnc = [1] # starting sequence
    m = 10 # modulo
    for i in range(len(sqnc)-1, n-1):
        s = 0 # sum for the next number
        j = i # used iterator
        acc = 0 # used accumulator for jumping the index
        while j >= 0:
            x = sqnc[j] # get the x from sqnc with index j
            s = (s + x) % m # add new value to the sum
            acc += x + 1 # increase the accumulator by x + 1
            j -= acc # and jump by acc back
        sqnc.append(s)

    return sqnc


def f(n):
    sqnc = [1] # starting sequence
    m = 10 # modulo
    for i in range(len(sqnc)-1, n-1):
        s = 0 # sum for the next number
        j = i # used iterator
        acc = 0 # used accumulator for jumping the index
        multiplier = 1
        while j >= 0:
            x = sqnc[j] # get the x from sqnc with index j
            s = (s + x * multiplier) % m # add new value to the sum
            multiplier = (multiplier + 1) % m # also increment multiplier
            acc += x + 1 # increase the accumulator by x + 1
            j -= acc # and jump by acc back
        sqnc.append(s)

    return sqnc


def f2(n):
    def rec(n, m):
        def f(n, acc):
            if n >= 0:
                return (sqnc[n]+f(n-acc-sqnc[n]-1, acc+sqnc[n]+1)) % m
                # return (sqnc[n]+f(n//2-1)) % m
                # return (sqnc[n]+f(n//2-1)+f(n//2-2)) % m
            return 0
        return f(n, 0)
    sqnc = [1] # starting sequence
    m = 10 # modulo
    for i in range(len(sqnc)-1, n-1):
        v = rec(i, m)
        sqnc.append(v)

    return sqnc

def get_sequence(n):
    def a(n):
        if n > 1:
            return f(n-1, 0)
        elif n == 1:
            return 1
        return 0
    def f(n, acc):
        if n >= 1:
            a_n = a(n)
            return (a_n+f(n-acc-a_n-1, acc+a_n+1)) % 10
        return 0
    lst = [a(i) for i in range(1, n+1)]
    return lst


def get_sequence_better(n):
    def a(n):
        if n in a.vals:
            return a.vals[n]
        if n == 1:
            return 1
        elif n < 1:
            return 0
        v = f(n-1, 0)
        a.vals[n] = v
        return v
    def f(n, acc):
        t = (n, acc)
        if t in f.vals:
            return f.vals[t]
        if n >= 1:
            a_n = a(n)
            v = (a_n+f(n-acc-a_n-1, acc+a_n+1)) % 10
            f.vals[t] = v
            return v
        return 0
    a.vals = {1: 1}
    f.vals = {}
    for i in range(1, n+1):
        a(i)
    return list(a.vals.values())


# def f_sum_increase_range(n):
#     lst = [1]
#     # idx = 3
#     # increase_factor = idx-1
#     sum_table = get_hanoi_sequence(n)
#     print("sum_table: {}".format(sum_table))
#     for i in range(1, n):
#         sum_table[i] = (sum_table[i]+sum_table[i-1]) # % 10
#     print("sum_table 2: {}".format(sum_table))
#     for i in range(1, n):
#         sum_table[i] = (sum_table[i]+sum_table[i-1]) # % 10
#     print("sum_table 3: {}".format(sum_table))
#     return sum_table

#     # idx = 0
#     # while idx < n:
#     #     increase_factor = sum_table[idx]
#     #     # increase_factor += (idx) % 3 == 0
#     #     s = 0
#     #     for i in range(idx-increase_factor+1, idx+1):
#     #         s += lst[i]
#     #     lst.append(s)
#     #     # lst.append(s % 10)
#     #     idx += 1

#     # return lst

def jumping_sum_index_sequence(n):
    idxs = list(range(1, n+1))
    lst = [0 for _ in range(0, n+1)]

    def f(n):
        return n + (-1 + 2 * (n % 2))

    l = [f(i) for i in range(1, n+1)]

    def f1(n):
        return 2**n

    f1 = f

    for i in range(1, n):
        if len(idxs) == 0:
            break

        lst[idxs.pop(0)] = i
        h = 0
        k = 1
        idx = f1(k)
        while True:
            if len(idxs) < idx:
                break
            h += idx-1
            if len(idxs) <= h:
                break
            j = idxs.pop(h)
            if j > n:
                break
            lst[j] = i

            k += 1
            idx = f1(k)

    return lst[1:]


def f_rec_mult_1(n, m=10):
    hanoi_seq = get_hanoi_sequence(n)
    print("hanoi_seq: {}".format(hanoi_seq))
    def next_num(n):
        s = a(n)
        next_num.vals.append(s)
        # print("n: {n}, s: {s}".format(n=n, s=s))
        return s
    def a(n):
        if n in a.vals:
            a.vals_count[n] += 1
            return a.vals[n]
        # if n == 0:
        #     return 0
        if n == 1:
            return 1
        elif n < 1:
            return 0
        # v = (f(n-1, 0)+n) # % m
        # print("v: {}".format(v))
        v = f(n-1, 0)
        a.vals_count[n] = 1
        a.vals[n] = v
        return v
    def f(n, acc):
        t = (n, acc)
        if t in f.vals:
            f.vals_count[t] += 1
            # print("f.vals[t]: {}".format(f.vals[t]))
            return f.vals[t]
        s = 0
        if n >= 1 or True:
            # # the sums of the sums!
            # for i in range(1, n+1):
            #     a_i = a(i)
            #     s = (s+f(i-acc-a_i-1, acc+a_i+1)+a_i) % m
            a_n = a(n) # % m
            # print("n: {}, a_n: {}".format(n, a_n))

            # the original one!
            s = (f(n-acc-a_n-1, acc+a_n+1)+a_n) % m
            # m: 2, [1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0,...]
            # m: 10, [1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 9, 0, 0, 7, 3, 2, 1,...]

            # # acc = acc+1
            # s = (f(n-a_n%m-1, 0)+a_n) # % m
            # s = (f(n-acc-1, acc+1)+a_n)# % m
            # m: 2, [1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1,...]
            # m: 10, [1, 1, 2, 3, 6, 0, 8, 2, 7, 1, 9, 9, 6, 6, 6, 4, 8, 6, 3,...]

            # # the correct one! no modulo!
            # s = (f(n-acc-a_n-1, acc+a_n+1)+a_n) % m
            # m: 2, [1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0,...]
            # m: 10, [1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 19, 20, 40, 77, 83,...]
            
            # hanoi jumping!
            # s = (f(n-hanoi_seq[acc], acc+1)+a_n) % m


            # # interesting too!
            # s = (f(n-acc-(a_n%m)-1, acc+(a_n%m)+1)+a_n) # % m

            f.vals_count[t] = 1
            f.vals[t] = s
        return s


    a.vals = {1: 1}
    a.vals_count = {1: 0}
    f.vals = {}
    f.vals_count = {}
    next_num.vals = []

    vals = [next_num(j) for j in range(1, n+1)]
    # print("vals:\n{}".format(vals))

    return vals, next_num, a, f


def f_jumping_modulo(n, m=10):
    idxs = list(range(1, n*m+1))
    # vals_mod = np.arange(0, n) % m
    vals = IndexedOrderedDict([(i, -1) for i in range(1, n*m+1)]) # {i: -1 for i in range(1, n*m+1)}

    vals_mod = [i for j in range(1, 30+1) for i in range(0, j)]

    smallest_idx = 0
    largest_idx = 0
    idx_now = 0
    # for i, v in enumerate(vals_mod[:50]):
    j = 1
    max_reached = False
    while not max_reached:
        for v in range(j-1, -1, -1):
        # for v in range(0, j):
            if v > idx_now:
                idx_now += v
            else:
                idx_now -= v
            if idx_now >= len(idxs):
                max_reached = True
                break
            idx = idxs.pop(idx_now)
            vals[idx] = v
            
            print("idx_now: {}, idx: {}, v: {}".format(idx_now, idx, v))
            
            if smallest_idx < idxs[0]:
                smallest_idx = idxs[0]
            # if largest_idx < idx:
            #     largest_idx = idx
            # lst_temp = list(vals.values())[:largest_idx]
            # print("i: {}, lst_temp:\n{}".format(i, lst_temp))
        j += 1
    lst = list(vals.values())[:smallest_idx-1]
    # print("lst: {}".format(lst))
    # print("idxs: {}".format(idxs))
    # arr = np.array(lst[:smallest_idx-1]) # % m
    # arr = np.array(lst[:smallest_idx-1]) # % m

    return lst
    # return arr


def f_with_recursion_mult_1(n):
    lst = [0, 1]
    modulo = 10
    def f1(n_start):
        def f2(n, s, acc):
            new_s = s
            if n >= 1:
                new_s = f2(
                            n-acc-lst[n]-1,
                            (s+lst[n])%modulo,
                            acc+lst[n]+1,
                        )
            return new_s
        return f2(n_start-1, 0, 0)

    len_lst = len(lst)
    for i in range(len_lst, len_lst+n-1):
        lst.append(f1(i))

    return lst[1:]


def f_with_recursion(n):
    lst = [0, 1]
    lengths = []
    modulo = 10
    def f1(n_start):
        def f2(n, s, acc, mult, times):
            new_s = s
            if n >= 1:
                new_s = f2(
                            n-acc-lst[n]-1,
                            (s+lst[n]*mult)%modulo,
                            acc+lst[n]+1,
                            (mult+1)%modulo,
                            times+1
                        )
            else:
                lengths.append(times)

            return new_s

        return f2(n_start-1, 0, 0, 1, 0)

    len_lst = len(lst)
    for i in range(len_lst, len_lst+n-1):
        lst.append(f1(i))

    print("lengths:\n{}".format(lengths))
    lens_arr = np.array(lengths)
    diff = lens_arr[1:]-lens_arr[:-1]
    print("diff:\n{}".format(diff))

    return lst[1:]


def get_hanoi_sequence(n):
    lst = [1]
    for i in range(2, n):
        lst = lst+[i]+lst
        if 2**i-1 > n:
            break
    return lst[:n]


if __name__ == "__main__":
    n = 200
    # lst = f_mult_1(n)
    m = 10
    # lst = f_jumping_modulo(n, m=m)
    # lst, next_num, a, f = f_rec_mult_1(n, m=m)
    # lst_hanoi = get_hanoi_sequence(5)
    # lst = f_sum_increase_range(n)
    # lst = jumping_sum_index_sequence(n)
    
    lst = get_sequence_better(n)
    # lst = get_sequence(n)

    # print("lst:\n{}".format(",".join(list(map(str, lst)))))
    print("n: {}, m: {}".format(n, m))
    print("lst:\n{}".format(lst))

    lst_with_recursion = f_with_recursion_mult_1(n)
    # print("lst_with_recursion:\n{}".format(",".join(list(map(str, lst_with_recursion)))))
    print("lst_with_recursion:\n{}".format(lst_with_recursion))

    arr = np.array(lst)
    # arr_mod = arr%m
    # print("arr_mod:\n{}".format(arr_mod))
    with open("b322670.txt", "w") as fout:
    # with open("b322670_f_with_recursion_mult_1.txt", "w") as fout:
    # with open("b_sequence.txt", "w") as fout:
        for i, x in enumerate(arr, 1):
            fout.write("{} {}\n".format(i, x))

    if True:
        arr_pattern, idx = utils_sequence.find_first_repeat_pattern(arr)

        if idx == None:
            print("No Pattern could be found! More information needed!")
        else:
            pattern_length = arr_pattern.shape[0]

            first_part = arr[:idx]
            amount_patterns = (arr.shape[0]-idx)//pattern_length
            last_part = arr[idx+amount_patterns*pattern_length:]

            print("\nfirst_part.shape[0]: {}".format(first_part.shape[0]))
            print("first_part:\n{}".format(first_part))
            
            print("\namount_patterns: {}".format(amount_patterns))
            print("arr_pattern.shape[0]: {}".format(arr_pattern.shape[0]))
            print("arr_pattern:\n{}".format(arr_pattern))
            
            print("\nlast_part.shape[0]: {}".format(last_part.shape[0]))
            print("last_part:\n{}".format(last_part))

    if False:
        s = arr[0]
        arr2 = [s]
        length = arr.shape[0]
        for i in range(1, length):
            s1 = s
            is_found = False
            for a in arr[i:]:
                s1 -= a+1
                if s1 < 1:
                    break
                if not s1 in arr2:
                    arr2.append(s1)
                    is_found = True
                    break

            if not is_found:
                s += arr[i]+1
                arr2.append(s)

            # print("is_found: {}, s: {}, s1: {}".format(is_found, s, s1))
            # input()
        print("arr2:\n{}".format(arr2))
