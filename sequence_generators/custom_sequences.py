#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import functools
import os
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
        if n in a.vals: return a.vals[n]
        elif n == 1: return 1
        elif n < 1: return 0
        a.vals[n] = f(n-1, 0)
        return a.vals[n]
    def f(n, acc):
        t = (n, acc)
        if t in f.vals: return f.vals[t]
        if n >= 1:
            f.vals[t] = (a(n)+f(n-acc-a(n)-1, acc+a(n)+1)) % 10
            return f.vals[t]
        return 0
    a.vals = {1: 1}
    f.vals = {}
    return [a(i) for i in range(1, n+1)]


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
    # hanoi_seq = get_hanoi_sequence(n)
    # print("hanoi_seq: {}".format(hanoi_seq))
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
        if n >= 1:
            # # the sums of the sums!
            # for i in range(1, n+1):
            #     a_i = a(i)
            #     s = (s+f(i-acc-a_i-1, acc+a_i+1)+a_i) % m
            a_n = a(n) # % m
            # print("n: {}, a_n: {}".format(n, a_n))

            s = (f(n-acc-1, acc+1)+a_n) % m

            # the original one!
            # s = (f(n-acc-a_n-1, acc+a_n+1)+a_n) % m
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


def f_rec_with_saving_2d(n, m=10, start_acc=0): #, choosen=0):
    def next_num(y, x):
        # next_num.t = (y, x)
        # next_num.idx_y = y - 1
        # next_num.idx_x = x - 1
        # a.t_calls_dict[next_num.t] = 0
        # f.t_calls_dict[next_num.t] = 0

        # if choosen == 0:
        # s = a(y, x)
        # else:
        s = an(y, x, start_acc)
        
        # print("y: {}, x: {}".format(y, x))
        # print("  s: {}".format(s))
        # input()
        next_num.vals[(y, x)] = s
        next_num.vals_arr[y-1, x-1] = s
        return s

    def an(y, x, acc):
        t = (y, x, acc)
        if t in an.vals:
            an.vals_count[t] += 1
            return an.vals[t]
        elif y >= 1 and x >= 1:
            nr = 4
            if nr == 1:
                # sirpinski triangle, with many previous triangles too, if acc < -2
                # m = 2, or other maybe
                y1 = y-1
                x1 = x-1
                
                a_yx_21 = an(y1, x, 0)
                a_yx_22 = an(y, x1, 0)

                a_yx_11 = an(y1, x-acc-1, acc+1)
                a_yx_12 = an(y-acc-1, x1, acc+1)

                v = (a_yx_11+a_yx_12 + a_yx_21+a_yx_22) % m
            elif nr == 2:
                # upper right corner has a pattern
                a0 = an(y-1, x-1, start_acc)
                a1 = an(y-1, x-start_acc, start_acc+1)
                a2 = an(y, x-1-a1, start_acc+2)
                a3 = an(y-1-a1-a2, x, start_acc+3)

                v = (a0+a1+a2+a3+start_acc) % m
            elif nr == 3:
                a0 = an(y-1, x, start_acc+1)
                a1 = an(y-a0, x-1-a0, start_acc+2)
                a2 = an(y-2, x-a1, start_acc+3)
                a3 = an(y-1-a0-a1-a2, x-a0-start_acc, start_acc+1)
                
                v = (a0+a1+a2+a3) % m
            elif nr == 4:
                if y > x:
                    a0 = an(y-1, x, start_acc+1)
                else:
                    a0 = an(y, x-1, start_acc+1)
                a1 = an(y-1-a0, x-1-a0, start_acc+1)
                a2 = an(y-1-start_acc, x-1-start_acc, start_acc+1)                
                
                v = (a0+a1+a2) % m

            # a_yx = an(y1, x1, 0)

            # a_yx = an(y-acc-1, x-acc-1, acc*2+1)
            # a_yx_y = an(y-acc-1, x, acc**2+1)
            # a_yx_x = an(y, x-acc-1, acc**2+1)

            # a_yx1 = an(y-acc-1, x, acc+1)
            # a_yx2 = an(y, x-acc-1, acc+1)

            # a_yx11 = an(y-acc-1, x-a_yx2, acc+1)
            # a_yx12 = an(y-a_yx1, x-acc-1, acc+1)
            
            # a_yx11 = an(y-acc-1, x-a_yx1, acc+1)
            # a_yx12 = an(y-a_yx2, x-acc-1, acc+1)
            # a_yx21 = an(y-acc-1, x-a_yx2, acc+1)
            # a_yx22 = an(y-a_yx1, x-acc-1, acc+1)
            
            # v = (a_yx+a_yx1+a_yx2+a_yx11+a_yx12+a_yx21+a_yx22) % m
            # v = (a_yx_y+a_yx_x+a_yx) % m
            # v = (a_yx_y+a_yx_x+a_yx_1+a_yx_2) % m
            # v = (a_yx+a_yx_1+a_yx_2) % m
            # v = (a_yx_1+y_yx_2+a_yx+a_yx1+a_yx2+a_yx11+a_yx12) % m
            
            # v = (a_yx+an(y-1-acc-1, x-1, acc+1)+an(y-1, x-1-acc-1, acc+1)) % m
            # v = (a_yx+an(y-acc-1, x, acc+1)+an(y, x-acc-1, acc+1)) % m
        else:
            return 0
        an.vals_count[t] = 1
        an.vals[t] = v
        return v

    def a(y, x):
        # a.t_calls_dict[next_num.t] += 1
        # a.t_calls_arr[next_num.idx_y, next_num.idx_x] += 1
        t = (y, x)
        if t in a.vals:
            a.vals_count[t] += 1
            return a.vals[t]
        # elif y == x and y == 1:
        #     return 1
        elif y >= 1 and x >= 1:
            v1 = f(y-1, x-1, 0) % m
            # v2 = f(y-v1, x-1, 0) % m
            # v3 = f(y-1, x-v2, 0) % m
            
            # v4 = f(x-1, y, 0) % m
            # v5 = f(x, y-1, 0) % m
            # v6 = f(y-1-v4, x-1-v5, 0) % m

            v = (v1) % m
            # v = (v3+v6) % m
            
            # v = (f(y-1, x, 0)+f(y, x-1, 0)) # % m
            # v = (f(y-1, x, 0)+f(y-1, x-1, 0)+f(y, x-2, 0))# % m
            # v = (f(y-1, x, 0)+2*f(y-1, x-1, 0)+f(y, x-1, 0))# % m
        # elif y > x and x >= 1:
        #     v = fy(y-1, x, 0)
        # elif x > y and y >= 1:
        #     v = fx(y, x-1, 0)
        # elif y == x and y > 1:
        #     v = fd(y-1, x-1, 0)
        else:
            return 0
        a.vals_count[t] = 1
        a.vals[t] = v
        return v
    def f(y, x, acc):
        # f.t_calls_dict[next_num.t] += 1
        # f.t_calls_arr[next_num.idx_y, next_num.idx_x] += 1
        t = (y, x, acc)
        if t in f.vals:
            f.vals_count[t] += 1
            return f.vals[t]
        s = 0
        if y >= 1 and x >= 1:
            a_yx = a(y, x)
            # s = (a_yx+f(y-acc-1, x, acc+1)) % m
            # s = (a_yx+f(y, x-acc-1, acc+1)) % m
            # s = (a_yx) % m
            s = (a_yx+f(y-acc-1, x, acc+1)+f(y, x-acc-1, acc+1)) % m

            f.vals_count[t] = 1
            f.vals[t] = s
        return s
    # def fy(y, x, acc):
    #     t = (y, x, acc)
    #     if t in fy.vals:
    #         fy.vals_count[t] += 1
    #         return fy.vals[t]
    #     s = 0
    #     if y >= 1:
    #     # if y > x:
    #         a_yx = a(y, x)
    #         s = (a_yx+fy(y-acc-a_yx-1, x, acc+a_yx+1)) % m

    #         fy.vals_count[t] = 1
    #         fy.vals[t] = s
    #     return s
    # def fx(y, x, acc):
    #     t = (y, x, acc)
    #     if t in fx.vals:
    #         fx.vals_count[t] += 1
    #         return fx.vals[t]
    #     s = 0
    #     if x >= 1:
    #     # if x > y:
    #         # print("fx: t: {}".format(t))
    #         a_yx = a(y, x)
    #         s = (a_yx+fx(y, x-acc-a_yx-1, acc+a_yx+1)) % m

    #         fx.vals_count[t] = 1
    #         fx.vals[t] = s
    #     return s
    # def fd(y, x, acc):
    #     # print("y: {}, x: {}".format(y, x))
    #     t = (y, x, acc)
    #     if t in fd.vals:
    #         fd.vals_count[t] += 1
    #         return fd.vals[t]
    #     s = 0
    #     if y >= 1 and x >= 1:
    #     # if y >= 1 and x >= 1:
    #         # print("fd: t: {}".format(t))
    #         a_yx = a(y, x)
    #         s = (a_yx+fd(y-acc-a_yx-1, x-acc-a_yx-1, acc+a_yx+1)) % m
    #         # s = (a_yx+fd(y-acc-a_yx-1-1, x-acc-a_yx-1, acc+a_yx+1)) % m

    #         fd.vals_count[t] = 1
    #         fd.vals[t] = s
    #     return s

    # def f():
    #     pass

    a.t_calls_dict = {}
    a.t_calls_arr = np.zeros((n, n), dtype=np.int)
    f.t_calls_dict = {}
    f.t_calls_arr = np.zeros((n, n), dtype=np.int)

    an.vals = {(1, 1, 0): 1}
    an.vals_count = {(1, 1, 0): 0}

    a.vals = {(1, 1): 1}
    a.vals_count = {(1, 1): 0}
    f.vals = {}
    f.vals_count = {}
    # fy.vals = {}
    # fy.vals_count = {}
    # fx.vals = {}
    # fx.vals_count = {}
    # fd.vals = {}
    # fd.vals_count = {}
    next_num.vals = {}
    next_num.vals_arr = np.zeros((n, n), dtype=np.int)

    # vals = [next_num(j) for j in range(1, n+1)]
    # print("vals:\n{}".format(vals))

    for i in range(1, n+1):
        print("i: {}".format(i))
        for j in range(1, n+1):
            next_num(j, i)

    return next_num.vals_arr, next_num, a, f
    # return next_num.vals_arr, next_num, a, fy, fx, fd
    # return next_num.vals, next_num, a, f


def f_rec_mult_2(n, m=10):
    # hanoi_seq = get_hanoi_sequence(n)
    # print("hanoi_seq: {}".format(hanoi_seq))
    div = 2
    def next_num(n):
        s = a(n)
        next_num.vals.append(s)
        return s
    def a(n):
        if n in a.vals:
            a.vals_count[n] += 1
            return a.vals[n]
        if n == 1:
            return 1
        elif n < 1:
            return 0
        v = f(n-1, (n-1)//div, div)
        a.vals_count[n] = 1
        a.vals[n] = v
        return v
    def f(n, acc, div):
        t = (n, acc, div)
        if t in f.vals:
            f.vals_count[t] += 1
            return f.vals[t]
        s = 0
        if n >= 1:
            s = a(n)

            if acc > 0:
                s = (s+f(n-acc, acc//div, div)) % m

            f.vals_count[t] = 1
            f.vals[t] = s
        return s

    a.vals = {1: 1}
    a.vals_count = {1: 0}
    f.vals = {}
    f.vals_count = {}
    next_num.vals = []

    for j in range(1, n+1):
        next_num(j)

    return next_num.vals, next_num, a, f


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


def get_1d_sequence_from_2d(arr):
    assert len(arr.shape) == 2
    rows, cols = arr.shape[:2]

    nums = []
    for i in range(1, rows+1):
        ys = np.arange(0, i)
        xs = np.arange(i-1, -1, -1)
        nums.extend(arr[ys, xs].tolist())

    return nums


if __name__ == "__main__":
    n = 300
    # lst = f_mult_1(n)
    m = 4
    # lst = f_jumping_modulo(n, m=m)

    # lst, next_num, a, fy, fx, fd = f_rec_with_saving_2d(n, m=m)
    # arr3, next_num, a, f = f_rec_with_saving_2d(n, m=m, choosen=0)
    

    # lst, next_num, a, f = f_rec_mult_2(n, m=m)
    # lst_hanoi = get_hanoi_sequence(5)
    # lst = f_sum_increase_range(n)
    # lst = jumping_sum_index_sequence(n)
    
    # lst = get_sequence_better(n)
    # lst = get_sequence(n)

    # print("lst:\n{}".format(",".join(list(map(str, lst)))))
    # print("n: {}, m: {}".format(n, m))
    # print("lst:\n{}".format(lst))
    # sys.exit(0)
    
    # lst_other, _, _, _ = f_rec_mult_1(n, m=m)
    # # lst, next_num, a, f = f_rec_mult_1(n, m=m)
    # print("lst_other:\n{}".format(lst_other))

    # lst_with_recursion = f_with_recursion_mult_1(n)
    # print("lst_with_recursion:\n{}".format(lst_with_recursion))

    # seq_1d_fied = get_1d_sequence_from_2d(arr)
    # print("seq_1d_fied:\n{}".format(seq_1d_fied))

    # arr = np.array(lst)

    
    path_custom_2d_sequences = "images/custom_2d_sequences/{}x{}_m_{}_2/".format(n, n, m)
    if not os.path.exists(path_custom_2d_sequences):
        os.makedirs(path_custom_2d_sequences)

    resize = 2
    start_accs = range(3, -4, -1)
    for start_acc in start_accs:
    # for start_acc in range(0, -31, -1):
        arr, next_num, a, f = f_rec_with_saving_2d(n, m=m, start_acc=start_acc) #, choosen=1)
        print("arr:\n{}".format(arr))

        arr2 = (arr*(256//m)).astype(np.uint8)
        new_size = (arr2.shape[1]*resize, arr2.shape[0]*resize)
        img = Image.fromarray(arr2).resize(new_size)
        file_name_suffix = str(start_acc)
        if "-" in file_name_suffix:
            file_name_suffix = file_name_suffix.replace("-", "neg_")
        else:
            file_name_suffix = "pos_"+file_name_suffix

        img.save(path_custom_2d_sequences+"sequence_2d_start_acc_{}.png".format(file_name_suffix))

    # Image.fromarray((((arr!=arr.T)+0)*255).astype(np.uint8)).resize(new_size).show()

    sys.exit(0)

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
