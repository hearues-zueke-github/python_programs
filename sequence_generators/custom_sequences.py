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

sys.path.append("../math_numbers")
import prime_numbers_fun

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

np.set_printoptions(threshold=sys.maxsize)

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


def get_sequence_A322670(n):
    def a(n):
        if n in a.vals: return a.vals[n]
        # elif n == 1: return 1
        # elif n < 1: return 0
        # a.vals[n] = f2(n, 0)
        a.vals[n] = f(n-1, 0)
        return a.vals[n]
    a.vals = {1: 1}
    
    def f(n, acc):
        t = (n, acc)
        if t in f.vals: return f.vals[t]
        if n >= 1:
            f.vals[t] = (a(n)+f(n-acc-a(n)-1, acc+a(n)+1)) % 10
            return f.vals[t]
        return 0
    f.vals = {(1, 0): 1}
    
    # def f2(n, acc):
    #     t = (n, acc)
    #     print("f2: {}, t: {}".format(f2, t))
    #     if t in f2.vals: return f2.vals[t]
    #     if n >= 1:
    #         f2.vals[t] = (f2(n-1, 0)+f2(n-1-acc-f2(n-1, 0)-1, acc+f2(n-1, 0)+1)) % 10
    #         return f2.vals[t]
    #     return 0
    # f2.vals = {(1, 0): 1}

    # print("f: {}".format(f))
    # print("f2: {}".format(f2))
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


def f_rec_1d_with_g_func(n, g, m=10):
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
        v = f(n-1, 0, 0)
        a.vals_count[n] = 1
        a.vals[n] = v
        return v
    def f(n, acc, acc2):
        t = (n, acc, acc2)
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
            if g:
                s = g(n, acc, acc2)
            else:
                a_n = a(n) # % m
            # print("n: {}, a_n: {}".format(n, a_n))

                s = (f(n-acc-1, acc+1)+a_n) % m

            # the original one!
                # s = (f(n-acc-a_n-1, acc+a_n+1)+a_n) % m
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

    g.a = a
    g.f = f

    a.vals = {1: 1}
    a.vals_count = {1: 0}
    f.vals = {}
    f.vals_count = {}
    next_num.vals = []

    vals = [next_num(j) for j in range(1, n+1)]
   # print("vals:\n{}".format(vals))

    return vals, next_num, a, f


def f_rec_with_saving_2d(n, m=10, start_acc=0, nr=1): #, choosen=0):
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
                a0 = an(y-1, x-1, acc)
                a1 = an(y-1, x-acc, acc+1)
                a2 = an(y, x-1-a1, acc+2)
                a3 = an(y-1-a1-a2, x, acc+3)

                v = (a0+a1+a2+a3+acc) % m
            elif nr == 3:
                a0 = an(y-1, x, acc+1)
                a1 = an(y-a0, x-1-a0, acc+2)
                a2 = an(y-2, x-a1, acc+3)
                a3 = an(y-1-a0-a1-a2, x-a0-acc, acc+1)
                
                v = (a0+a1+a2+a3) % m
            elif nr == 4:
                if y > x:
                    a0 = an(y-1, x-acc, acc+1)
                else:
                    a0 = an(y-1-acc, x-1, acc+1)
                a1 = an(y-1-a0, x-1-a0, acc+1)
                a2 = an(y-1-acc, x-1-acc, acc+1)                
                
                v = (a0+a1+a2) % m
            elif nr == 5:
                a0 = an(y, x-acc, acc+1)
                a1 = an(y-acc, x, acc+1)
                a2 = an(y-1, x, acc)
                a3 = an(y, x-1, acc)

                v = (a0+a1+a2+a3) % m
            elif nr == 6:
                if y % 2 == 0 and x % 2 == 0:
                    a00 = an(y-1, x-1, acc+1)
                    a01 = an(y-2, x-2, acc+1)
                    a02 = an(y-3, x-3, acc+1)
                else:
                    a00 = 0
                    a01 = 0
                    a02 = 0
                a0 = an(y-1, x, acc+1)
                a1 = an(y, x-1, acc+1)
                a2 = an(y-1, x-1, acc+1)

                a3 = an(y-acc, x-acc, acc+1)

                # a3 = an(y-a0-acc, x, acc+1)
                # a4 = an(y, x-a1-acc, acc+1)
                # a5 = an(y-a0-acc, x-a1-acc, acc+1)
                # a6 = an(y-a1-acc, x-a0-acc, acc+1)

                v = (a00+a01+a02+a0+a1+a2+a3+1) % m
                # v = (a00+a01+a02+a0+a1+a2+a3+a4+a5+a6+1) % m

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


def f_rec_1d_pos_neg(n, m=10):
    # hanoi_seq = get_hanoi_sequence(n)
    # print("hanoi_seq: {}".format(hanoi_seq))
    # div = 2
    def next_num(n):
        s = a(n)
        if n < 0:
            next_num.vals_i_neg.append(s)
        else:
            next_num.vals_i_pos.append(s)
        return s
    def a(n):
        if n in a.vals:
            a.vals_count[n] += 1
            return a.vals[n]
        if n == -1 or n == 0 or n == 1:
            return 1
        # elif n < 1:
        #     return 0
        v = f(n, 0)
        a.vals_count[n] = 1
        a.vals[n] = v
        return v
    def f(n, acc):
        t = (n, acc)
        if t in f.vals:
            f.vals_count[t] += 1
            return f.vals[t]
        # s = 0
        # if n >= 1:
        #     s = a(n)

            # if acc > 0:

        if n >= -1 and n <= 1:
            return a(n)

        if n < -1:
            s = (acc+f(n+acc+1+a(-acc), acc+1)+a(-acc)) % m
        else:
            s = (acc+f(n-acc-1-a(acc), acc+1)+a(acc)) % m

        f.vals_count[t] = 1
        f.vals[t] = s
        return s

    a.vals = {-1: 1, 0: 1, 1: 1}
    a.vals_count = {-1: 0, 0: 0, 1: 0}
    f.vals = {}
    f.vals_count = {}
    next_num.vals_i_pos = []
    next_num.vals_i_neg = []

    next_num(0)
    for j in range(1, n+1):
        next_num(j)
        next_num(-j)

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


def create_2d_images(n, m, suffix_dir_name='', nr=1):
    path_custom_2d_sequences = "images/custom_2d_sequences/{}x{}_m_{}{}/".format(n, n, m, '' if suffix_dir_name == '' else '_{}'.format(suffix_dir_name))
    if not os.path.exists(path_custom_2d_sequences):
        os.makedirs(path_custom_2d_sequences)

    resize = 2
    start_accs = range(3, -4, -1)
    for start_acc in start_accs:
    # for start_acc in range(0, -31, -1):
        arr, next_num, a, f = f_rec_with_saving_2d(n, m=m, start_acc=start_acc, nr=nr) #, choosen=1)
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


def do_1_2d_sequence(suffix_dir_name='', nr=1, start_acc=0):
    path_custom_2d_sequences = "images/custom_2d_sequences/{}x{}_m_{}{}/".format(n, n, m, '' if suffix_dir_name == '' else '_{}'.format(suffix_dir_name))
    if not os.path.exists(path_custom_2d_sequences):
        os.makedirs(path_custom_2d_sequences)

    resize = 2
    arr, next_num, a, f = f_rec_with_saving_2d(n, m=m, start_acc=start_acc, nr=nr) #, choosen=1)
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


def do_generic_1d_sequence(n, m, funcs_amount=1):
    def fa_primitive(n):
        if n == 1:
            return 1
        return 0

    # def count_a6():
    #     pass
    # count_a6.counter = 0

    using_numbers = [1, 1, 1, 1, 1, 1]

    def get_fa_fan(fa2, fa3, num=0):
        def fan(n, i, j):
            fan.counter += 1

            t = (n, i, j)
            if t in fan.vals:
                fan.vals_count[t] += 1
                return fan.vals[t]

            v = 0
            if n >= 1:
                a1, a2, a3, a4, a5, a6 = (0, )*6
                if using_numbers[0]:
                    a1 = fan(n-1, i, j)
                if using_numbers[1]:
                    a2 = fan(n-1-i-1, i+1, j)
                if using_numbers[2]:
                    a3 = fan(n-1-j-1, i, j+1)
                if using_numbers[3]:
                    a4 = fa2(n)
                if using_numbers[4]:
                    a5 = fa3(n)
                if using_numbers[5]:
                    a6 = fan(n-1-a4-1, i+1, j+1)
                # v = (a1+a2+a3+a4+a5) % m
                v = (a1+a2+a3+a4+a5+a6) % m
                # v = (a1+a4+a5+a6) % m
                # v = (a1+a2+a3+a4+a5+a6) % m
            fan.vals[t] = v
            fan.vals_count[t] = 1
            return v
        fan.vals = {(1, 0, 0): 1}
        fan.vals_count = {(1, 0, 0): 1}
        fan.counter = 0
        
        def fa(n):
            fa.counter += 1
            if n in fa.vals:
                fa.vals_count[n] += 1
                return fa.vals[n]

            v = fan(n, 0, 0)
            fa.vals_count[n] = 0
            fa.vals[n] = v
            return v

        # fa.vals = {0: 0, 1: fa2(num)}
        # fa.vals_count = {0: 0, 1: 0}
        # fa.vals = {0: 0, 1: fa2(num%m+1), 2: fa3((num+1)%m+1)}
        # fa.vals = {0: 0, 1: 1}
        fa.vals = {0: 0, 1: (fa2(num%m+1)+fa3(num%m+1))%m}
        fa.vals_count = {0: 0, 1: 0}
        fa.num = num
        fa.counter = 0
        fa.fa2 = fa2
        fa.fan = fan

        def fstart(n):
            # fa.counter2 = 0
            val = fa(n)
            # fa.vals_count[n] = fa.counter2
            return val

        fstart.fa = fa
        fstart.fan = fan

        return fstart

    fstart_prev2 = fa_primitive
    fstart_prev = fa_primitive
    fstart_lst = []

    for num in range(1, funcs_amount+1):
        fa = get_fa_fan(fstart_prev2, fstart_prev, num=num)
        for i in range(1, n): fa(i)

        print(' '*20, end='\r', flush=True)
        print('num: {}'.format(num), end='\r', flush=True)

        fstart_lst.append(fa)
        fstart_prev2 = fstart_prev
        fstart_prev = fa
        num += 1
    print()

    # for index, fa in enumerate(fstart_lst, 0):
    #     fan_counter = fa.fan.counter
    #     print("index: {}, fan_counter: {}".format(index, fan_counter))

    # print("count_a6.counter: {}".format(count_a6.counter))

    return fstart_lst


def van_eck_sequence(n):
    numbers = {0: 0}
    lst = [0, 0, 1]
    for j in range(2, n):
        v = lst[j]
        if not v in numbers:
            lst.append(0)
            numbers[v] = 0
        else:
            for k in range(1, j+1):
                if lst[-1-k]==v:
                    lst.append(k)
                    break
    return lst


def van_eck_sequence_2(n):
    numbers = {0: 0}
    lst = [0, 0, 1]
    j = 2
    while j < n:
        v = lst[j]
        if not v in numbers:
            lst.append(0)
            numbers[v] = 0
            j += 1
        else:
            for k in range(1, j+1):
                if lst[-1-k]==v:
                    lst.append(k)
                    j += 1
                    break
    return lst
 

def n3p1_sequence(n):
    lengths = {1: 0, 2: 0, 4: 0}
    new_lengths = []

    for i in range(3, n+1):
        if not i in lengths:
            v = i        
            counts = [v]
            while not v in lengths:
                if v % 2 != 0:
                    v = (v*3)+1
                    counts.append(v)
                v //= 2
                counts.append(v)
            c_prev = counts[-1]
            for c in counts[:-1][::-1]:
                lengths[c] = lengths[c_prev]+1
                c_prev = c
            # print("counts: {}".format(counts))
            new_lengths.append(len(counts)-1)

    return lengths, new_lengths


if __name__ == "__main__":
    lst = van_eck_sequence_2(10000)
    arr = np.array(lst)
    print("len(lst): {}".format(len(lst)))

    max_n = np.max(arr)

    print("max_n: {}".format(max_n))

    n = 10**7
    lengths, new_lengths = n3p1_sequence(n)
    nl_u, nl_uc = np.unique(new_lengths, return_counts=True)
    print("n: {}".format(n))
    print("nl_uc: {}".format(nl_uc))
    sys.exit(-1234567)
    # ps = prime_numbers_fun.get_primes(100)
    # print("ps: {}".format(ps))

    # print("np.cumsum(ps): {}".format(np.cumsum(ps).tolist()))

    # lst = [1]
    # for _ in range(1, 100):
    #     i = len(lst)-1
    #     j = 0
    #     s = 0
    #     while i >= 0:
    #         s += lst[i]
    #         i -= ps[j]
    #         j += 1
    #     # s %= 10
    #     lst.append(s)
    # print("lst: {}".format(lst))

    # sys.exit(0)


    n = 20
    m = 10

    # testing other 1d sequences!
    fstart_lst = do_generic_1d_sequence(n, m, funcs_amount=150)

    vals_lst = [list(fstart.fa.vals.values()) for fstart in fstart_lst]
    arr_orig = np.array(vals_lst, dtype=np.uint8).T
    print("arr_orig[:4]:\n{}".format(arr_orig[:4]))
    print("arr_orig[-4:]:\n{}".format(arr_orig[-4:]))

    arr = arr_orig.T.copy()
    arr *= 28

    resize = 1
    img = Image.fromarray(arr)
    img = img.resize((img.width*resize, img.height*resize))
    img.show()

    sys.exit(0)


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

    create_2d_images(n, m, suffix_dir_name='5', nr=1)
    sys.exit(0)

    # # arr_mod = arr%m
    # # print("arr_mod:\n{}".format(arr_mod))
    # with open("b322670.txt", "w") as fout:
    # # with open("b322670_f_with_recursion_mult_1.txt", "w") as fout:
    # # with open("b_sequence.txt", "w") as fout:
    #     for i, x in enumerate(arr, 1):
    #         fout.write("{} {}\n".format(i, x))

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
