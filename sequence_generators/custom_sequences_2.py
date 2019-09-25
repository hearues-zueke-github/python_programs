#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import sys

from copy import deepcopy

sys.path.append("../combinatorics/")
import different_combinations as combinations
sys.path.append("../math_numbers/")
from prime_numbers_fun import get_primes

from time import time
from functools import reduce

from PIL import Image

import numpy as np

sys.path.append("../")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

from PIL import ImageTk
from tkinter import Tk, Label, BOTH
from tkinter.ttk import Frame, Style

import utils

def time_measure(f, args):
    start_time = time()
    ret = f(*args)
    end_time = time()
    diff_time = end_time-start_time
    return ret, diff_time

def get_a():
    def a(n):
        if n < 1:
            return 0
        elif n == 1:
            return 1

        i = 0
        if n % 2 == 0:
            i += a(n//2)+1
        if n % 3 == 0:
            i += a(n//3)+2
        if n % 5 == 0:
            i += a(n//5)+3
        if n % 7 == 0:
            i += a(n//7)+4
        if n % 11 == 0:
            i += a(n//11)+5
        if n % 13 == 0:
            i += a(n//13)+6
        if n % 17 == 0:
            i += a(n//17)+7
        if n % 19 == 0:
            i += a(n//19)+8
        if n % 23 == 0:
            i += a(n//23)+9
        if n % 29 == 0:
            i += a(n//29)+10
        if n % 31 == 0:
            i += a(n//31)+11
        if n % 37 == 0:
            i += a(n//37)+12
        return i
    return a


def get_a2(m):
    def a(n):
        if n in a.vals: return a.vals[n]
        elif n <= 0: return 0
        s = 0
        i = n
        while i > 0:
            c = a(i-s-1)
            s = s+c+1
            i = i-s-1
        s %= m
        a.vals[n] = s
        return s
    a.vals = {1: 1}
    return a


def get_a3(m):
    def a(n):
        if n in a.vals: return a.vals[n]
        elif n <= 0: return 0
        v = a1(n, 0)
        a.vals[n] = v
        return v
    def a1(n, acc):
        t = (n, acc)
        if t in a1.vals: return a1.vals[t]
        elif n <= 0: return 0
        n1 = n-acc-1
        c1 = a1(n1,0)
        v = (a1(n1-c1-1,acc+c1+1)+c1+1)%m
        a1.vals[t] = v
        return v
    a.vals = {1: 1}
    a1.vals = {(1, 0): 1}
    a.a1 = a1
    return a


def get_a4(m):
    def a(n):
        if n in a.vals: return a.vals[n]
        elif n <= 0: return 0
        v = a1(n, 0)
        a.vals[n] = v
        return v
    def a1(n, acc):
        t = (n, acc)
        if t in a1.vals: return a1.vals[t]
        elif n <= 0: return 0
        n1 = n-acc-1
        c1 = a1(n1,0)
        c2 = a1(n1-c1-1,0)
        v = (a1(n1-c1-1,acc+c1+1)+c1+1+c2)%m
        a1.vals[t] = v
        return v
    a.vals = {1: 1}
    a1.vals = {(1, 0): 1}
    a.a1 = a1
    return a


def get_a5(m):
    def a(n):
        if n in a.vals: return a.vals[n]
        elif n <= 0: return 0
        v = a1(n, 0)
        a.vals[n] = v
        return v
    def a1(n, acc):
        t = (n, acc)
        if t in a1.vals: return a1.vals[t]
        elif n <= 0: return 0
        n1 = n-acc-1
        c1 = a1(n1,0)
        c2 = a1(n1-c1-1,0)
        c3 = a1(n1-c2-1,c1)
        c4 = a1(n1-c2-1-c3,c1+c2+c3+1)
        v = (a1(n1-c1-1,acc+c1+1+c4)+c1+1+c2+1+c3+1)%m
        a1.vals[t] = v
        return v
    a.vals = {1: 1}
    a1.vals = {(1, 0): 1}
    a.a1 = a1
    return a


def get_a6(m):
    def a(n):
        if n in a.vals: return a.vals[n]
        elif n <= 0: return 0
        v = a1(n, 0)
        a.vals[n] = v
        return v
    def a1(n, acc):
        t = (n, acc)
        if t in a1.vals: return a1.vals[t]
        elif n <= 0: return 0
        c1 = a1(n-1,0)
        c2 = a1(n-2,0)
        c3 = a1(n-c1-c2-1,0)
        # c4 = a1(n1-c2-1-c3,c1+c2+c3+1)
        v = (a1(n-c1-c2-c3-1-acc,acc+c1+1)+c1+c2+c3+1+acc)%m
        # v = (a1(n1-c1-1,acc+c1+1+c4)+c1+1+c2+1+c3+1)%m
        a1.vals[t] = v
        return v
    a.vals = {1: 1}
    a1.vals = {(1, 0): 1}
    a.a1 = a1
    return a


if __name__ == "__main__":
    m = 12
    print("m: {}".format(m))
    nn = 10000
    a1 = get_a2(m)
    lst1 = [a1(i) for i in range(1, nn+1)]
    a2 = get_a3(m)
    lst2 = [a2(i) for i in range(1, nn+1)]

    arr1 = np.array(lst1)
    arr2 = np.array(lst2)
    
    print("arr1[:100]: {}".format(arr1[:100]))
    print("arr2[:100]: {}".format(arr2[:100]))
    
    assert np.all(arr1==arr2)
    
    # def f(a, nn):
    #     return [a(i) for i in range(1, nn+1)]

    # nn = 20000
    # print("Starting f1...")
    # r1, diff_time_1 = time_measure(f, (a1, nn))
    # print("Starting f2...")
    # r2, diff_time_2 = time_measure(f, (a2, nn))

    # print("diff_time_1: {}".format(diff_time_1))
    # print("diff_time_2: {}".format(diff_time_2))

    # m = 10

    nn = 36000
    a3 = get_a4(m)
    lst3 = [a3(i) for i in range(1, nn+1)]
    arr3 = np.array(lst3)
    print("arr3:\n{}".format(arr3))
    a4 = get_a5(m)
    lst4 = [a4(i) for i in range(1, nn+1)]
    arr4 = np.array(lst4)
    print("arr4:\n{}".format(arr4))

    arr5 = np.tile(np.arange(0, m), 100)

    factors1 = utils.get_sequence_randomness_analysis(m, arr1)
    factors2 = utils.get_sequence_randomness_analysis(m, arr2)
    factors3 = utils.get_sequence_randomness_analysis(m, arr3)
    factors4 = utils.get_sequence_randomness_analysis(m, arr4)
    factors5 = utils.get_sequence_randomness_analysis(m, arr5)

    print("factors1: {}".format(factors1))
    print("factors2: {}".format(factors2))
    print("factors3: {}".format(factors3))
    print("factors4: {}".format(factors4))
    print("factors5: {}".format(factors5))

    for m in range(2, 16):
        func_a = get_a6(m)
        lst = [func_a(i) for i in range(1, nn+1)]
        arr = np.array(lst)
        # print("arr[:50]:\n{}".format(arr[:50]))
        counts, factors = utils.get_sequence_randomness_analysis(m, arr)
        print("m: {}, factors: {}, max_factors: {}".format(m, factors, nn/m*np.sqrt(m-1)))
