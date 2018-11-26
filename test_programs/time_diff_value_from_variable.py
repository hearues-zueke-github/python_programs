#! /usr/bin/python3.5

import time

import numpy as np

def get_time(f, args):
    start_time = time.time()
    ret = f(*args)
    end_time = time.time()
    return ret, end_time-start_time

def f1(n):
    for _ in range(0, n):
        pass
    return None

def f2(n):
    a = 2
    for _ in range(0, n):
        a
    return None

def f3(n):
    arr = [0]
    idx = 0
    for _ in range(0, n):
        arr[idx]
    return None

def f4(n):
    arr = [0]
    idx = 0
    for _ in range(0, n):
        arr[arr[idx]]
    return None

def f5(n):
    arr = [0]
    idx = 0
    for _ in range(0, n):
        arr[arr[arr[idx]]]
        arr[arr[arr[idx]]]
    return None

def f6(n):
    arr = [0]
    idx = 0
    for _ in range(0, n):
        arr[arr[arr[arr[idx]]]]
    return None

if __name__ == "__main__":
    args = (5000000, )


    # ret1, diff1 = get_time(f1, args)
    # ret2, diff2 = get_time(f2, args)
    # ret3, diff3 = get_time(f3, args)
    # ret4, diff4 = get_time(f4, args)

    # print("diff1: {:.3f}".format(diff1))
    # print("diff2: {:.3f}".format(diff2))
    # print("diff3: {:.3f}".format(diff3))
    # print("diff4: {:.3f}".format(diff4))

    str1 = "ret{}, diff{} = get_time(f{}, args)"
    str2 = "print(\"diff{}: {{:.3f}}\".format(diff{}))"
    for i in range(1, 7):
        exec(str1.format(i, i, i))
        exec(str2.format(i, i))
