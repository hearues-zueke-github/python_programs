#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import numpy as np

from math import gcd

def sqnc_1():
    lst = []
    s = 0
    for m in range(1, 100+1):
        s = (s+(m-1)*2+(m-1)**2+1) % (m)
        print("m: {:3}, s: {}".format(m, s))
        lst.append(s)
    print("lst: {}".format(lst))


def sqnc_2():
    def get_sqnc(jump):
        lst = [0, 1]
        if gcd(jump, 2**10) > 1:
            # print("gcd(jump, 2**10) == jump!!!")
            return None
        i = 1
        while jump > len(lst):
            idx = jump%len(lst)
            lst_shift = lst[idx:]+lst[:idx]
            lst = lst+lst_shift
            # print("i: {}, lst_shift: {}".format(i, lst_shift))
            # print("lst: {}".format(lst))
            i += 1

        for i in range(i, 8):
            lst_shift = lst[jump:]+lst[:jump]
            # lst_shift = lst[-i:]+lst[:-i]
            lst = lst+lst_shift
            # print("i: {}, lst_shift: {}".format(i, lst_shift))
            # print("lst: {}".format(lst))
        return lst
    for jump in list(range(-40, 0))+list(range(1, 41)):
        lst = get_sqnc(jump)
        if isinstance(lst, type(None)):
            continue
        print("jump: {:3}, lst: {}".format(jump, lst))


def sqnc_3():
    # A000009
    arr = np.array([1], dtype=np.int)
    # i = 1

    for i in range(1, 101):
        arr = np.hstack((arr, np.zeros((i, ), dtype=np.int)))
        arr[i:] += arr[:-i]
        print("i: {}, arr:\n{}".format(i, arr[:i+1].tolist()))

def sqnc_4():
    n = 1000000
    d = {1: 4}

    steps = {1: 1}

    lens = []

    max_step = 0
    max_step_lst = []

    next_num = lambda n: n//2 if n%2 == 0 else n*3+1
    def do_next_num(n):
        if not n in d:
            n1 = next_num(n)
            do_next_num(n1)
            d[n] = n1
            steps[n] = steps[n1]+1

    for i in range(1, n+1):
        do_next_num(i)
        # while not n in d:
        #     n1 = next_num(n)
        #     d[n] = n1
        #     n = n1
        lens.append(len(d))

        if max_step < steps[i]:
            max_step = steps[i]
            max_step_lst.append(i)

        print("i: {}".format(i))
        print("d[i]: {}, steps[i]: {}".format(d[i], steps[i]))

    # print("d: {}".format(d))
    # print("lens: {}".format(lens))
    print("max_step_lst: {}".format(max_step_lst))
    globals()["d"] = d
    globals()["steps"] = steps
    globals()["lens"] = lens
    globals()["max_step_lst"] = max_step_lst


if __name__ == "__main__":
    # sqnc_1()    
    # sqnc_2()    

    # sqnc_3()
    sqnc_4()
