#! /usr/bin/python3.5

import decimal
import math
import select
import sys

import numpy as np

import matplotlib.pyplot as plt

# from decimal import Decimal as D
from functools import reduce
from time import time

# decimal.getcontext().prec = 2000

num_to_str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
str_to_num = {s: n for n, s in enumerate(num_to_str)}

def get_random_base_number(b, length):
    return "".join([num_to_str[np.random.randint(0, b)] for _ in range(0, length)])

def gcd(a, b):
    if a < b:
        a, b = b, a

    while b > 0:
        t = a % b
        a = b
        b = t

    return a

def calc_num_to_base(i, b):
    l = []
    while i > 0:
        l.append(i%b)
        i //= b
    l = l[::-1]
    return "".join(map(lambda x: num_to_str[x], l))

def digit_diff(n, b):
    l = list(map(str_to_num.get, n))
    l_new = [n[0]]+list(map(lambda x, y: num_to_str[(x*(b+1)-y)%b], l[1:], l[:-1]))
    return "".join(l_new)

def add_two_str_num(str1, str2, b):
    return calc_num_to_base(int(str1, b)+int(str2, b), b)

def test_implementation_difference():
    iters = 100000
    num = "1234567890ABCDEF"

    start = time()
    for _ in range(0, iters):
        _ = calc_num_to_int(num, 16)
        _ = calc_num_to_int(num, 17)
        _ = calc_num_to_int(num, 18)
        _ = calc_num_to_int(num, 19)
        _ = calc_num_to_int(num, 20)
    time_own_function = time()-start

    start = time()
    for _ in range(0, iters):
        _ = int(num, 16)
        _ = int(num, 17)
        _ = int(num, 18)
        _ = int(num, 19)
        _ = int(num, 20)
    time_int_function = time()-start

    print("time needed for own function for {} iters: {} s".format(iters, time_own_function))
    print("time needed for int function for {} iters: {} s".format(iters, time_int_function))

def test_str_adding():
    str1 = "1253499"
    str2 = "5432189"
    print("str1: {}".format(str1))
    print("str2: {}".format(str2))

    str3 = add_two_str_num(str1, str2, 10)
    print("base 10, str3: {}".format(str3))

    str3 = add_two_str_num(str1, str2, 11)
    print("base 11, str3: {}".format(str3))

    str3 = add_two_str_num(str1, str2, 12)
    print("base 12, str3: {}".format(str3))


a = "129111258974879546"
print("a: {}".format(a))
lengths = []
for b in range(10, 17):
    n = a
    for i in range(0, 5000):
        n = digit_diff(n, b)
        # print("b: {}, i: {:5}, n: {}".format(b, i, n))
        if n == a:
            break
    lengths.append(i)
    # print("")

print("lengths: {}".format(lengths))

a = int(a)
num_base_10 = calc_num_to_base(a, 10)
print("num_base_10: {}".format(num_base_10))
num_base_11 = calc_num_to_base(a, 11)
print("num_base_11: {}".format(num_base_11))
num_base_12 = calc_num_to_base(a, 12)
print("num_base_12: {}".format(num_base_12))
num_base_16 = calc_num_to_base(a, 16)
print("num_base_16: {}".format(num_base_16))

print("int(num_base_10, 10): {}".format(int(num_base_10, 10)))
print("int(num_base_11, 11): {}".format(int(num_base_11, 11)))
print("int(num_base_12, 12): {}".format(int(num_base_12, 12)))
print("int(num_base_16, 16): {}".format(int(num_base_16, 16)))

b = 16
digits = 2
i = "1"+"0"*(digits-1)
# m = {}
d = []
# for _ in range(0, b**(digits-1)):
for _ in range(0, b**digits-b**(digits-1)):
    # d[i] = digit_diff(i, b)
    if gcd(int(i[0], b), b) == 1:
        d.append((i, digit_diff(i, b)))
    i = add_two_str_num(i, "1", b)
# m[b] = d
print("d: {}".format(d))

with open("graph.dot", "w") as f:
    f.write("digraph g\n{\n")

    for x, _ in d:
        f.write("a{} [label=\"{}\"]".format(x, x))

    f.write("\n")

    for x, y in d:
        f.write("    a{} -> a{}\n".format(x, y))

    f.write("}\n")

lengths = []
for b in range(2, 17):
    print("b: {}".format(b))
    lengths_base = []
    for digits in range(2, 101, 10):
        print("digits: {}".format(digits))
        lengths_digit = []
        max_i = "0"
        max_iters = 0

        i = "1"+"0"*(digits-1)
        # for _ in range(0, 2):
        # for _ in range(b**(digits-1), b**digits):
            # i_rand = get_random_base_number(b, digits)
            # if gcd(int(i_rand[0], b), b) == 1:
            # if gcd(int(i[0], b), b) == 1:
                # j = digit_diff(i_rand, b)
        j = digit_diff(i, b)

        iters = 1
        while j != i:
        # while j != i_rand:
            iters += 1
            j = digit_diff(j, b)

        if max_iters < iters:
            # max_i = i_rand
            max_i = i
            max_iters = iters

            # i = add_two_str_num(i, "1", b)

        lengths_base.append((b, digits, max_i, max_iters))
        
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            break


    lengths.append(lengths_base)

# print("lengths: {}".format(lengths))

for lengths_base in lengths:
    for b, digits, max_i, max_iters in lengths_base:
        print("b: {}, digits: {}, max_iters: {}".format(b, digits, max_iters))
    print("")
