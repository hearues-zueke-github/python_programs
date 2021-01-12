#! /usr/bin/python3 -i

# -*- coding: utf-8 -*-

globals().clear()
print('\nDoing Test Nr. 1:')

# Test 1

from copy import deepcopy

b = 2
def foo(a):
    c = a + b
    return c

print("foo(2): {}".format(foo(2)))

bar = deepcopy(foo)

print("foo(3): {}".format(foo(3)))
print("bar(3): {}".format(bar(3)))

b = 400

print("foo(4): {}".format(foo(4)))
print("bar(4): {}".format(bar(4)))

globals().clear()
print('\nDoing Test Nr. 2:')

# Test 2

from copy import deepcopy

d_glob = {'b': 2}
d_loc = {}
func_str = """
def foo(a):
    c = a + b
    return c
"""
exec(func_str, d_glob, d_loc)

foo = d_loc['foo']

print("foo(2): {}".format(foo(2)))

bar = deepcopy(foo)

print("foo(3): {}".format(foo(3)))
print("bar(3): {}".format(bar(3)))

d_glob['b'] = 400

print("foo(4): {}".format(foo(4)))
print("bar(4): {}".format(bar(4)))

globals().clear()
print('\nDoing Test Nr. 3:')

# Test 3

from copy import deepcopy

d_glob_1 = {'b': 2}
d_glob_2 = {'b': 2}
d_loc = {}
func_str = """
def foo(a):
    c = a + b
    return c
"""

exec(func_str, d_glob_1, d_loc)
foo = d_loc['foo']


print("foo(2): {}".format(foo(2)))

exec(func_str, d_glob_2, d_loc)
bar = d_loc['foo']

print("foo(3): {}".format(foo(3)))
print("bar(3): {}".format(bar(3)))

d_glob_1['b'] = 400

print("foo(4): {}".format(foo(4)))
print("bar(4): {}".format(bar(4)))

globals().clear()
print('\nDoing Test Nr. 4:')

# Test 4

from copy import deepcopy
from types import FunctionType

b = 2
def foo(a):
    c = a + b
    return c

print("foo(2): {}".format(foo(2)))

bar = FunctionType(foo.__code__, deepcopy(foo.__globals__), foo.__name__, foo.__defaults__, foo.__closure__)

print("foo(3): {}".format(foo(3)))
print("bar(3): {}".format(bar(3)))

b = 400

print("foo(4): {}".format(foo(4)))
print("bar(4): {}".format(bar(4)))

globals().clear()

