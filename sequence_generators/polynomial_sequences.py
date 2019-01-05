#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import functools
import sys
import time

import numpy as np

from indexed import IndexedOrderedDict

from dotmap import DotMap
from PIL import Image, ImageTk

import multiprocessing as mp
from multiprocessing import Process, Pipe, Lock

import utils_sequence

def get_f(m):
    local = {}
    parameters = np.random.randint(0, m, (6, ))
    # parameters = [5, 1, 3]
    calc_part_lst = []
    for i, a in enumerate(parameters, 0):
        if a > 0:
            calc_part_lst.append("{}*x**{}".format(a, i))

    calc_part = "+".join(calc_part_lst)
    f_str = "def f(x):\n return ({}) % m\n".format("x" if calc_part == "" else calc_part)
    print("f_str:\n{}".format(f_str))
    exec(f_str, {"m": m}, local)
    return local["f"], parameters


def f_iter(n, f):
    lst = [f(i) for i in range(0, n)]
    return np.array(lst)

def f_iter_gen(n, g):
    lst = list(g.values)+[g() for i in range(0, n)]
    # lst = [g.__next__() for i in range(0, n)]
    # lst = [next(g.__next__) for i in range(0, n)]
    return np.array(lst)


def get_g_2(m, values):
    def pol_func(x):
        return (x**2*3+x*4+6) % m
    def g(x=0):
        idxs = []
        vals = []

        j = g.i
        g.i += 1

        s = x
        acc = 0
        jumps = 0

        while j >= 0:
            y = values[j]
            idxs.append(j)
            vals.append(y)

            s = (s+y) % m
            # acc += y+1
            acc += pol_func(y)+1
            j -= acc
            # j -= pol_func(y)+1

        g.all_idxs.append(idxs)
        g.all_vals.append(vals)
        values.append(s)
        
        return s
    g.i = len(values) - 1
    g.m = m
    g.values = values
    g.all_idxs = []
    g.all_vals = []
    return g


def create_func_f(x, y, z, m, params, max_jumps=-1):
    def get_g(m, values):
        def g(x=0):
            idxs = []
            vals = []

            j = g.i
            g.i += 1

            s = x
            acc = 0
            jumps = 0

            while j >= 0:
                y = values[j]
                idxs.append(j)
                vals.append(y)

                s = (s+y) % m
                acc += y+1
                j -= acc

                if max_jumps > 0 and jumps >= max_jumps:
                    break
                jumps += 1

            g.all_idxs.append(idxs)
            g.all_vals.append(vals)
            values.append(s)
            
            return s
        g.i = len(values) - 1
        g.m = m
        g.values = values
        g.all_idxs = []
        g.all_vals = []
        return g
    g = get_g(m, [1])

    rows, cols = params.shape

    str_get_vals = "        {} = {}\n".format(
            ", ".join(["a{}".format(i) for i in range(1, cols)]),
            ", ".join(["f.a{}".format(i) for i in range(1, cols)]),
        )
    str_set_vals = "    {} = {}\n".format(
            ", ".join(["f.a{}".format(i) for i in range(1, cols)]),
            ", ".join(["a{}".format(i) for i in range(1, cols)]),
        )
    str_temp_2 = "        f.a{{}} = ({{}}+{}) % f.m\n".format(
            "+".join(["a{}*{{}}".format(i) for i in range(1, cols)])
        )
    str_calc_values = "".join([str_temp_2.format(i+1, *params[i]) for i in range(0, rows)])

    str_ret_val = "        return f.a{}\n".format(1)
    # str_ret_val = "        return f.a{}\n".format(cols-1)

    func_str = (
        "def get_f():\n"+
        "    def f():\n"+
        "        v = f.h()\n"+
        "        f.values.append(v)\n"+
        "        return v\n"+
        "    def h():\n"+
        str_get_vals+
        str_calc_values+
        str_ret_val+
        "    def get_gen():\n"+
        "        v = h()\n"+
        "        return v\n"+
        str_set_vals+
        "    f.m = m\n"+
        "    f.g = g\n"+
        "    f.values = []\n"+
        "\n"+
        "    f.h = h\n"
        "    f.__next__ = get_gen\n"
        "\n"+
        "    return f\n"+
        "f = get_f()\n"
    )
    
    variables = {"a{}".format(i): 0 for i in range(1, cols)}
    scope = {"m": m, "g": g}
    whole_scope = {**variables, **scope}
    loc = {}
    exec(func_str, whole_scope, loc)
    f = loc["f"]

    return f


def get_polynomial_modulo_sequence(params, modulo):
    len_params = len(params)
    func_str = "f = lambda x: ("
    lst_func_part = []
    if len_params >= 1:
        lst_func_part.append("{}".format(params[0]))
    if len_params >= 2:
        lst_func_part.append("{}*x".format(params[1]))
    for i in range(2, len_params):
        lst_func_part.append("{}*x**{}".format(params[i], i))
    func_str += "+".join(lst_func_part)+") % m\n"
    print("func_str: {}".format(func_str))

    loc = {}
    exec(func_str, {"m": modulo}, loc)
    f = loc["f"]
    # print("f: {}".format(f))
    ys = [f(i) for i in range(0, modulo)]
    # print("ys: {}".format(ys))
    dm = DotMap()
    dm.ys = ys
    dm.params = params
    dm.modulo = modulo
    dm.f = f

    return dm


def get_random_double_polynomial_modulo_sequence(len_params, modulo):
    fails = 0
    while True:
        params = np.random.randint(0, m, (len_params, ))
        is_inplace_shifted = 1
        # is_inplace_shifted = int(np.random.randint(0, 2))
        params[0] = 0
        lst_func_part = []
        if len_params >= 1:
            lst_func_part.append("{}".format(params[0]))
        if len_params >= 2:
            lst_func_part.append("{}*x".format(params[1]))
            # lst_func_part.append("{}*{}".format(params[1], "x" if params_shift[1] == 0 else "(x+{})".format(params_shift[1])))
        for i in range(2, len_params):
            lst_func_part.append("{}*x**{}".format(params[i], i))
        
        func_str = "f = lambda x: ({}) % m"
        if is_inplace_shifted:
            func_str = func_str.format("+".join(lst_func_part))
        else:
            func_str = func_str.format("(lambda x: x+((x)%2))("+"+".join(lst_func_part)+")")
        # print("func_str: {}".format(func_str))

        loc = {}
        exec(func_str, {"m": modulo}, loc)
        f = loc["f"]
        ys = [f(i) for i in range(0, modulo)]

        if np.all(np.sort(ys)==np.arange(0, modulo)):
            break
        fails += 1
        # print("fails: {}".format(fails))
    dm = DotMap()
    dm.ys = ys
    dm.func_str = func_str
    dm.params = tuple(params.tolist())
    # dm.params_shift = tuple(params_shift.tolist())
    dm.is_inplace_shifted = is_inplace_shifted
    dm.modulo = modulo
    dm.f = f

    return dm
        

if __name__ == "__main__":
    # n = 1000
    len_params = 4
    m = 8

    dms_params = {}
    dms_ys = {}
    for i in range(0, 10000):
        dm = get_random_double_polynomial_modulo_sequence(len_params, m)
        if not dm.params in dms_params:
            dms_params[dm.params] = dm
            print("dm.params: {}".format(dm.params))
            print("dm.ys: {}".format(dm.ys))
            print("dm.func_str: {}".format(dm.func_str))
        ys = tuple(dm.ys)
        if not ys in dms_ys:
            dms_ys[ys] = dm
        # else:
            # print("ALREADY! dm.params: {}".format(dm.params))

    keys_params = sorted(list(dms_params.keys()))
    print("keys_params:\n{}".format(keys_params))
    print("len_params: {}, m: {}".format(len_params, m))
    print("len(keys_params): {}".format(len(keys_params)))
    
    keys_ys = sorted(list(dms_ys.keys()))
    print("len(keys_ys): {}".format(len(keys_ys)))

    # print("dm.func_str: {}".format(dm.func_str))
    # get_polynomial_modulo_sequence([3, 5, 4, 6], 10)

    # g = get_g_2(m, [1])
    # # params = np.random.randint(0, m, (4, 5))
    # # f = create_func_f(0, 0, 0, m, params, max_jumps=-1)

    # arr = f_iter_gen(n, g)
    # arr_pattern, idx = utils_sequence.find_first_repeat_pattern(arr)

    # print("arr.tolist():\n{}".format(arr.tolist()))
    # print("idx: {}".format(idx))
    # print("arr_pattern:\n{}".format(arr_pattern))
