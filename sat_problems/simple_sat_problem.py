#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import random
import sys

from pysat.solvers import Glucose3

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

from pycryptosat import Solver

if __name__ == "__main__":
    print("Hello World!")

    def expr_1(a, b, c):
        return (a and not b and not c) or (not a and b and not c) or (not a and not b and c)
    def expr_2(a, b, c):
        return (a or b or c) and (not a or not b) and (not a or not c) and (not b or not c)

    params = [(a, b, c) for a in (True, False) for b in (True, False) for c in (True, False)]
    lst1 = [expr_1(*p) for p in params]
    lst2 = [expr_2(*p) for p in params]

    print("lst1: {}".format(lst1))
    print("lst2: {}".format(lst2))


    corner_positions = [
        [(1, 1, 1), (4, 2, 1), (6, 1, 2)],
        [(1, 1, 2), (5, 1, 1), (4, 2, 2)],
        [(1, 2, 2), (3, 1, 2), (5, 2, 1)],
        [(1, 2, 1), (6, 2, 2), (3, 1, 1)],
        [(2, 1, 1), (3, 2, 1), (6, 2, 1)],
        [(2, 1, 2), (5, 2, 2), (3, 2, 2)],
        [(2, 2, 2), (4, 1, 2), (5, 1, 2)],
        [(2, 2, 1), (6, 1, 1), (4, 1, 1)],
    ]

    colors = [1, 2, 3, 4, 5, 6]

    corner_colors = [
        (1, 4, 6),
        (1, 6, 3),
        (1, 3, 5),
        (1, 5, 4),
        (2, 3, 6),
        (2, 6, 4),
        (2, 4, 5),
        (2, 5, 3),
    ]

    corner_positions = [[tuple(i-1 for i in t) for t in l] for l in corner_positions]
    colors = [i-1 for i in colors]
    corner_colors = [tuple(i-1 for i in t) for t in corner_colors]

    # create clauses for each cell field for each color once!

    cnfs = []

    for corner_pos in corner_positions:
        for c_pos in corner_pos:
            prefix_num = c_pos[0]*2*2*6+c_pos[1]*2*6+c_pos[2]*6+1
            variable_names = [prefix_num+c for c in colors]
            # print("c_pos: {}".format(c_pos))
            # print("prefix_num: {}".format(prefix_num))
            # print("variable_names: {}".format(variable_names))
            # input("ENTER...")
            cnfs.append(variable_names)
            cnfs.extend([[-v1, -v2] for i, v1 in enumerate(variable_names[:-1], 0) for v2 in variable_names[i+1:]])

    def get_var_num(l):
        return l[0]*2*2*6+l[1]*2*6+l[2]*6+l[3]+1
    
    field_position_pairs = [
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(1, 1), (0, 1)],
        [(1, 1), (1, 0)],
        [(1, 1), (0, 0)],
        [(1, 0), (0, 1)],
    ]
    for field_num in range(0, 6):
        for c in colors:
            new_cnfs = [[-get_var_num([field_num]+list(v1)+[c]), -get_var_num([field_num]+list(v2)+[c])] for v1, v2 in field_position_pairs]
            cnfs.extend(new_cnfs)

    for lst_corner_pos in corner_positions:
        new_cnfs = []
        for c in colors:
            new_cnfs.extend([[-get_var_num(cp1+(c, )), -get_var_num(cp2+(c, ))] for cp1, cp2 in zip(lst_corner_pos, lst_corner_pos[1:]+lst_corner_pos[:1])])
        cnfs.extend(new_cnfs)

    amount_variables = len(set([abs(i) for l in cnfs for i in l]))
    amount_clauses = len(cnfs)

    with open("sat_problem.txt", "w") as f:
        f.write("p cnf {} {}\n".format(amount_variables, amount_clauses))

        for clause in cnfs:
            f.write("{} 0\n".format(" ".join(map(str, clause))))

    var_num = 145
    for tpl_cols in corner_colors:
        lst_variables = []
        for lst_corner_pos in corner_positions:
            for tpl_pos in zip(lst_corner_pos, lst_corner_pos[1:]+lst_corner_pos[:1], lst_corner_pos[2:]+lst_corner_pos[:2]):
                lst_variables.append(tuple(get_var_num(p+(c, )) for p, c in zip(tpl_pos, tpl_cols)))
        print("lst_variables:\n{}".format(lst_variables))
        new_variables = [var_num+i for i in range(0, len(lst_variables))]
        var_num += len(lst_variables)
        new_cnfs = []
        for tpl_v, v in zip(lst_variables, new_variables):
            new_cnfs.extend([[-v_ for v_ in tpl_v]+[v]]+[[v_, -v] for v_ in tpl_v])
        new_cnfs.append(new_variables)
        for i, v1 in enumerate(new_variables, 0):
            for v2 in new_variables[i+1:]:
                new_cnfs.append([-v1, -v2])
        # sys.exit()
        cnfs.extend(new_cnfs)

    s = Solver()
    random.shuffle(cnfs)
    for clause in cnfs:
        s.add_clause(clause)

    with Glucose3(bootstrap_with=cnfs) as m:
        print(m.solve())
        # print(m.solve(assumptions=[1, -3]))
        print("m.__dict__: {}".format(m.__dict__))
        print(list(m.get_model()))
        # print(list(m.enum_models()))
        # print(m.get_proof())
        models = [m for m, _ in zip(m.enum_models(), range(0, 1000))]
        idxs = [i-1 for i in models[np.random.randint(0, len(models))] if i > 0 and i < 145]
    idxs = np.array(idxs)
    # sys.exit()

    # sat, solution = s.solve()
    # print("sat: {}".format(sat))
    # print("solution: {}".format(solution))

    # arr = np.array(list(map(int, solution[1:])))
    # print("arr: {}".format(arr))

    # idxs = np.where(arr)[0]
    # idxs = idxs[idxs<144]

    color_map = np.array([(idxs//2//2//6)%6, (idxs//2//6)%2, (idxs//6)%2, (idxs)%6]).T
    print("color_map:\n{}".format(color_map))

    # get a simplified version of the current field!

    arr_field = np.zeros((8, 6), dtype=np.int)
    field_pos = {0: (2, 2), 1: (6, 2), 2: (4, 2), 3: (0, 2), 4: (2, 4), 5: (2, 0)}

    for f, y, x, c in color_map:
        y1, x1 = field_pos[f]
        arr_field[y1+y, x1+x] = c+1
    print("arr_field:\n{}".format(arr_field))

    colors_arr = np.array([
        [0x00, 0x00, 0x00],
        [0xFF, 0xFF, 0xFF],
        [0xFF, 0xFF, 0x00],
        [0xFF, 0xA5, 0x00],
        [0xFF, 0x00, 0x00],
        [0x00, 0x80, 0x00],
        [0x00, 0x00, 0xFF],
    ], dtype=np.uint8)

    pix = colors_arr[arr_field]
    img = Image.fromarray(pix)
    resize_factor = 20
    img = img.resize((img.width*resize_factor, img.height*resize_factor))
    img.show()

    # solver2 = Solver()
    # solver2.add_clause([1, 2])
    # solver2.add_clause([3, 4])
    # solver2.add_clause([-1, -5])
    # solver2.add_clause([1, 5])
    # solver2.add_clause([-2, -6])
    # solver2.add_clause([2, 6])
    # solver2.add_clause([-3, -7])
    # solver2.add_clause([3, 7])
    # solver2.add_clause([-4, -8])
    # solver2.add_clause([4, 8])

    # sat2, solution2 = solver2.solve()
    # print("sat2: {}, solution2: {}".format(sat2, solution2))
