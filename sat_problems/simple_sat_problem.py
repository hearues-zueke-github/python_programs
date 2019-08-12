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


def test_a_simple_expression_comparison():
    def expr_1(a, b, c):
        return (a and not b and not c) or (not a and b and not c) or (not a and not b and c)
    def expr_2(a, b, c):
        return (a or b or c) and (not a or not b) and (not a or not c) and (not b or not c)

    params = [(a, b, c) for a in (True, False) for b in (True, False) for c in (True, False)]
    lst1 = [expr_1(*p) for p in params]
    lst2 = [expr_2(*p) for p in params]

    print("lst1: {}".format(lst1))
    print("lst2: {}".format(lst2))


def find_no_8_neighbor_same_color_for_rubiks_cube_2x2():
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

    cnfs = []

    # add the restriction for each cell that only one color is possible!
    for f in range(0, 6):
        for y in range(0, 2):
            for x in range(0, 2):
                prefix_num = f*2*2*6+y*2*6+x*6+1
                variable_names = [prefix_num+c for c in colors]
                cnfs.append(variable_names)
                cnfs.extend([[-v1, -v2] for i, v1 in enumerate(variable_names[:-1], 0) for v2 in variable_names[i+1:]])


    def get_var_num(l):
        return l[0]*2*2*6+l[1]*2*6+l[2]*6+l[3]+1
    
    # add the restriction for each field (2x2) to not to have the same color!
    field_position_pairs = (
        [[(j, i), (j, i+1)] for j in range(0, 2) for i in range(0, 1)]+
        [[(j, i), (j+1, i)] for j in range(0, 1) for i in range(0, 2)]+
        [[(j, i), (j+1, i+1)] for j in range(0, 1) for i in range(0, 1)]+
        [[(j, i+1), (j+1, i)] for j in range(0, 1) for i in range(0, 1)]
    )
    for field_num in range(0, 6):
        for c in colors:
            new_cnfs = [[-get_var_num([field_num]+list(v1)+[c]), -get_var_num([field_num]+list(v2)+[c])] for v1, v2 in field_position_pairs]
            cnfs.extend(new_cnfs)

    # maybe this restriction is too much for now!
    # # add for each color for each corner position the restriction for not to have the same color!
    # for lst_corner_pos in corner_positions:
    #     new_cnfs = []
    #     for c in colors:
    #         new_cnfs.extend([[-get_var_num(cp1+(c, )), -get_var_num(cp2+(c, ))] for cp1, cp2 in zip(lst_corner_pos, lst_corner_pos[1:]+lst_corner_pos[:1])])
    #     cnfs.extend(new_cnfs)

    # add each corner with the colors at least and most once!
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
        cnfs.extend(new_cnfs)

    # write the sat problem as a dimacs format
    amount_variables = len(set([abs(i) for l in cnfs for i in l]))
    amount_clauses = len(cnfs)
    with open("sat_problem.txt", "w") as f:
        f.write("p cnf {} {}\n".format(amount_variables, amount_clauses))

        for clause in cnfs:
            f.write("{} 0\n".format(" ".join(map(str, clause))))

    s = Solver()
    random.shuffle(cnfs)
    for clause in cnfs:
        s.add_clause(clause)

    with Glucose3(bootstrap_with=cnfs) as m:
        print(m.solve())
        print(list(m.get_model()))
        models = [m for m, _ in zip(m.enum_models(), range(0, 1000))]
        idxs = [i-1 for i in models[np.random.randint(0, len(models))] if i > 0 and i < 145]
    idxs = np.array(idxs)


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


def find_no_8_neighbor_same_color_for_rubiks_cube_4x4():
    corner_positions = [
        [(1, 1, 1), (4, 4, 1), (6, 1, 4)],
        [(1, 1, 4), (5, 1, 1), (4, 4, 4)],
        [(1, 4, 4), (3, 1, 4), (5, 4, 1)],
        [(1, 4, 1), (6, 4, 4), (3, 1, 1)],
        [(2, 1, 1), (3, 4, 1), (6, 4, 1)],
        [(2, 1, 4), (5, 4, 4), (3, 4, 4)],
        [(2, 4, 4), (4, 1, 4), (5, 1, 4)],
        [(2, 4, 1), (6, 1, 1), (4, 1, 1)],
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

    cnfs = []

    # add the restriction for each cell that only one color is possible!
    for f in range(0, 6):
        # break
        for y in range(0, 4):
            for x in range(0, 4):
                prefix_num = f*4*4*6+y*4*6+x*6+1
                variable_names = [prefix_num+c for c in colors]
                cnfs.append(variable_names)
                cnfs.extend([[-v1, -v2] for i, v1 in enumerate(variable_names[:-1], 0) for v2 in variable_names[i+1:]])

    def get_var_num(l):
        return l[0]*4*4*6+l[1]*4*6+l[2]*6+l[3]+1
    
    # add the restriction for each field (2x2) to not to have the same color!
    field_position_pairs = (
        [[(j, i), (j, i+1)] for j in range(0, 4) for i in range(0, 3)]+
        [[(j, i), (j+1, i)] for j in range(0, 3) for i in range(0, 4)]+
        [[(j, i), (j+1, i+1)] for j in range(0, 3) for i in range(0, 3)]+
        [[(j, i+1), (j+1, i)] for j in range(0, 3) for i in range(0, 3)]
    )
    for field_num in range(0, 6):
        # break
        for c in colors:
            new_cnfs = [[-get_var_num([field_num]+list(v1)+[c]), -get_var_num([field_num]+list(v2)+[c])] for v1, v2 in field_position_pairs]
            cnfs.extend(new_cnfs)

    new_var_num = 4*4*6*6+1
    
    # add each corner with the colors at least once and most once!
    for tpl_cols in corner_colors:
        # break
        lst_variables = []
        for lst_corner_pos in corner_positions:
            for tpl_pos in zip(lst_corner_pos, lst_corner_pos[1:]+lst_corner_pos[:1], lst_corner_pos[2:]+lst_corner_pos[:2]):
                lst_variables.append(tuple(get_var_num(p+(c, )) for p, c in zip(tpl_pos, tpl_cols)))
        new_variables = [new_var_num+i for i in range(0, len(lst_variables))]
        new_var_num += len(lst_variables)
        new_cnfs = []
        for tpl_v, v in zip(lst_variables, new_variables):
            new_cnfs.extend([[-v_ for v_ in tpl_v]+[v]]+[[v_, -v] for v_ in tpl_v])
        new_cnfs.append(new_variables)
        for i, v1 in enumerate(new_variables, 0):
            for v2 in new_variables[i+1:]:
                new_cnfs.append([-v1, -v2])
        cnfs.extend(new_cnfs)

    # add restrictions for the middle centers, where each center color is max 4 times appearing!
    field_centers = [[f, j, i] for f in range(0, 6) for j in range(1, 3) for i in range(1, 3)]
    for c in colors:
        # break
        lst_variables = []
        for pos in field_centers:
            lst_variables.append(get_var_num(pos+[c]))
        
        lst_s_rows = []
        lst_r_rows = []
        lst_s_row = [new_var_num+i for i in range(0, 5)]
        new_var_num += len(lst_s_row)
        lst_s_rows.append(lst_s_row)    
        for i in range(0, 23):
            lst_s_row = [new_var_num+i for i in range(0, 5)]
            new_var_num += len(lst_s_row)
            lst_r_row = [new_var_num+i for i in range(0, 4)]
            new_var_num += len(lst_r_row)
            lst_s_rows.append(lst_s_row)
            lst_r_rows.append(lst_r_row)

        # add the known values for first and last s and r values
        new_cnfs = []
        v_0 = lst_variables[0]
        s_0_0 = lst_s_rows[0][0]
        new_cnfs.extend([[s_0_0, -v_0], [-s_0_0, v_0]])
        new_cnfs.extend([[-v] for v in lst_s_rows[0][1:]])
        new_cnfs.extend([[i*v] for i, v in zip([-1, -1, 1, -1, -1], lst_s_rows[-1])])

        # add the tseyten transformation for the addition/arithmetic stuff!
        for v, s_row_prev, s_row_next, r_row in zip(
                lst_variables[1:],
                lst_s_rows[:-1],
                lst_s_rows[1:], lst_r_rows):
            s_p = s_row_prev[0]
            s_n = s_row_next[0]
            r = r_row[0]
            new_cnfs.extend([[-s_p, -v, -s_n], [s_p, v, -s_n], [-s_p, v, s_n], [s_p, -v, s_n]])
            new_cnfs.extend([[-s_p, -v, r], [s_p, -r], [v, -r]])
            for s_p, r_p, s_n, r_n in zip(s_row_prev[1:-1], r_row[:-1], s_row_next[1:-1], r_row[1:]):
                new_cnfs.extend([[-s_p, -r_p, -s_n], [s_p, r_p, -s_n], [-s_p, r_p, s_n], [s_p, -r_p, s_n]])
                new_cnfs.extend([[-s_p, -r_p, r_n], [s_p, -r_n], [r_p, -r_n]])
            s_p = s_row_prev[-1]
            s_n = s_row_next[-1]
            r = r_row[-1]
            new_cnfs.extend([[-s_p, -r, -s_n], [s_p, r, -s_n], [-s_p, r, s_n], [s_p, -r, s_n]])
        cnfs.extend(new_cnfs)

    # add restrictions for edges!
    edges_positions = [
        [(3, 3, 1), (0, 0, 1)],
        [(0, 0, 2), (3, 3, 2)],
        [(5, 2, 3), (0, 2, 0)],
        [(0, 1, 0), (5, 1, 3)],
        [(0, 3, 1), (2, 0, 1)],
        [(2, 0, 2), (0, 3, 2)],
        [(4, 1, 0), (0, 1, 3)],
        [(0, 2, 3), (4, 2, 0)],

        [(4, 3, 1), (2, 1, 3)],
        [(2, 2, 3), (4, 3, 2)],
        [(4, 0, 2), (3, 1, 3)],
        [(3, 2, 3), (4, 0, 1)],

        [(3, 1, 0), (5, 0, 1)],
        [(5, 0, 2), (3, 2, 0)],
        [(2, 1, 0), (5, 3, 2)],
        [(5, 3, 1), (2, 2, 0)],

        [(1, 0, 2), (2, 3, 2)],
        [(2, 3, 1), (1, 0, 1)],
        [(1, 3, 1), (3, 0, 1)],
        [(3, 0, 2), (1, 3, 2)],

        [(4, 2, 3), (1, 1, 3)],
        [(1, 2, 3), (4, 1, 3)],
        [(1, 1, 0), (5, 2, 0)],
        [(5, 1, 0), (1, 2, 0)],
    ]
    edges_colors = [
        (0, 2), (2, 0),
        (0, 4), (4, 0),
        (0, 3), (3, 0),
        (0, 5), (5, 0),
        
        (2, 4), (4, 2),
        (3, 4), (4, 3),
        (3, 5), (5, 3),
        (2, 5), (5, 2),
        
        (1, 2), (2, 1),
        (1, 4), (4, 1),
        (1, 3), (3, 1),
        (1, 5), (5, 1),
    ]

    for c1, c2 in edges_colors:
        # break
        lst_variables = [[get_var_num(t1+(c1, )), get_var_num(t2+(c2, ))] for t1, t2 in edges_positions]
        new_variables = [new_var_num+i for i in range(0, len(lst_variables))]
        new_var_num += len(lst_variables)
        new_cnfs = []
        for (v1, v2), v in zip(lst_variables, new_variables):
            new_cnfs.extend([[-v1, -v2, v], [v1, -v], [v2, -v]])
        new_cnfs.extend([[-v1, -v2] for i, v1 in enumerate(new_variables, 0) for v2 in new_variables[i+1:]])
        cnfs.append(new_variables)
        cnfs.extend(new_cnfs)

    # write the sat problem as a dimacs format
    amount_variables = len(set([abs(i) for l in cnfs for i in l]))
    amount_clauses = len(cnfs)
    with open("sat_problem_rubiks_cube_4x4.txt", "w") as f:
        f.write("p cnf {} {}\n".format(amount_variables, amount_clauses))

        for clause in cnfs:
            f.write("{} 0\n".format(" ".join(map(str, clause))))

    globals()['cnfs'] = cnfs

    s = Solver()
    random.shuffle(cnfs)
    for clause in cnfs:
        s.add_clause(clause)

    with Glucose3(bootstrap_with=cnfs) as m:
        print(m.solve())
        print(list(m.get_model()))
        models = [m for m, _ in zip(m.enum_models(), range(0, 10000))]
    
    for i in np.random.permutation(np.arange(0, len(models)))[:5]:
        print("i: {}".format(i))
        model = models[i]
        idxs = [i-1 for i in model if i > 0 and i < 577]
        # idxs = [i-1 for i in models[np.random.randint(0, len(models))] if i > 0 and i < 577]
        idxs = np.array(idxs)

        color_map = np.array([(idxs//4//4//6)%6, (idxs//4//6)%4, (idxs//6)%4, (idxs)%6]).T
        # print("color_map:\n{}".format(color_map))

        arr_field = np.zeros((16, 12), dtype=np.int)
        field_pos = {0: (4, 4), 1: (12, 4), 2: (8, 4), 3: (0, 4), 4: (4, 8), 5: (4, 0)}

        for f, y, x, c in color_map:
            y1, x1 = field_pos[f]
            arr_field[y1+y, x1+x] = c+1
        # print("arr_field:\n{}".format(arr_field))

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


if __name__ == "__main__":
    # test_a_simple_expression_comparison()
    # find_no_8_neighbor_same_color_for_rubiks_cube_2x2()
    find_no_8_neighbor_same_color_for_rubiks_cube_4x4()
