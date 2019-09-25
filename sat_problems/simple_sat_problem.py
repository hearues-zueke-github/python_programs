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

# from pycryptosat import Solver


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

    # s = Solver()
    # random.shuffle(cnfs)
    # for clause in cnfs:
    #     s.add_clause(clause)

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


def get_rubiks_image(idxs, n):
    color_map = np.array([(idxs//n//n//6)%6, (idxs//n//6)%n, (idxs//6)%n, (idxs)%6]).T

    fields = np.zeros((6, n, n), dtype=np.int)
    for f, y, x, c in color_map:
        fields[f, y, x] = c+1

    colors_arr = np.array([
        [0x00, 0x00, 0x00],
        [0xFF, 0xFF, 0xFF],
        [0xFF, 0xFF, 0x00],
        [0xFF, 0xA5, 0x00],
        [0xFF, 0x00, 0x00],
        [0x00, 0x80, 0x00],
        [0x00, 0x00, 0xFF],
        [0x80, 0x80, 0x80],
        [0xC0, 0xC0, 0xC0],
    ], dtype=np.uint8)

    resize_factor = 20
    cell_space = 1
    field_space = 2
    arr_field_new = np.zeros((resize_factor*n*4+cell_space*(n-1)*4+field_space*2*4+cell_space*3, resize_factor*n*3+cell_space*(n-1)*3+field_space*2*3+cell_space*2), dtype=np.uint8)

    field_size_px = resize_factor*n+cell_space*(n-1)+field_space*2
    arr_field_new[field_size_px*1+cell_space*1:field_size_px*2+cell_space*1, field_size_px*0+cell_space*0:field_size_px*1+cell_space*0] = 7
    arr_field_new[field_size_px*1+cell_space*1:field_size_px*2+cell_space*1, field_size_px*1+cell_space*1:field_size_px*2+cell_space*1] = 7
    arr_field_new[field_size_px*1+cell_space*1:field_size_px*2+cell_space*1, field_size_px*2+cell_space*2:field_size_px*3+cell_space*2] = 7
    arr_field_new[field_size_px*0+cell_space*0:field_size_px*1+cell_space*0, field_size_px*1+cell_space*1:field_size_px*2+cell_space*1] = 7
    arr_field_new[field_size_px*2+cell_space*2:field_size_px*3+cell_space*2, field_size_px*1+cell_space*1:field_size_px*2+cell_space*1] = 7
    arr_field_new[field_size_px*3+cell_space*3:field_size_px*4+cell_space*3, field_size_px*1+cell_space*1:field_size_px*2+cell_space*1] = 7

    h = field_size_px-field_space*2
    w = field_size_px-field_space*2
    y = field_size_px*1+field_space+cell_space*1
    x = field_size_px*0+field_space+cell_space*0
    arr_field_new[y:y+h, x:x+w] = 8
    x += field_size_px+cell_space*1
    arr_field_new[y:y+h, x:x+w] = 8
    x += field_size_px+cell_space*1
    arr_field_new[y:y+h, x:x+w] = 8
    y -= field_size_px+cell_space*1
    x -= field_size_px+cell_space*1
    arr_field_new[y:y+h, x:x+w] = 8
    y += field_size_px*2+cell_space*2
    arr_field_new[y:y+h, x:x+w] = 8
    y += field_size_px*1+cell_space*1
    arr_field_new[y:y+h, x:x+w] = 8

    y = field_size_px*1+field_space+cell_space*1
    x = field_size_px*0+field_space+cell_space*0
    for j, row in enumerate(fields[5], 0):
        y2 = y+cell_space*j+resize_factor*j
        for i, v in enumerate(row, 0):
            x2 = x+cell_space*i+resize_factor*i
            arr_field_new[y2:y2+resize_factor, x2:x2+resize_factor] = v

    x += field_size_px*1+cell_space*1
    for j, row in enumerate(fields[0], 0):
        y2 = y+cell_space*j+resize_factor*j
        for i, v in enumerate(row, 0):
            x2 = x+cell_space*i+resize_factor*i
            arr_field_new[y2:y2+resize_factor, x2:x2+resize_factor] = v

    x += field_size_px*1+cell_space*1
    for j, row in enumerate(fields[4], 0):
        y2 = y+cell_space*j+resize_factor*j
        for i, v in enumerate(row, 0):
            x2 = x+cell_space*i+resize_factor*i
            arr_field_new[y2:y2+resize_factor, x2:x2+resize_factor] = v

    y -= field_size_px*1+cell_space*1
    x -= field_size_px*1+cell_space*1
    for j, row in enumerate(fields[3], 0):
        y2 = y+cell_space*j+resize_factor*j
        for i, v in enumerate(row, 0):
            x2 = x+cell_space*i+resize_factor*i
            arr_field_new[y2:y2+resize_factor, x2:x2+resize_factor] = v

    y += field_size_px*2+cell_space*2
    for j, row in enumerate(fields[2], 0):
        y2 = y+cell_space*j+resize_factor*j
        for i, v in enumerate(row, 0):
            x2 = x+cell_space*i+resize_factor*i
            arr_field_new[y2:y2+resize_factor, x2:x2+resize_factor] = v

    y += field_size_px*1+cell_space*1
    for j, row in enumerate(fields[1], 0):
        y2 = y+cell_space*j+resize_factor*j
        for i, v in enumerate(row, 0):
            x2 = x+cell_space*i+resize_factor*i
            arr_field_new[y2:y2+resize_factor, x2:x2+resize_factor] = v

    # TODO: add gray color for cell_space and other color for field_space!
    pix2 = colors_arr[arr_field_new]
    img2 = Image.fromarray(pix2)
    img = img2

    return img


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

    # add for each corner the same color as the field number
    cnfs.extend([[get_var_num([f, y, x, f])] for f in range(0, 6) for y in [0, 3] for x in [0, 3]])

    # cnfs.extend([
    #     [get_var_num([0, 3, 1, 2])],
    #     [get_var_num([2, 0, 1, 1])],
    #     [get_var_num([0, 3, 2, 3])],
    #     [get_var_num([2, 0, 2, 0])],

    #     # [get_var_num([0, 0, 1, 2])],
    #     # [get_var_num([3, 3, 1, 0])],
    #     # [get_var_num([0, 0, 2, 3])],
    #     # [get_var_num([3, 3, 2, 1])],
    # ])

    # write the sat problem as a dimacs format
    amount_variables = len(set([abs(i) for l in cnfs for i in l]))
    amount_clauses = len(cnfs)
    with open("sat_problem_rubiks_cube_4x4.txt", "w") as f:
        f.write("p cnf {} {}\n".format(amount_variables, amount_clauses))

        for clause in cnfs:
            f.write("{} 0\n".format(" ".join(map(str, clause))))

    globals()['cnfs'] = cnfs

    # s = Solver()
    # random.shuffle(cnfs)
    # for clause in cnfs:
    #     s.add_clause(clause)

    rubiks_cube_amount = 30
    with Glucose3(bootstrap_with=cnfs) as m:
        print(m.solve())
        print(list(m.get_model()))
        models = [m for m, _ in zip(m.enum_models(), range(0, rubiks_cube_amount))]
    
    # save the found models into a txt file!
    with open("found_valid_4x4_no_neightbor.txt", "w") as f:
        for model in models:
            idxs = [i-1 for i in model if i > 0 and i < 577]
            idxs = np.array(idxs)
            color_map = np.array([(idxs//4//4//6)%6, (idxs//4//6)%4, (idxs//6)%4, (idxs)%6]).T
            f.write("".join(map(str, color_map[:, -1].tolist()))+"\n")
    # sys.exit(-1)

    for i in np.random.permutation(np.arange(0, len(models)))[:1]:
        print("i: {}".format(i))
        model = models[i]
        idxs = [i-1 for i in model if i > 0 and i < 577]
        # idxs = [i-1 for i in models[np.random.randint(0, len(models))] if i > 0 and i < 577]
        idxs = np.array(idxs)

        img = get_rubiks_image(idxs, 4)
        img.show()


def get_solution_plain_latin_square(n, k):
    variables = [[[i1*n**2+i2*n+i3+1 for i3 in range(0, n)] for i2 in range(0, n)] for i1 in range(0, n)]
    vars_dict = {variables[i1][i2][i3]: (i1, i2, i3) for i1 in range(0, n) for i2 in range(0, n) for i3 in range(0, n)}
    globals()["variables"] = variables
    globals()["vars_dict"] = vars_dict
    last_new_var_num = variables[-1][-1][-1]+1

    cnfs = []
    
    # add cell restriction, only one number per cell!
    new_cnfs = []
    for i1 in range(0, n):
        for i2 in range(0, n):
            vars_part = [variables[i1][i2][i3] for i3 in range(0, n)]
            new_cnfs.append(vars_part)
            new_cnfs.extend([[-j1, -j2] for k, j1 in enumerate(vars_part[:-1], 0) for j2 in vars_part[k+1:]])
    cnfs.extend(new_cnfs)

    # add row restriction, only one number per row!
    new_cnfs = []
    for i1 in range(0, n):
        for i3 in range(0, n):
            vars_part = [variables[i1][i2][i3] for i2 in range(0, n)]
            new_cnfs.append(vars_part)
            new_cnfs.extend([[-j1, -j2] for k, j1 in enumerate(vars_part[:-1], 0) for j2 in vars_part[k+1:]])
    cnfs.extend(new_cnfs)

    # add column restriction, only one number per column!
    new_cnfs = []
    for i2 in range(0, n):
        for i3 in range(0, n):
            vars_part = [variables[i1][i2][i3] for i1 in range(0, n)]
            new_cnfs.append(vars_part)
            new_cnfs.extend([[-j1, -j2] for k, j1 in enumerate(vars_part[:-1], 0) for j2 in vars_part[k+1:]])
    cnfs.extend(new_cnfs)

    # add exact amount of a certain number!
    amount_bits = len(bin(n**2)[2:])
    # i3 = 0 # the 0 can only appear n times! if n=4 -> it would be 4 times!
    for i3 in range(0, n):
        vars_s = [[last_new_var_num+i1*amount_bits+i2 for i2 in range(0, amount_bits)] for i1 in range(0, n**2+1)]
        last_new_var_num = vars_s[-1][-1]+1
        vars_r = [[last_new_var_num+i1*(amount_bits-1)+i2 for i2 in range(0, amount_bits-1)] for i1 in range(0, n**2)]
        last_new_var_num = vars_r[-1][-1]+1

        used_variables = [variables[i1][i2][i3] for i1 in range(0, n) for i2 in range(0, n)]
        globals()['used_variables'] = used_variables
        new_cnfs = []
        for v, row_s1, row_s2, row_r in zip(used_variables, vars_s[:-1], vars_s[1:], vars_r):
            s1 = row_s1[0]
            s2 = row_s2[0]
            r = row_r[0]
            new_cnfs.extend([[-s1, -v, -s2], [s1, v, -s2], [-s1, v, s2], [s1, -v, s2]]) # xor
            new_cnfs.extend([[-s1, -v, r], [s1, -r], [v, -r]]) # and
            for s1, s2, r in zip(row_s1[1:], row_s2[1:], row_r):
                new_cnfs.extend([[-s1, -r, -s2], [s1, r, -s2], [-s1, r, s2], [s1, -r, s2]]) # xor
            for s1, r1, r2 in zip(row_s1[1:-1], row_r[:-1], row_r[1:]):
                new_cnfs.extend([[-s1, -r1, r2], [s1, -r2], [r1, -r2]]) # and
        cnfs.extend(new_cnfs)
        
        cnfs.extend([[-s] for s in vars_s[0]])

        needed_bin_num = list(map(int, bin(n)[2:].zfill(amount_bits)))[::-1]
        # needed_bin_num = list(map(int, bin(1)[2:].zfill(amount_bits)))[::-1]
        # print("needed_bin_num: {}".format(needed_bin_num))
        clause_end = [[s if i==1 else -s] for s, i in zip(vars_s[-1], needed_bin_num)]
        # globals()['clause_end'] = clause_end
        cnfs.extend(clause_end)

    # add neighbor restrions
    # 3x3 field need to add diagonals restrictions too! k=1 is 3x3, k=2 is 5x5 etc.
    # k = 3
    new_cnfs = []
    for i3 in range(0, n):
        for k1 in range(1, k+1):
            for i1 in range(0, n-k1):
                for i2 in range(0, n):
                    new_cnfs.append([-variables[i1][i2][i3], -variables[i1+k1][i2][i3]])
            for i1 in range(0, n):
                for i2 in range(0, n-k1):
                    new_cnfs.append([-variables[i1][i2][i3], -variables[i1][i2+k1][i3]])

            for k2 in range(1, k+1):
                for i1 in range(0, n-k1):
                    for i2 in range(0, n-k2):
                        new_cnfs.append([-variables[i1][i2][i3], -variables[i1+k1][i2+k2][i3]])
                for i1 in range(k1, n):
                    for i2 in range(0, n-k2):
                        new_cnfs.append([-variables[i1][i2][i3], -variables[i1-k1][i2+k2][i3]])
    cnfs.extend(new_cnfs)

    new_cnfs = []
    n_sqrt = int(np.sqrt(n))
    if n_sqrt**2==n:
        for i3 in range(0, n):
            for j1 in range(0, n_sqrt):
                for j2 in range(0, n_sqrt):
                    variables_part = [variables[j1*n_sqrt+i1][j2*n_sqrt+i2][i3] for i1 in range(0, n_sqrt) for i2 in range(0, n_sqrt)]
                    for i, v1 in enumerate(variables_part[:-1], 0):
                        for v2 in variables_part[i+1:]:
                            new_cnfs.append([-v1, -v2])

            # only needed for a special sudoku type!
            for j1 in range(0, n_sqrt):
                for j2 in range(0, n_sqrt):
                    variables_part = [variables[j1+i1*n_sqrt][j2+i2*n_sqrt][i3] for i1 in range(0, n_sqrt) for i2 in range(0, n_sqrt)]
                    for i, v1 in enumerate(variables_part[:-1], 0):
                        for v2 in variables_part[i+1:]:
                            new_cnfs.append([-v1, -v2])
    cnfs.extend(new_cnfs)

    print("len(cnfs): {}".format(len(cnfs)))

    with Glucose3(bootstrap_with=cnfs) as m:
        print(m.solve())
        print("len(list(m.get_model())): {}".format(len(list(m.get_model()))))
        models = []
        for i, m in zip(range(1, 1+1), m.enum_models()):
            print("found model nr. i: {}".format(i))
            models.append(m)
        # models = [m for m, _ in zip(m.enum_models(), range(0, 1))]

    def convert_model_to_field(m):
        # arr = np.array(models[0])
        arr = np.array(m)
        arr = arr[(arr>0)&(arr<=n**3)]
        field = np.zeros((n, n), dtype=np.int)
        for v in arr:
            tpl = vars_dict[v]
            field[tpl[:2]] = tpl[2]
        return field

    for i, m in enumerate(models, 1):
        field = convert_model_to_field(m)
        # field = convert_model_to_field(models[0])
        print("i: {}, field:\n{}".format(i, field))

    return field


if __name__ == "__main__":
    # test_a_simple_expression_comparison()
    # find_no_8_neighbor_same_color_for_rubiks_cube_2x2()

    # get random 6xnxn cube!
    # n = 10

    # idxs = [6*n*n*f+6*n*y+6*x+int(np.random.randint(0, 6)) for f in range(0, 6) for y in range(0, n) for x in range(0, n)]
    # idxs = np.array(idxs)

    # img = get_rubiks_image(idxs, n)
    # img.show()
    # sys.exit(0)

    find_no_8_neighbor_same_color_for_rubiks_cube_4x4()
    sys.exit(0)
    # k=0, n=2
    # k=1, n=5
    # k=2, n=10
    # k=3, n=17
    # k=4, n=26

    n_sqrt = 3
    n = 9
    k = 1

    field = get_solution_plain_latin_square(n=n, k=k)

    # check, if field is really a OK sudoku field?!
    print("check rows: ")
    set_lens_row = []
    for j in range(0, n):
        values = []
        for i in range(0, n):
            values.append(field[j, i])
        vals_set = set(values)
        set_lens_row.append((j, len(vals_set)))

    print("check cols: ")
    set_lens_col = []
    for i in range(0, n):
        values = []
        for j in range(0, n):
            values.append(field[j, i])
        vals_set = set(values)
        set_lens_col.append((i, len(vals_set)))

    print("check sqrt block: ")
    set_lens_sqrt_block = []
    for j1 in range(0, n_sqrt):
        for i1 in range(0, n_sqrt):
            values = []
            for j2 in range(0, n_sqrt):
                for i2 in range(0, n_sqrt):
                    values.append(field[j1*n_sqrt+j2, i1*n_sqrt+i2])
            vals_set = set(values)
            set_lens_sqrt_block.append((j1*n_sqrt+i1, len(vals_set)))

    print("check sqrt 1 cell: ")
    set_lens_sqrt_1_cell = []
    for j1 in range(0, n_sqrt):
        for i1 in range(0, n_sqrt):
            values = []
            for j2 in range(0, n_sqrt):
                for i2 in range(0, n_sqrt):
                    values.append(field[j1+j2*n_sqrt, i1+i2*n_sqrt])
            vals_set = set(values)
            set_lens_sqrt_1_cell.append((j1*n_sqrt+i1, len(vals_set)))

    print("check neighbor (k*2+1)x(k*2+1): ")
    field_neighbor_check = np.zeros((n, n), dtype=np.int)
    for j in range(k, n-k):
        for i in range(k, n-k):
            f_part = field[j-k:j+k+1, i-k:i+k+1]
            field_neighbor_check[j, i] = np.sum(f_part==field[j, i])

    print("set_lens_row: {}".format(set_lens_row))
    print("set_lens_col: {}".format(set_lens_col))
    print("set_lens_sqrt_block: {}".format(set_lens_sqrt_block))
    print("set_lens_sqrt_1_cell: {}".format(set_lens_sqrt_1_cell))

    idxs_x = np.zeros((n, n), dtype=np.int)+np.arange(0, n)
    idxs_y = idxs_x.T
    idxs = np.dstack((idxs_y, idxs_x))

    idxs1 = idxs-k
    idxs2 = idxs+k+1

    idxs1[idxs1<0] = 0
    idxs2[idxs2>=n] = n

    for yxs, yxs1, yxs2 in zip(idxs, idxs1, idxs2):
        for (y, x), (y1, x1), (y2, x2) in zip(yxs, yxs1, yxs2):
            # print("y1: {}, x1: {}, y2: {}, x2: {}".format(y1, x1, y2, x2))
            field_part = field[y1:y2, x1:x2]
            field_neighbor_check[y, x] = np.sum(field_part==field[y, x])
            # print(" - field_part:\n{}".format(field_part))
    print("field_neighbor_check:\n{}".format(field_neighbor_check))
    