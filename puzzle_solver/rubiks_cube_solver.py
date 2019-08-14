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


# TODO create a similar function for the rotation of the other fields too! (ring rotation vs field rotation!)


class RubiksCube(Exception):
    coordinates = np.dstack((np.repeat(np.arange(0, 4), 4).reshape((4, 4)), np.tile(np.arange(0, 4), 4).reshape((4, 4))))
    coordinates_new_clockwise = np.flip(coordinates, axis=1).transpose(1, 0, 2)
    coordinates_new_counter_clockwise = np.flip(coordinates, axis=0).transpose(1, 0, 2)
    get_tpl = lambda x: (lambda y: (y[0], y[1]))(x.reshape((-1, 2)).T)
    tpl = get_tpl(coordinates)
    tpl_new_clockwise = get_tpl(coordinates_new_clockwise)
    tpl_new_counter_clockwise = get_tpl(coordinates_new_counter_clockwise)
    
    def __init__(self, n):
        self.n = n # will be defined for 4x4 now!
        self._reset_fields()
        # self.fields[0, 0, 0] = 1
        # self.fields[0, 0, 1] = 2
        # self.fields[0, 1, 1] = 3
        # self.fields[1, 0, 0] = 2
        # self.fields[1, 0, 1] = 3
        # self.fields[1, 1, 1] = 4

        # Colors: 0123456 -> WYORGB
        self.dict_col_int_col_str = {0: 'W', 1: 'Y', 2: 'O', 3: 'R', 4: 'G', 5: 'B'}

        moving_dicts_basis = {
            "U":  {"U": "U", "D": "D", "F": "L", "R": "F", "B": "R", "L": "B"},
            "U'": {"U": "U", "D": "D", "F": "R", "R": "B", "B": "L", "L": "F"},
            "F":  {"U": "R", "D": "L", "F": "F", "R": "D", "B": "B", "L": "U"},
            "F'": {"U": "L", "D": "R", "F": "F", "R": "U", "B": "B", "L": "D"},
            "R":  {"U": "B", "D": "F", "F": "U", "R": "R", "B": "D", "L": "L"},
            "R'": {"U": "F", "D": "B", "F": "D", "R": "R", "B": "U", "L": "L"},
        }
        
        moving_dicts_part = {
            k: {
                **v,
                **{k1.lower(): v1.lower() for k1, v1 in v.items()},
                **{k1+"w": v1+"w" for k1, v1 in v.items()}
            } for k, v in moving_dicts_basis.items()
        }
        self.moving_dicts = {
            k: {
                **v,
                **{k1+"'": v1+"'" for k1, v1 in v.items()},
                **{k1+"2": v1+"2" for k1, v1 in v.items()},
            } for k, v in moving_dicts_part.items()
        }

        self.moving_table = {
            "U": lambda: (self.rotate_u(0), ),
            "D": lambda: (self.rotate_ui(3), ),
            "F": lambda: (self.rotate_f(0), ),
            "B": lambda: (self.rotate_fi(3), ),
            "R": lambda: (self.rotate_r(0), ),
            "L": lambda: (self.rotate_ri(3), ),
            "U'": lambda: (self.rotate_ui(0), ),
            "D'": lambda: (self.rotate_u(3), ),
            "F'": lambda: (self.rotate_fi(0), ),
            "B'": lambda: (self.rotate_f(3), ),
            "R'": lambda: (self.rotate_ri(0), ),
            "L'": lambda: (self.rotate_r(3), ),
            "U2": lambda: (self.rotate_u(0), self.rotate_u(0)),
            "D2": lambda: (self.rotate_ui(3), self.rotate_ui(3)),
            "F2": lambda: (self.rotate_f(0), self.rotate_f(0)),
            "B2": lambda: (self.rotate_fi(3), self.rotate_fi(3)),
            "R2": lambda: (self.rotate_r(0), self.rotate_r(0)),
            "L2": lambda: (self.rotate_ri(3), self.rotate_ri(3)),
            "u": lambda: (self.rotate_u(1), ),
            "d": lambda: (self.rotate_ui(2), ),
            "f": lambda: (self.rotate_f(1), ),
            "b": lambda: (self.rotate_fi(2), ),
            "r": lambda: (self.rotate_r(1), ),
            "l": lambda: (self.rotate_ri(2), ),
            "u'": lambda: (self.rotate_ui(1), ),
            "d'": lambda: (self.rotate_u(2), ),
            "f'": lambda: (self.rotate_fi(1), ),
            "b'": lambda: (self.rotate_f(2), ),
            "r'": lambda: (self.rotate_ri(1), ),
            "l'": lambda: (self.rotate_r(2), ),
            "u2": lambda: (self.rotate_u(1), self.rotate_u(1)),
            "d2": lambda: (self.rotate_ui(2), self.rotate_ui(2)),
            "f2": lambda: (self.rotate_f(1), self.rotate_f(1)),
            "b2": lambda: (self.rotate_fi(2), self.rotate_fi(2)),
            "r2": lambda: (self.rotate_r(1), self.rotate_r(1)),
            "l2": lambda: (self.rotate_ri(2), self.rotate_ri(2)),
            "Uw": lambda: (self.rotate_u(0), self.rotate_u(1)),
            "Dw": lambda: (self.rotate_ui(3), self.rotate_ui(2)),
            "Fw": lambda: (self.rotate_f(0), self.rotate_f(1)),
            "Bw": lambda: (self.rotate_fi(3), self.rotate_fi(2)),
            "Rw": lambda: (self.rotate_r(0), self.rotate_r(1)),
            "Lw": lambda: (self.rotate_ri(3), self.rotate_ri(2)),
            "Uw'": lambda: (self.rotate_ui(0), self.rotate_ui(1)),
            "Dw'": lambda: (self.rotate_u(3), self.rotate_u(2)),
            "Fw'": lambda: (self.rotate_fi(0), self.rotate_fi(1)),
            "Bw'": lambda: (self.rotate_f(3), self.rotate_f(2)),
            "Rw'": lambda: (self.rotate_ri(0), self.rotate_ri(1)),
            "Lw'": lambda: (self.rotate_r(3), self.rotate_r(2)),
            "Uw2": lambda: (self.rotate_u(0), self.rotate_u(0), self.rotate_u(1), self.rotate_u(1)),
            "Dw2": lambda: (self.rotate_ui(3), self.rotate_ui(3), self.rotate_ui(2), self.rotate_ui(2)),
            "Fw2": lambda: (self.rotate_f(0), self.rotate_f(0), self.rotate_f(1), self.rotate_f(1)),
            "Bw2": lambda: (self.rotate_fi(3), self.rotate_fi(3), self.rotate_fi(2), self.rotate_fi(2)),
            "Rw2": lambda: (self.rotate_r(0), self.rotate_r(0), self.rotate_r(1), self.rotate_r(1)),
            "Lw2": lambda: (self.rotate_ri(3), self.rotate_ri(3), self.rotate_ri(2), self.rotate_ri(2)),
        }

        self.possible_moves = np.array([k for k in self.moving_table])


    def _reset_fields(self):
        self.fields = np.array([[[i]*self.n]*self.n for i in range(0, 6)])


    def __rotate_field_counter_clockwise_4x4(self, field):
        field[RubiksCube.tpl] = field[RubiksCube.tpl_new_clockwise]


    def __rotate_field_clockwise_4x4(self, field):
        field[RubiksCube.tpl] = field[RubiksCube.tpl_new_counter_clockwise]


    def apply_move_lst(self, move_lst):
        for m in move_lst:
            self.moving_table[m]()


    def print_field(self):
        s = ""
        for row in self.fields[3]:
            s += " #"*self.n+" | "+" ".join(map(lambda x: self.dict_col_int_col_str[x], row))+" |"+" #"*self.n+"\n"
        s += " -"*self.n+" +"+" -"*self.n+" +"+" -"*self.n+"\n"
        for row1, row2, row3 in zip(self.fields[5], self.fields[0], self.fields[4]):
            s += " "+" ".join(map(lambda x: self.dict_col_int_col_str[x], row1))
            s += " |"
            s += " "+" ".join(map(lambda x: self.dict_col_int_col_str[x], row2))
            s += " |"
            s += " "+" ".join(map(lambda x: self.dict_col_int_col_str[x], row3))
            s += "\n"
        s += " -"*self.n+" +"+" -"*self.n+" +"+" -"*self.n+"\n"
        for row in self.fields[2]:
            s += " #"*self.n+" | "+" ".join(map(lambda x: self.dict_col_int_col_str[x], row))+" |"+" #"*self.n+"\n"
        s += " #"*self.n+" +"+" -"*self.n+" +"+" #"*self.n+"\n"
        for row in self.fields[1]:
            s += " #"*self.n+" | "+" ".join(map(lambda x: self.dict_col_int_col_str[x], row))+" |"+" #"*self.n+"\n"
        print("s:\n{}".format(s))


    # UFR -> in clockwise!
    def rotate_u(self, layer):
        # layer = 0..n-1
        # if layer==0: rotate U layer (W)
        # if layer==n-1: rotate D layer (Y')
        n = self.n
        if layer==0:
            self.__rotate_field_clockwise_4x4(self.fields[0])
        elif layer==n-1:
            self.__rotate_field_counter_clockwise_4x4(self.fields[1])


        row_b = self.fields[5, :, n-1-layer]
        row_r = self.fields[3, n-1-layer, :]
        row_g = self.fields[4, :, layer]
        row_o = self.fields[2, layer, :]

        temp = row_r.copy()
        row_r[:] = np.flip(row_b)
        row_b[:] = row_o
        row_o[:] = np.flip(row_g)
        row_g[:] = temp

    def rotate_f(self, layer):
        # layer = 0..n-1
        # if layer==0: rotate U layer (W)
        # if layer==n-1: rotate D layer (Y')
        n = self.n
        if layer==0:
            self.__rotate_field_clockwise_4x4(self.fields[2])
        elif layer==n-1:
            self.__rotate_field_counter_clockwise_4x4(self.fields[3])


        row_b = self.fields[5, n-1-layer, ::]
        row_w = self.fields[0, n-1-layer, :]
        row_g = self.fields[4, n-1-layer, :]
        row_y = self.fields[1, layer, :]

        temp = row_b.copy()
        row_b[:] = np.flip(row_y)
        row_y[:] = np.flip(row_g)
        row_g[:] = row_w
        row_w[:] = temp


    def rotate_r(self, layer):
        n = self.n
        fields = self.fields

        if layer==0:
            self.__rotate_field_clockwise_4x4(fields[4])
        elif layer==n-1:
            self.__rotate_field_counter_clockwise_4x4(fields[5])


        row_r = fields[3, :, n-1-layer]
        row_w = fields[0, :, n-1-layer]
        row_o = fields[2, :, n-1-layer]
        row_y = fields[1, :, n-1-layer]

        temp = row_r.copy()
        row_r[:] = row_w
        row_w[:] = row_o
        row_o[:] = row_y
        row_y[:] = temp


    def rotate_ui(self, layer):
        # layer = 0..n-1
        # if layer==0: rotate U layer (W)
        # if layer==n-1: rotate D layer (Y')
        n = self.n
        if layer==0:
            self.__rotate_field_counter_clockwise_4x4(self.fields[0])
        elif layer==n-1:
            self.__rotate_field_clockwise_4x4(self.fields[1])


        row_b = self.fields[5, :, n-1-layer]
        row_o = self.fields[2, layer, :]
        row_g = self.fields[4, :, layer]
        row_r = self.fields[3, n-1-layer, :]

        temp = row_r.copy()
        row_r[:] = row_g
        row_g[:] = np.flip(row_o)
        row_o[:] = row_b
        row_b[:] = np.flip(temp)


    def rotate_fi(self, layer):
        # layer = 0..n-1
        # if layer==0: rotate U layer (W)
        # if layer==n-1: rotate D layer (Y')
        n = self.n
        if layer==0:
            self.__rotate_field_counter_clockwise_4x4(self.fields[2])
        elif layer==n-1:
            self.__rotate_field_clockwise_4x4(self.fields[3])


        row_b = self.fields[5, n-1-layer, ::]
        row_w = self.fields[0, n-1-layer, :]
        row_g = self.fields[4, n-1-layer, :]
        row_y = self.fields[1, layer, :]

        temp = row_b.copy()
        row_b[:] = row_w
        row_w[:] = row_g
        row_g[:] = np.flip(row_y)
        row_y[:] = np.flip(temp)


    def rotate_ri(self, layer):
        n = self.n
        fields = self.fields

        if layer==0:
            self.__rotate_field_counter_clockwise_4x4(fields[4])
        elif layer==n-1:
            self.__rotate_field_clockwise_4x4(fields[5])


        row_r = fields[3, :, n-1-layer]
        row_w = fields[0, :, n-1-layer]
        row_o = fields[2, :, n-1-layer]
        row_y = fields[1, :, n-1-layer]

        temp = row_r.copy()
        row_r[:] = row_y
        row_y[:] = row_o
        row_o[:] = row_w
        row_w[:] = temp


    def mix_cube(self):
        possible_moves = self.possible_moves
        random_move_lst = np.random.choice(possible_moves, size=(100, ))
        print("random_move_lst: {}".format(random_move_lst))
        for m in random_move_lst:
            self.moving_table[m]()


    def rotate_moves_lst(self, moves_lst, rotation):
        assert rotation in self.moving_dicts
        d = self.moving_dicts[rotation]
        return [d[m] for m in moves_lst]


    def inverse_moves_lst(self, moves_lst):
        return [m if "2" in m else m.replace("'", "") if "'" in m else m+"'" for m in reversed(moves_lst)]


    def solve_cube(self):
        # TODO: !!!!!
        # WRITE THE MOST SIMPLEST WAY FOR SOLVING THE 4x4x4 CUBE!!!!

        moving_table_solve_centers = [
            (((0, 1, 1), ), [
                (((0, 1, 2), ), ["U'"]),
                (((0, 2, 1), ), ["U"]),
                (((0, 2, 2), ), ["U2"]),
                
                (((1, 1, 1), ), ["l2"]),
                (((1, 1, 2), ), ["D'", "l2"]),
                (((1, 2, 1), ), ["D", "l2"]),
                (((1, 2, 2), ), ["D2", "l2"]),

                (((2, 1, 1), ), ["l'"]),
                (((2, 1, 2), ), ["F'", "l'"]),
                (((2, 2, 1), ), ["F", "l'"]),
                (((2, 2, 2), ), ["F2", "l'"]),

                (((3, 1, 1), ), ["l"]),
                (((3, 1, 2), ), ["B'", "l"]),
                (((3, 2, 1), ), ["B", "l"]),
                (((3, 2, 2), ), ["B2", "l"]),

                (((4, 1, 1), ), ["b"]),
                (((4, 1, 2), ), ["R'", "b"]),
                (((4, 2, 1), ), ["R", "b"]),
                (((4, 2, 2), ), ["R2", "b"]),

                (((5, 1, 1), ), ["b'"]),
                (((5, 1, 2), ), ["L'", "b'"]),
                (((5, 2, 1), ), ["L", "b'"]),
                (((5, 2, 2), ), ["L2", "b'"]),
            ]),

            (((0, 1, 2), ), [
                (((0, 2, 1), ), ["U"]),
                (((0, 2, 2), ), ["r", "B'", "r'"]),
                
                (((1, 1, 1), ), ["D", "r2"]),
                (((1, 1, 2), ), ["r2"]),
                (((1, 2, 1), ), ["D2", "r2"]),
                (((1, 2, 2), ), ["D'", "r2"]),

                (((2, 1, 1), ), ["F", "r"]),
                (((2, 1, 2), ), ["r"]),
                (((2, 2, 1), ), ["F2", "r"]),
                (((2, 2, 2), ), ["F'", "r"]),

                (((3, 1, 1), ), ["B", "r'"]),
                (((3, 1, 2), ), ["r'"]),
                (((3, 2, 1), ), ["B2", "r'"]),
                (((3, 2, 2), ), ["B'", "r'"]),

                (((4, 1, 1), ), ["R'", "f'", "U"]),
                (((4, 1, 2), ), ["R2", "f'", "U"]),
                (((4, 2, 1), ), ["f'", "U"]),
                (((4, 2, 2), ), ["R", "f'", "U"]),

                (((5, 1, 1), ), ["L'", "f", "U"]),
                (((5, 1, 2), ), ["L2", "f", "U"]),
                (((5, 2, 1), ), ["f", "U"]),
                (((5, 2, 2), ), ["L", "f", "U"]),
            ]),

            (((0, 2, 1), ), [
                (((0, 2, 2), ), ["U'"]),
                
                (((1, 1, 1), ), ["D", "f2"]),
                (((1, 1, 2), ), ["f2"]),
                (((1, 2, 1), ), ["D2", "f2"]),
                (((1, 2, 2), ), ["D'", "f2"]),

                (((2, 1, 1), ), ["F", "l", "F'", "l'"]),
                (((2, 1, 2), ), ["l", "F'", "l'"]),
                (((2, 2, 1), ), ["F2", "l", "F'", "l'"]),
                (((2, 2, 2), ), ["F'", "l", "F'", "l'"]),

                (((3, 1, 1), ), ["B", "l'", "B'", "l"]),
                (((3, 1, 2), ), ["l'", "B'", "l"]),
                (((3, 2, 1), ), ["B2", "l'", "B'", "l"]),
                (((3, 2, 2), ), ["B'", "l'", "B'", "l"]),

                (((4, 1, 1), ), ["R'", "f'"]),
                (((4, 1, 2), ), ["R2", "f'"]),
                (((4, 2, 1), ), ["f'"]),
                (((4, 2, 2), ), ["R", "f'"]),

                (((5, 1, 1), ), ["L'", "f"]),
                (((5, 1, 2), ), ["L2", "f"]),
                (((5, 2, 1), ), ["f"]),
                (((5, 2, 2), ), ["L", "f"]),
            ]),

            (((0, 2, 2), ), [
                (((1, 1, 1), ), ["r2", "D", "r2"]),
                (((1, 1, 2), ), ["D'", "r2", "D", "r2"]),
                (((1, 2, 1), ), ["D", "r2", "D", "r2"]),
                (((1, 2, 2), ), ["D2", "r2", "D", "r2"]),

                (((2, 1, 1), ), ["r'", "F", "r"]),
                (((2, 1, 2), ), ["F'", "r'", "F", "r"]),
                (((2, 2, 1), ), ["F", "r'", "F", "r"]),
                (((2, 2, 2), ), ["F2", "r'", "F", "r"]),

                (((3, 1, 1), ), ["r", "B", "r'"]),
                (((3, 1, 2), ), ["B'", "r", "B", "r'"]),
                (((3, 2, 1), ), ["B", "r", "B", "r'"]),
                (((3, 2, 2), ), ["B2", "r", "B", "r'"]),

                (((4, 1, 1), ), ["f", "R'", "f'"]),
                (((4, 1, 2), ), ["R'", "f", "R'", "f'"]),
                (((4, 2, 1), ), ["R", "f", "R'", "f'"]),
                (((4, 2, 2), ), ["R2", "f", "R'", "f'"]),

                (((5, 1, 1), ), ["f'", "L'", "f"]),
                (((5, 1, 2), ), ["L'", "f'", "L'", "f"]),
                (((5, 2, 1), ), ["L", "f'", "L'", "f"]),
                (((5, 2, 2), ), ["L2", "f'", "L'", "f"]),
            ]),

            (((1, 1, 1), ), [
                (((1, 1, 2), ), ["D'"]),
                (((1, 2, 1), ), ["D"]),
                (((1, 2, 2), ), ["D2"]),

                (((2, 1, 1), ), ["F", "r'", "D'", "r"]),
                (((2, 1, 2), ), ["r'", "D'", "r"]),
                (((2, 2, 1), ), ["F2", "r'", "D'", "r"]),
                (((2, 2, 2), ), ["F'", "r'", "D'", "r"]),

                (((3, 1, 1), ), ["B", "r", "D'", "r'"]),
                (((3, 1, 2), ), ["r", "D'", "r'"]),
                (((3, 2, 1), ), ["B2", "r", "D'", "r'"]),
                (((3, 2, 2), ), ["B'", "r", "D'", "r'"]),

                (((4, 1, 1), ), ["R", "b'", "D", "b"]),
                (((4, 1, 2), ), ["b'", "D", "b"]),
                (((4, 2, 1), ), ["R2", "b'", "D", "b"]),
                (((4, 2, 2), ), ["R'", "b'", "D", "b"]),

                (((5, 1, 1), ), ["L", "b", "D", "b'"]),
                (((5, 1, 2), ), ["b", "D", "b'"]),
                (((5, 2, 1), ), ["L2", "b", "D", "b'"]),
                (((5, 2, 2), ), ["L'", "b", "D", "b'"]),
            ]),

            (((1, 1, 2), ), [
                (((1, 2, 1), ), ["D"]),
                (((1, 2, 2), ), ["r", "F'", "r'"]),

                (((2, 1, 1), ), ["r", "F", "r'"]),
                (((2, 1, 2), ), ["F'", "r", "F", "r'"]),
                (((2, 2, 1), ), ["F", "r", "F", "r'"]),
                (((2, 2, 2), ), ["F2", "r", "F", "r'"]),

                (((3, 1, 1), ), ["r'", "B", "r"]),
                (((3, 1, 2), ), ["B'", "r'", "B", "r"]),
                (((3, 2, 1), ), ["B", "r'", "B", "r"]),
                (((3, 2, 2), ), ["B2", "r'", "B", "r"]),

                (((4, 1, 1), ), ["R", "b'", "D", "b"]),
                (((4, 1, 2), ), ["b'", "D", "b"]),
                (((4, 2, 1), ), ["R2", "b'", "D", "b"]),
                (((4, 2, 2), ), ["R'", "b'", "D", "b"]),

                (((5, 1, 1), ), ["L", "b", "D", "b'"]),
                (((5, 1, 2), ), ["b", "D", "b'"]),
                (((5, 2, 1), ), ["L2", "b", "D", "b'"]),
                (((5, 2, 2), ), ["L'", "b", "D", "b'"]),
            ]),

            (((1, 2, 1), ), [
                (((1, 2, 2), ), ["D'"]),

                (((2, 1, 1), ), ["F", "l'", "F'", "l"]),
                (((2, 1, 2), ), ["l'", "F'", "l"]),
                (((2, 2, 1), ), ["F2", "l'", "F'", "l"]),
                (((2, 2, 2), ), ["F'", "l'", "F'", "l"]),

                (((3, 1, 1), ), ["B", "l", "B'", "l'"]),
                (((3, 1, 2), ), ["l", "B'", "l'"]),
                (((3, 2, 1), ), ["B2", "l", "B'", "l'"]),
                (((3, 2, 2), ), ["B'", "l", "B'", "l'"]),

                (((4, 1, 1), ), ["R2", "b", "R'", "b'"]),
                (((4, 1, 2), ), ["R", "b", "R'", "b'"]),
                (((4, 2, 1), ), ["R'", "b", "R'", "b'"]),
                (((4, 2, 2), ), ["b", "R'", "b'"]),

                (((5, 1, 1), ), ["L2", "b'", "L'", "b"]),
                (((5, 1, 2), ), ["L", "b'", "L'", "b"]),
                (((5, 2, 1), ), ["L'", "b'", "L'", "b"]),
                (((5, 2, 2), ), ["b'", "L'", "b"]),
            ]),

            (((1, 2, 2), ), [
                (((2, 1, 1), ), ["r", "F", "r'"]),
                (((2, 1, 2), ), ["F'", "r", "F", "r'"]),
                (((2, 2, 1), ), ["F", "r", "F", "r'"]),
                (((2, 2, 2), ), ["F2", "r", "F", "r'"]),

                (((3, 1, 1), ), ["r'", "B", "r"]),
                (((3, 1, 2), ), ["B'", "r'", "B", "r"]),
                (((3, 2, 1), ), ["B", "r'", "B", "r"]),
                (((3, 2, 2), ), ["B2", "r'", "B", "r"]),

                (((4, 1, 1), ), ["R2", "b", "R'", "b'"]),
                (((4, 1, 2), ), ["R", "b", "R'", "b'"]),
                (((4, 2, 1), ), ["R'", "b", "R'", "b'"]),
                (((4, 2, 2), ), ["b", "R'", "b'"]),

                (((5, 1, 1), ), ["L2", "b'", "L'", "b"]),
                (((5, 1, 2), ), ["L", "b'", "L'", "b"]),
                (((5, 2, 1), ), ["L'", "b'", "L'", "b"]),
                (((5, 2, 2), ), ["b'", "L'", "b"]),
            ]),

            (((2, 1, 1), ), [
                (((2, 1, 2), ), ["F'"]),
                (((2, 2, 1), ), ["F"]),
                (((2, 2, 2), ), ["F2"]),

                (((3, 1, 1), ), ["B2", "u2"]),
                (((3, 1, 2), ), ["B", "u2"]),
                (((3, 2, 1), ), ["B'", "u2"]),
                (((3, 2, 2), ), ["u2"]),

                (((4, 1, 1), ), ["R'", "u"]),
                (((4, 1, 2), ), ["R2", "u"]),
                (((4, 2, 1), ), ["u"]),
                (((4, 2, 2), ), ["R", "u"]),

                (((5, 1, 1), ), ["L", "u'"]),
                (((5, 1, 2), ), ["u'"]),
                (((5, 2, 1), ), ["L2", "u'"]),
                (((5, 2, 2), ), ["L'", "u'"]),
            ]),

            (((2, 1, 2), ), [
                (((2, 2, 1), ), ["F"]),
                (((2, 2, 2), ), ["d", "F", "d'", "F'"]),

                (((3, 1, 1), ), ["B", "d2", "F"]),
                (((3, 1, 2), ), ["d2", "F"]),
                (((3, 2, 1), ), ["B2", "d2", "F"]),
                (((3, 2, 2), ), ["B'", "d2", "F"]),

                (((4, 1, 1), ), ["R2", "d'", "F"]),
                (((4, 1, 2), ), ["R", "d'", "F"]),
                (((4, 2, 1), ), ["R'", "d'", "F"]),
                (((4, 2, 2), ), ["d'", "F"]),

                (((5, 1, 1), ), ["d", "F"]),
                (((5, 1, 2), ), ["L'", "d", "F"]),
                (((5, 2, 1), ), ["L", "d", "F"]),
                (((5, 2, 2), ), ["L2", "d", "F"]),
            ]),

            (((2, 2, 1), ), [
                (((2, 2, 2), ), ["F'"]),

                (((3, 1, 1), ), ["B", "d2"]),
                (((3, 1, 2), ), ["d2"]),
                (((3, 2, 1), ), ["B2", "d2"]),
                (((3, 2, 2), ), ["B'", "d2"]),

                (((4, 1, 1), ), ["R2", "d'"]),
                (((4, 1, 2), ), ["R", "d'"]),
                (((4, 2, 1), ), ["R'", "d'"]),
                (((4, 2, 2), ), ["d'"]),

                (((5, 1, 1), ), ["d"]),
                (((5, 1, 2), ), ["L'", "d"]),
                (((5, 2, 1), ), ["L", "d"]),
                (((5, 2, 2), ), ["L2", "d"]),
            ]),

            (((2, 2, 2), ), [
                (((3, 1, 1), ), ["B2", "d2", "B'", "d2"]),
                (((3, 1, 2), ), ["B", "d2", "B'", "d2"]),
                (((3, 2, 1), ), ["B'", "d2", "B'", "d2"]),
                (((3, 2, 2), ), ["d2", "B'", "d2"]),

                (((4, 1, 1), ), ["R'", "d", "R'", "d'"]),
                (((4, 1, 2), ), ["R2", "d", "R'", "d'"]),
                (((4, 2, 1), ), ["d", "R'", "d'"]),
                (((4, 2, 2), ), ["R", "d", "R'", "d'"]),

                (((5, 1, 1), ), ["L", "d'", "L'", "d"]),
                (((5, 1, 2), ), ["d'", "L'", "d"]),
                (((5, 2, 1), ), ["L2", "d'", "L'", "d"]),
                (((5, 2, 2), ), ["L'", "d'", "L'", "d"]),
            ]),

            (((4, 1, 1), ), [
                (((4, 1, 2), ), ["R'"]),
                (((4, 2, 1), ), ["R"]),
                (((4, 2, 2), ), ["R2"]),

                (((3, 1, 1), ), ["u'", "B'", "u"]),
                (((3, 1, 2), ), ["B'", "u'", "B'", "u"]),
                (((3, 2, 1), ), ["B", "u'", "B'", "u"]),
                (((3, 2, 2), ), ["B2", "u'", "B'", "u"]),

                (((5, 1, 1), ), ["L'", "u2", "L'", "u2"]),
                (((5, 1, 2), ), ["L2", "u2", "L'", "u2"]),
                (((5, 2, 1), ), ["u2", "L'", "u2"]),
                (((5, 2, 2), ), ["L", "u2", "L'", "u2"]),
            ]),

            (((4, 2, 1), ), [
                (((4, 2, 1), ), []),
                (((4, 1, 2), ), ["R'"]),
                (((4, 2, 2), ), ["d", "B'", "d'", "R'"]),

                (((3, 1, 1), ), ["B'", "d", "B", "d'", "R'"]),
                (((3, 1, 2), ), ["B2", "d", "B", "d'", "R'"]),
                (((3, 2, 1), ), ["d", "B", "d'", "R'"]),
                (((3, 2, 2), ), ["B", "d", "B", "d'", "R'"]),

                (((5, 1, 1), ), ["L2", "d2", "L", "d2", "R'"]),
                (((5, 1, 2), ), ["L", "d2", "L", "d2", "R'"]),
                (((5, 2, 1), ), ["L'", "d2", "L", "d2", "R'"]),
                (((5, 2, 2), ), ["d2", "L", "d2", "R'"]),
            ]),

            (((4, 1, 2), ), [
                (((4, 2, 2), ), ["R"]),
                (((4, 1, 2), ), []),

                (((3, 1, 1), ), ["B'", "d", "B", "d'"]),
                (((3, 1, 2), ), ["B2", "d", "B", "d'"]),
                (((3, 2, 1), ), ["d", "B", "d'"]),
                (((3, 2, 2), ), ["B", "d", "B", "d'"]),

                (((5, 1, 1), ), ["L2", "d2", "L", "d2"]),
                (((5, 1, 2), ), ["L", "d2", "L", "d2"]),
                (((5, 2, 1), ), ["L'", "d2", "L", "d2"]),
                (((5, 2, 2), ), ["d2", "L", "d2"]),
            ]),

            (((4, 2, 2), ), [
                (((3, 1, 1), ), ["B'", "d", "B", "d'"]),
                (((3, 1, 2), ), ["B2", "d", "B", "d'"]),
                (((3, 2, 1), ), ["d", "B", "d'"]),
                (((3, 2, 2), ), ["B", "d", "B", "d'"]),

                (((5, 1, 1), ), ["L2", "d2", "L", "d2"]),
                (((5, 1, 2), ), ["L", "d2", "L", "d2"]),
                (((5, 2, 1), ), ["L'", "d2", "L", "d2"]),
                (((5, 2, 2), ), ["d2", "L", "d2"]),
            ]),

            (((3, 2, 1), ), [
                (((3, 1, 1), ), ["B'"]),
                (((3, 1, 2), ), ["B2"]),
                (((3, 2, 1), ), []),
                (((3, 2, 2), ), ["B"]),

                (((5, 1, 1), ), ["L2", "d", "L", "d'", "B'"]),
                (((5, 1, 2), ), ["L", "d", "L", "d'", "B'"]),
                (((5, 2, 1), ), ["L'", "d", "L", "d'", "B'"]),
                (((5, 2, 2), ), ["d", "L", "d'", "B'"]),
            ]),

            (((3, 2, 2), ), [
                (((3, 1, 1), ), ["B'"]),
                (((3, 1, 2), ), ["d", "L'", "d'", "B'"]),
                (((3, 2, 2), ), []),

                (((5, 1, 1), ), ["L2", "d", "L", "d'", "B'"]),
                (((5, 1, 2), ), ["L", "d", "L", "d'", "B'"]),
                (((5, 2, 1), ), ["L'", "d", "L", "d'", "B'"]),
                (((5, 2, 2), ), ["d", "L", "d'", "B'"]),
            ]),

            (((3, 1, 1), (3, 1, 2)), [
                (((3, 1, 1), (3, 1, 2)), []),
                
                (((3, 1, 1), (5, 1, 1)), ["L2", "d", "L", "d'"]),
                (((3, 1, 1), (5, 1, 2)), ["L", "d", "L", "d'"]),
                (((3, 1, 1), (5, 2, 1)), ["L'", "d", "L", "d'"]),
                (((3, 1, 1), (5, 2, 2)), ["d", "L", "d'"]),

                (((3, 1, 2), (5, 1, 1)), ["L", "d", "L'", "d'"]),
                (((3, 1, 2), (5, 1, 2)), ["d", "L'", "d'"]),
                (((3, 1, 2), (5, 2, 1)), ["L2", "d", "L'", "d'"]),
                (((3, 1, 2), (5, 2, 2)), ["L'", "d", "L'", "d'"]),
                
                (((5, 1, 1), (5, 1, 2)), ["L", "d", "L2", "d'"]),
                (((5, 1, 2), (5, 2, 2)), ["d", "L2", "d'"]),
                (((5, 2, 2), (5, 2, 1)), ["L'", "d", "L2", "d'"]),
                (((5, 2, 1), (5, 1, 1)), ["L2", "d", "L2", "d'"]),

                (((5, 2, 1), (5, 1, 2)), ["d", "L", "d'", "L'", "d", "L2", "d'"]),
                (((5, 1, 1), (5, 2, 2)), ["d", "L'", "d'", "L", "d", "L2", "d'"]),
            ]),

            (((5, 1, 1), (5, 1, 2), (5, 2, 1), (5, 2, 2)), [])

            # TODO: add edges positions + color combinations!
        ]

        fields = self.fields
        cells_lst = []
        for cells, moving_lst in moving_table_solve_centers:
            # if cells==((3, 2, 2), ) or cells==((3, 1, 1), (3, 1, 2)) or cells==((4, 2, 2), ) or cells==((4, 2, 1), ) or cells==((4, 1, 2), ) or cells==((4, 1, 1), ):
            #     print("Printing the found cells! cells: {}".format(cells))
            #     self.print_field()

            for pos_col, moves_lst in moving_lst:
                is_pos_col = True
                for pos, cell in zip(pos_col, cells):
                    if fields[pos]!=cell[0]:
                        is_pos_col = False
                        break
                # if fields[pos_col[0]]==cells[0][0]:
                if is_pos_col:
                    # print("pos_col: {}".format(pos_col))
                    # print("moves_lst: {}".format(moves_lst))
                    print("cells: {}, pos_col: {}, moves_lst: {}".format(cells, pos_col, moves_lst))
                    self.apply_move_lst(moves_lst)
                    break

            try:
                for cell in cells:
                    assert fields[cell]==cell[0]
            except Exception as e:
                print(e)

                print("for cell: {}, cells: {}".format(cell, cells))
                self.print_field()
                sys.exit(-1)


            for cells in cells_lst:
                for cell in cells:
                    assert fields[cell]==cell[0]
            
            cells_lst.append(cells)


        # solve the pairs of the cube!
        lst_basic_moves_1 = ["U", "D", "F", "B", "R", "L"]
        lst_possible_moves_1 = lst_basic_moves_1+[v+"'" for v in lst_basic_moves_1]+[v+"2" for v in lst_basic_moves_1]
        lst_basic_moves_2 = ["D", "B", "R", "L"]
        lst_possible_moves_2 = lst_basic_moves_2+[v+"'" for v in lst_basic_moves_2]+[v+"2" for v in lst_basic_moves_2]
        lst_basic_moves_3 = ["D", "R", "L"]
        lst_possible_moves_3 = lst_basic_moves_3+[v+"'" for v in lst_basic_moves_3]+[v+"2" for v in lst_basic_moves_3]

        f = self.fields
        while self.count_finished_pairs() <= 8:
            while (0+np.all(f[0, -1, 1]==f[0, -1, 1:-1])+np.all(f[2, 0, 1]==f[2, 0, 1:-1]))==2:
                self.apply_move_lst([lst_possible_moves_1[np.random.randint(0, len(lst_possible_moves_1))]])
            while ((f[0, 3, 1]!=f[3, 3, 2]) or (f[2, 0, 1]!=f[0, 0, 2])):
                self.apply_move_lst([lst_possible_moves_2[np.random.randint(0, len(lst_possible_moves_2))]])
            while (0+np.all(f[0, 1, -1]==f[0, 1:-1, -1])+np.all(f[4, 1, 0]==f[4, 1:-1, 0]))==2:
                self.apply_move_lst([lst_possible_moves_3[np.random.randint(0, len(lst_possible_moves_3))]])
            
            self.apply_move_lst(["l'", "B'", "R", "B", "l"])
            print("found one not a pair!")
        
        if self.count_finished_pairs() <= 10:
            while (0+np.all(f[0, -1, 1]==f[0, -1, 1:-1])+np.all(f[2, 0, 1]==f[2, 0, 1:-1]))==2:
                self.apply_move_lst([lst_possible_moves_1[np.random.randint(0, len(lst_possible_moves_1))]])
            while ((f[0, 3, 1]!=f[3, 3, 2]) or (f[2, 0, 1]!=f[0, 0, 2])):
                self.apply_move_lst([lst_possible_moves_2[np.random.randint(0, len(lst_possible_moves_2))]])
            if self.count_finished_pairs() == 9:
                # do the algorithm for pairing the 3 pairs!
                while (0+np.all(f[0, 1, -1]==f[0, 1:-1, -1])+np.all(f[4, 1, 0]==f[4, 1:-1, 0]))==2:
                    self.apply_move_lst([lst_possible_moves_3[np.random.randint(0, len(lst_possible_moves_3))]])
                # self.apply_move_lst(["l'", "B'", "R", "B", "l"])
                print("do the pairing of three pairs!")
                if (f[0, 0, 1]==f[4, 1, 0]) and (f[3, 3, 1]==f[0, 1, 3]):
                    # do the first algo!
                    self.apply_move_lst(["l'", "B'", "R", "B", "l"])
                else:
                    # do the second algo!
                    self.apply_move_lst(["r'", "F", "R'", "F'", "r"])
            elif self.count_finished_pairs() == 10:
                # do the algorithm for pairing only 2 pairs together!
                print("do the pairing of two pairs!")
                self.apply_move_lst(["B'", "U", "R'", "f", "R'", "F'", "R", "f'", "U'", "R'", "U", "b", "U'", "F'", "U", "b'"])

        assert self.count_finished_pairs()==12


        n = self.n
        # dict_nxn_to_3x3, dnt3
        d = {
            (0, 0, 0): (0, 0, 0),
            (0, 0, 1): (0, 0, 1),
            (0, 0, 2): (0, 0, n-1),
            (0, 1, 0): (0, 1, 0),
            (0, 1, 1): (0, 1, 1),
            (0, 1, 2): (0, 1, n-1),
            (0, 2, 0): (0, n-1, 0),
            (0, 2, 1): (0, n-1, 1),
            (0, 2, 2): (0, n-1, n-1),

            (1, 0, 0): (1, 0, 0),
            (1, 0, 1): (1, 0, 1),
            (1, 0, 2): (1, 0, n-1),
            (1, 1, 0): (1, 1, 0),
            (1, 1, 1): (1, 1, 1),
            (1, 1, 2): (1, 1, n-1),
            (1, 2, 0): (1, n-1, 0),
            (1, 2, 1): (1, n-1, 1),
            (1, 2, 2): (1, n-1, n-1),

            (2, 0, 0): (2, 0, 0),
            (2, 0, 1): (2, 0, 1),
            (2, 0, 2): (2, 0, n-1),
            (2, 1, 0): (2, 1, 0),
            (2, 1, 1): (2, 1, 1),
            (2, 1, 2): (2, 1, n-1),
            (2, 2, 0): (2, n-1, 0),
            (2, 2, 1): (2, n-1, 1),
            (2, 2, 2): (2, n-1, n-1),

            (3, 0, 0): (3, 0, 0),
            (3, 0, 1): (3, 0, 1),
            (3, 0, 2): (3, 0, n-1),
            (3, 1, 0): (3, 1, 0),
            (3, 1, 1): (3, 1, 1),
            (3, 1, 2): (3, 1, n-1),
            (3, 2, 0): (3, n-1, 0),
            (3, 2, 1): (3, n-1, 1),
            (3, 2, 2): (3, n-1, n-1),

            (4, 0, 0): (4, 0, 0),
            (4, 0, 1): (4, 0, 1),
            (4, 0, 2): (4, 0, n-1),
            (4, 1, 0): (4, 1, 0),
            (4, 1, 1): (4, 1, 1),
            (4, 1, 2): (4, 1, n-1),
            (4, 2, 0): (4, n-1, 0),
            (4, 2, 1): (4, n-1, 1),
            (4, 2, 2): (4, n-1, n-1),

            (5, 0, 0): (5, 0, 0),
            (5, 0, 1): (5, 0, 1),
            (5, 0, 2): (5, 0, n-1),
            (5, 1, 0): (5, 1, 0),
            (5, 1, 1): (5, 1, 1),
            (5, 1, 2): (5, 1, n-1),
            (5, 2, 0): (5, n-1, 0),
            (5, 2, 1): (5, n-1, 1),
            (5, 2, 2): (5, n-1, n-1),
        }

        di = {v: k for k, v in d.items()}

        move_tbl_exchange_edge_left = ["U'", "L'", "U", "L", "U", "F", "U'", "F'"]
        move_tbl_exchange_edge_right = ["U", "R", "U'", "R'", "U'", "F'", "U", "F"]

        move_tbl_exchange_edge_l_D_F = self.rotate_moves_lst(self.rotate_moves_lst(move_tbl_exchange_edge_left, "F"), "F")
        move_tbl_exchange_edge_r_D_F = self.rotate_moves_lst(self.rotate_moves_lst(move_tbl_exchange_edge_right, "F"), "F")

        move_tbl_exchange_edge_l_D_R = self.rotate_moves_lst(move_tbl_exchange_edge_l_D_F, "U'")
        move_tbl_exchange_edge_r_D_R = self.rotate_moves_lst(move_tbl_exchange_edge_r_D_F, "U'")
        
        move_tbl_exchange_edge_l_D_B = self.rotate_moves_lst(move_tbl_exchange_edge_l_D_R, "U'")
        move_tbl_exchange_edge_r_D_B = self.rotate_moves_lst(move_tbl_exchange_edge_r_D_R, "U'")

        move_tbl_exchange_edge_l_D_L = self.rotate_moves_lst(move_tbl_exchange_edge_l_D_B, "U'")
        move_tbl_exchange_edge_r_D_L = self.rotate_moves_lst(move_tbl_exchange_edge_r_D_B, "U'")

        move_tbl_exchange_edge_left_inv = self.inverse_moves_lst(move_tbl_exchange_edge_left)
        move_tbl_exchange_edge_right_inv = self.inverse_moves_lst(move_tbl_exchange_edge_right)

        # print("move_tbl_exchange_edge_left: {}".format(move_tbl_exchange_edge_left))
        # print("move_tbl_exchange_edge_right: {}".format(move_tbl_exchange_edge_right))
        # print("move_tbl_exchange_edge_left_inv: {}".format(move_tbl_exchange_edge_left_inv))
        # print("move_tbl_exchange_edge_right_inv: {}".format(move_tbl_exchange_edge_right_inv))
        # print("move_tbl_exchange_edge_l_D_R: {}".format(move_tbl_exchange_edge_l_D_R))
        # print("move_tbl_exchange_edge_r_D_R: {}".format(move_tbl_exchange_edge_r_D_R))
        # sys.exit(-1234567)

        move_tbl_exchange_edge_l_D_F_inv = self.rotate_moves_lst(self.rotate_moves_lst(move_tbl_exchange_edge_left_inv, "F"), "F")
        move_tbl_exchange_edge_r_D_F_inv = self.rotate_moves_lst(self.rotate_moves_lst(move_tbl_exchange_edge_right_inv, "F"), "F")

        move_tbl_exchange_edge_l_D_R_inv = self.rotate_moves_lst(move_tbl_exchange_edge_l_D_F_inv, "U'")
        move_tbl_exchange_edge_r_D_R_inv = self.rotate_moves_lst(move_tbl_exchange_edge_r_D_F_inv, "U'")
        
        move_tbl_exchange_edge_l_D_B_inv = self.rotate_moves_lst(move_tbl_exchange_edge_l_D_R_inv, "U'")
        move_tbl_exchange_edge_r_D_B_inv = self.rotate_moves_lst(move_tbl_exchange_edge_r_D_R_inv, "U'")

        move_tbl_exchange_edge_l_D_L_inv = self.rotate_moves_lst(move_tbl_exchange_edge_l_D_B_inv, "U'")
        move_tbl_exchange_edge_r_D_L_inv = self.rotate_moves_lst(move_tbl_exchange_edge_r_D_B_inv, "U'")

        moving_table_solve_3x3 = [
            ((d[(0, 2, 1)], d[(2, 0, 1)]), [
                ((d[(0, 2, 1)], d[(2, 0, 1)]), []),
                ((d[(0, 1, 0)], d[(5, 1, 2)]), ["U'"]),
                ((d[(0, 0, 1)], d[(3, 2, 1)]), ["U2"]),
                ((d[(0, 1, 2)], d[(4, 1, 0)]), ["U"]),

                ((d[(2, 0, 1)], d[(0, 2, 1)]), ["F", "R", "U"]),
                ((d[(5, 1, 2)], d[(0, 1, 0)]), ["L", "F"]),
                ((d[(3, 2, 1)], d[(0, 0, 1)]), ["B'", "R'", "U"]),
                ((d[(4, 1, 0)], d[(0, 1, 2)]), ["R'", "F'"]),

                ((d[(1, 0, 1)], d[(2, 2, 1)]), ["F2"]),
                ((d[(1, 1, 2)], d[(4, 1, 2)]), ["D'", "F2"]),
                ((d[(1, 2, 1)], d[(3, 0, 1)]), ["D2", "F2"]),
                ((d[(1, 1, 0)], d[(5, 1, 0)]), ["D", "F2"]),

                ((d[(2, 2, 1)], d[(1, 0, 1)]), ["F'", "R", "U"]),
                ((d[(4, 1, 2)], d[(1, 1, 2)]), ["R", "F'"]),
                ((d[(3, 0, 1)], d[(1, 2, 1)]), ["B", "R'", "U"]),
                ((d[(5, 1, 0)], d[(1, 1, 0)]), ["L'", "F"]),

                ((d[(2, 1, 2)], d[(4, 2, 1)]), ["R", "U"]),
                ((d[(4, 0, 1)], d[(3, 1, 2)]), ["B", "U2"]),
                ((d[(3, 1, 0)], d[(5, 0, 1)]), ["L", "U'"]),
                ((d[(5, 2, 1)], d[(2, 1, 0)]), ["F"]),

                ((d[(4, 2, 1)], d[(2, 1, 2)]), ["F'"]),
                ((d[(3, 1, 2)], d[(4, 0, 1)]), ["R'", "U"]),
                ((d[(5, 0, 1)], d[(3, 1, 0)]), ["B'", "U2"]),
                ((d[(2, 1, 0)], d[(5, 2, 1)]), ["L'", "U'"]),
            ]),

            ((d[(0, 1, 2)], d[(4, 1, 0)]), [
                ((d[(0, 1, 2)], d[(4, 1, 0)]), []),
                ((d[(0, 0, 1)], d[(3, 2, 1)]), ["B", "U'", "B'", "U"]),
                ((d[(0, 1, 0)], d[(5, 1, 2)]), ["L", "U2", "L'", "U2"]),

                ((d[(4, 1, 0)], d[(0, 1, 2)]), ["R'", "U", "F'", "U'"]),
                ((d[(3, 2, 1)], d[(0, 0, 1)]), ["B'", "R'"]),
                ((d[(5, 1, 2)], d[(0, 1, 0)]), ["L", "U", "F", "U'"]),

                ((d[(1, 0, 1)], d[(2, 2, 1)]), ["D","R2"]),
                ((d[(1, 1, 2)], d[(4, 1, 2)]), ["R2"]),
                ((d[(1, 2, 1)], d[(3, 0, 1)]), ["D'", "R2"]),
                ((d[(1, 1, 0)], d[(5, 1, 0)]), ["D2", "R2"]),

                ((d[(2, 2, 1)], d[(1, 0, 1)]), ["F'", "R", "F"]),
                ((d[(4, 1, 2)], d[(1, 1, 2)]), ["R", "U", "F'", "U'"]),
                ((d[(3, 0, 1)], d[(1, 2, 1)]), ["B", "R'"]),
                ((d[(5, 1, 0)], d[(1, 1, 0)]), ["L", "U'", "B'", "U"]),

                ((d[(2, 1, 2)], d[(4, 2, 1)]), ["R"]),
                ((d[(4, 0, 1)], d[(3, 1, 2)]), ["U'", "B", "U"]),
                ((d[(3, 1, 0)], d[(5, 0, 1)]), ["U2", "L", "U2"]),
                ((d[(5, 2, 1)], d[(2, 1, 0)]), ["U", "F", "U'"]),

                ((d[(4, 2, 1)], d[(2, 1, 2)]), ["U", "F'", "U'"]),
                ((d[(3, 1, 2)], d[(4, 0, 1)]), ["R'"]),
                ((d[(5, 0, 1)], d[(3, 1, 0)]), ["U'", "B'", "U"]),
                ((d[(2, 1, 0)], d[(5, 2, 1)]), ["U2", "L'", "U2"]),
            ]),

            ((d[(0, 0, 1)], d[(3, 2, 1)]), [
                ((d[(0, 0, 1)], d[(3, 2, 1)]), []),
                ((d[(0, 1, 0)], d[(5, 1, 2)]), ["L", "U'", "L'", "U"]),

                ((d[(3, 2, 1)], d[(0, 0, 1)]), ["B'", "U", "R'", "U'"]),
                ((d[(5, 1, 2)], d[(0, 1, 0)]), ["L'", "B'"]),

                ((d[(1, 0, 1)], d[(2, 2, 1)]), ["D2","B2"]),
                ((d[(1, 1, 2)], d[(4, 1, 2)]), ["D", "B2"]),
                ((d[(1, 2, 1)], d[(3, 0, 1)]), ["B2"]),
                ((d[(1, 1, 0)], d[(5, 1, 0)]), ["D'", "B2"]),

                ((d[(2, 2, 1)], d[(1, 0, 1)]), ["D'", "L", "B'"]),
                ((d[(4, 1, 2)], d[(1, 1, 2)]), ["D2", "L", "B'"]),
                ((d[(3, 0, 1)], d[(1, 2, 1)]), ["D", "L", "B'"]),
                ((d[(5, 1, 0)], d[(1, 1, 0)]), ["L", "B'"]),

                ((d[(2, 1, 2)], d[(4, 2, 1)]), ["U", "R", "U'"]),
                ((d[(4, 0, 1)], d[(3, 1, 2)]), ["B"]),
                ((d[(3, 1, 0)], d[(5, 0, 1)]), ["U'", "L", "U"]),
                ((d[(5, 2, 1)], d[(2, 1, 0)]), ["U2", "F", "U2"]),

                ((d[(4, 2, 1)], d[(2, 1, 2)]), ["U2", "F'", "U2"]),
                ((d[(3, 1, 2)], d[(4, 0, 1)]), ["U", "R'", "U'"]),
                ((d[(5, 0, 1)], d[(3, 1, 0)]), ["B'"]),
                ((d[(2, 1, 0)], d[(5, 2, 1)]), ["U'", "L'", "U"]),
            ]),

            ((d[(0, 1, 0)], d[(5, 1, 2)]), [
                ((d[(0, 1, 0)], d[(5, 1, 2)]), []),

                ((d[(5, 1, 2)], d[(0, 1, 0)]), ["L'", "U", "B'", "U'"]),

                ((d[(1, 0, 1)], d[(2, 2, 1)]), ["D'","L2"]),
                ((d[(1, 1, 2)], d[(4, 1, 2)]), ["D2", "L2"]),
                ((d[(1, 2, 1)], d[(3, 0, 1)]), ["D", "L2"]),
                ((d[(1, 1, 0)], d[(5, 1, 0)]), ["L2"]),

                ((d[(2, 2, 1)], d[(1, 0, 1)]), ["F", "L'", "F'"]),
                ((d[(4, 1, 2)], d[(1, 1, 2)]), ["D'", "F", "L'", "F'"]),
                ((d[(3, 0, 1)], d[(1, 2, 1)]), ["B'", "L", "B"]),
                ((d[(5, 1, 0)], d[(1, 1, 0)]), ["L", "U", "B'", "U'"]),

                ((d[(2, 1, 2)], d[(4, 2, 1)]), ["U2", "R", "U2"]),
                ((d[(4, 0, 1)], d[(3, 1, 2)]), ["U", "B", "U'"]),
                ((d[(3, 1, 0)], d[(5, 0, 1)]), ["L"]),
                ((d[(5, 2, 1)], d[(2, 1, 0)]), ["U'", "F", "U"]),

                ((d[(4, 2, 1)], d[(2, 1, 2)]), ["U'", "F'", "U"]),
                ((d[(3, 1, 2)], d[(4, 0, 1)]), ["U2", "R'", "U2"]),
                ((d[(5, 0, 1)], d[(3, 1, 0)]), ["U", "B'", "U'"]),
                ((d[(2, 1, 0)], d[(5, 2, 1)]), ["L'"]),
            ]),

            ((d[(0, 2, 2)], d[(2, 0, 2)], d[(4, 2, 0)]), [
                ((d[(0, 2, 2)], d[(2, 0, 2)], d[(4, 2, 0)]), []),
                ((d[(4, 2, 0)], d[(0, 2, 2)], d[(2, 0, 2)]), ["R'", "D'", "R", "D", "R'", "D'", "R"]),
                ((d[(2, 0, 2)], d[(4, 2, 0)], d[(0, 2, 2)]), ["R'", "D", "R", "D'", "R'", "D", "R"]),

                ((d[(0, 2, 0)], d[(5, 2, 2)], d[(2, 0, 0)]), ["F'", "D", "F", "D", "R'", "D'", "R"]),
                ((d[(2, 0, 0)], d[(0, 2, 0)], d[(5, 2, 2)]), ["F'", "D'", "F", "D2", "R'", "D'", "R"]),
                ((d[(5, 2, 2)], d[(2, 0, 0)], d[(0, 2, 0)]), ["L", "D", "L'", "D'", "R'", "D", "R"]),

                ((d[(0, 0, 0)], d[(3, 2, 0)], d[(5, 0, 2)]), ["B", "D2", "B'", "R'", "D'", "R"]),
                ((d[(5, 0, 2)], d[(0, 0, 0)], d[(3, 2, 0)]), ["L'", "D2", "L", "R'", "D'", "R"]),
                ((d[(3, 2, 0)], d[(5, 0, 2)], d[(0, 0, 0)]), ["B", "D", "B'", "R'", "D", "R"]),

                ((d[(0, 0, 2)], d[(4, 0, 0)], d[(3, 2, 2)]), ["B'", "D2", "B", "R'", "D", "R"]),
                ((d[(3, 2, 2)], d[(0, 0, 2)], d[(4, 0, 0)]), ["B'", "D'", "B", "R'", "D'", "R"]),
                ((d[(4, 0, 0)], d[(3, 2, 2)], d[(0, 0, 2)]), ["R", "D2", "R2", "D", "R"]),

                ((d[(1, 0, 2)], d[(4, 2, 2)], d[(2, 2, 2)]), ["R'", "D2", "R", "D", "R'", "D'", "R"]),
                ((d[(2, 2, 2)], d[(1, 0, 2)], d[(4, 2, 2)]), ["D'", "R'", "D", "R"]),
                ((d[(4, 2, 2)], d[(2, 2, 2)], d[(1, 0, 2)]), ["R'", "D'", "R"]),
                
                ((d[(1, 0, 0)], d[(2, 2, 0)], d[(5, 2, 0)]), ["D", "R'", "D2", "R", "D", "R'", "D'", "R"]),
                ((d[(5, 2, 0)], d[(1, 0, 0)], d[(2, 2, 0)]), ["R'", "D", "R"]),
                ((d[(2, 2, 0)], d[(5, 2, 0)], d[(1, 0, 0)]), ["D", "R'", "D'", "R"]),

                ((d[(1, 2, 0)], d[(5, 0, 0)], d[(3, 0, 0)]), ["D2", "R'", "D2", "R", "D", "R'", "D'", "R"]),
                ((d[(3, 0, 0)], d[(1, 2, 0)], d[(5, 0, 0)]), ["R'", "D2", "R"]),
                ((d[(5, 0, 0)], d[(3, 0, 0)], d[(1, 2, 0)]), ["D2", "R'", "D'", "R"]),

                ((d[(1, 2, 2)], d[(3, 0, 2)], d[(4, 0, 2)]), ["D'", "R'", "D2", "R", "D", "R'", "D'", "R"]),
                ((d[(4, 0, 2)], d[(1, 2, 2)], d[(3, 0, 2)]), ["D2", "R'", "D", "R"]),
                ((d[(3, 0, 2)], d[(4, 0, 2)], d[(1, 2, 2)]), ["D'", "R'", "D'", "R"]),
            ]),

            ((d[(0, 2, 0)], d[(5, 2, 2)], d[(2, 0, 0)]), [
                ((d[(0, 2, 0)], d[(5, 2, 2)], d[(2, 0, 0)]), []),
                ((d[(2, 0, 0)], d[(0, 2, 0)], d[(5, 2, 2)]), ["F'", "D'", "F", "D", "F'", "D'", "F"]),
                ((d[(5, 2, 2)], d[(2, 0, 0)], d[(0, 2, 0)]), ["L", "D", "L'", "D2", "F'", "D", "F"]),

                ((d[(0, 0, 0)], d[(3, 2, 0)], d[(5, 0, 2)]), ["L'", "D'", "L", "F'", "D2", "F"]),
                ((d[(5, 0, 2)], d[(0, 0, 0)], d[(3, 2, 0)]), ["L'", "D2", "L2", "D'", "L'"]),
                ((d[(3, 2, 0)], d[(5, 0, 2)], d[(0, 0, 0)]), ["B", "D", "B'", "D'", "F'", "D", "F"]),

                ((d[(0, 0, 2)], d[(4, 0, 0)], d[(3, 2, 2)]), ["R", "D", "R'", "D", "F'", "D'", "F"]),
                ((d[(3, 2, 2)], d[(0, 0, 2)], d[(4, 0, 0)]), ["B'", "D2", "B", "F'", "D'", "F"]),
                ((d[(4, 0, 0)], d[(3, 2, 2)], d[(0, 0, 2)]), ["R", "D", "R'", "F'", "D", "F"]),

                ((d[(1, 0, 2)], d[(4, 2, 2)], d[(2, 2, 2)]), ["D'", "L", "D2", "L'", "D2", "F'", "D", "F"]),
                ((d[(2, 2, 2)], d[(1, 0, 2)], d[(4, 2, 2)]), ["D2", "F'", "D", "F"]),
                ((d[(4, 2, 2)], d[(2, 2, 2)], d[(1, 0, 2)]), ["L", "D'", "L'"]),
                
                ((d[(1, 0, 0)], d[(2, 2, 0)], d[(5, 2, 0)]), ["F'", "D2", "F", "D", "F'", "D'", "F"]),
                ((d[(5, 2, 0)], d[(1, 0, 0)], d[(2, 2, 0)]), ["D'", "F'", "D", "F"]),
                ((d[(2, 2, 0)], d[(5, 2, 0)], d[(1, 0, 0)]), ["D", "L", "D'", "L'"]),

                ((d[(1, 2, 0)], d[(5, 0, 0)], d[(3, 0, 0)]), ["D", "F'", "D2", "F", "D", "F'", "D'", "F"]),
                ((d[(3, 0, 0)], d[(1, 2, 0)], d[(5, 0, 0)]), ["F'", "D", "F"]),
                ((d[(5, 0, 0)], d[(3, 0, 0)], d[(1, 2, 0)]), ["D2", "L", "D'", "L'"]),

                ((d[(1, 2, 2)], d[(3, 0, 2)], d[(4, 0, 2)]), ["D2", "F'", "D2", "F", "D", "F'", "D'", "F"]),
                ((d[(4, 0, 2)], d[(1, 2, 2)], d[(3, 0, 2)]), ["F'", "D2", "F"]),
                ((d[(3, 0, 2)], d[(4, 0, 2)], d[(1, 2, 2)]), ["L", "D2", "L'"]),
            ]),

            ((d[(0, 0, 0)], d[(3, 2, 0)], d[(5, 0, 2)]), [
                ((d[(0, 0, 0)], d[(3, 2, 0)], d[(5, 0, 2)]), []),
                ((d[(5, 0, 2)], d[(0, 0, 0)], d[(3, 2, 0)]), ["L'", "D'", "L", "D", "L'", "D'", "L"]),
                ((d[(3, 2, 0)], d[(5, 0, 2)], d[(0, 0, 0)]), ["B", "D", "B'", "D2", "L'", "D", "L"]),

                ((d[(0, 0, 2)], d[(4, 0, 0)], d[(3, 2, 2)]), ["B'", "D'", "B", "L'", "D2", "L"]),
                ((d[(3, 2, 2)], d[(0, 0, 2)], d[(4, 0, 0)]), ["B'", "D'", "B2", "D2", "B'"]),
                ((d[(4, 0, 0)], d[(3, 2, 2)], d[(0, 0, 2)]), ["R", "D", "R'", "D'", "L'", "D", "L"]),

                ((d[(1, 0, 2)], d[(4, 2, 2)], d[(2, 2, 2)]), ["D2", "B", "D2", "B'", "D2", "L'", "D", "L"]),
                ((d[(2, 2, 2)], d[(1, 0, 2)], d[(4, 2, 2)]), ["L'","D2", "L"]),
                ((d[(4, 2, 2)], d[(2, 2, 2)], d[(1, 0, 2)]), ["B", "D2", "B'"]),
                
                ((d[(1, 0, 0)], d[(2, 2, 0)], d[(5, 2, 0)]), ["D'", "B", "D2", "B'", "D2", "L'", "D", "L"]),
                ((d[(5, 2, 0)], d[(1, 0, 0)], d[(2, 2, 0)]), ["D", "L'", "D2", "L"]),
                ((d[(2, 2, 0)], d[(5, 2, 0)], d[(1, 0, 0)]), ["B", "D'", "B'"]),

                ((d[(1, 2, 0)], d[(5, 0, 0)], d[(3, 0, 0)]), ["B", "D2", "B'", "D2", "L'", "D", "L"]),
                ((d[(3, 0, 0)], d[(1, 2, 0)], d[(5, 0, 0)]), ["D'", "L'", "D", "L"]),
                ((d[(5, 0, 0)], d[(3, 0, 0)], d[(1, 2, 0)]), ["D", "B", "D'", "B'"]),

                ((d[(1, 2, 2)], d[(3, 0, 2)], d[(4, 0, 2)]), ["D" ,"L'", "D2", "L", "D", "L'", "D'", "L"]),
                ((d[(4, 0, 2)], d[(1, 2, 2)], d[(3, 0, 2)]), ["L'", "D", "L"]),
                ((d[(3, 0, 2)], d[(4, 0, 2)], d[(1, 2, 2)]), ["D2", "B", "D'", "B'"]),
            ]),

            ((d[(0, 0, 2)], d[(4, 0, 0)], d[(3, 2, 2)]), [
                ((d[(0, 0, 2)], d[(4, 0, 0)], d[(3, 2, 2)]), []),
                ((d[(3, 2, 2)], d[(0, 0, 2)], d[(4, 0, 0)]), ["B'", "D'", "B", "D", "B'", "D'", "B"]),
                ((d[(4, 0, 0)], d[(3, 2, 2)], d[(0, 0, 2)]), ["R", "D", "R'", "D2", "B'", "D", "B"]),

                ((d[(1, 0, 2)], d[(4, 2, 2)], d[(2, 2, 2)]), ["D", "B'", "D2", "B", "D", "B'", "D'", "B"]),
                ((d[(2, 2, 2)], d[(1, 0, 2)], d[(4, 2, 2)]), ["B'","D", "B"]),
                ((d[(4, 2, 2)], d[(2, 2, 2)], d[(1, 0, 2)]), ["D", "B'", "D'", "B"]),
                
                ((d[(1, 0, 0)], d[(2, 2, 0)], d[(5, 2, 0)]), ["D2", "B'", "D2", "B", "D", "B'", "D'", "B"]),
                ((d[(5, 2, 0)], d[(1, 0, 0)], d[(2, 2, 0)]), ["B'", "D2", "B"]),
                ((d[(2, 2, 0)], d[(5, 2, 0)], d[(1, 0, 0)]), ["R", "D2", "R'"]),

                ((d[(1, 2, 0)], d[(5, 0, 0)], d[(3, 0, 0)]), ["D'", "R", "D2", "R'", "D", "B'", "D2", "B"]),
                ((d[(3, 0, 0)], d[(1, 2, 0)], d[(5, 0, 0)]), ["D", "B'", "D2", "B"]),
                ((d[(5, 0, 0)], d[(3, 0, 0)], d[(1, 2, 0)]), ["D", "R", "D2", "R'"]),

                ((d[(1, 2, 2)], d[(3, 0, 2)], d[(4, 0, 2)]), ["B'" ,"D2", "B", "D'", "R", "D2", "R'"]),
                ((d[(4, 0, 2)], d[(1, 2, 2)], d[(3, 0, 2)]), ["D'", "B'", "D", "B"]),
                ((d[(3, 0, 2)], d[(4, 0, 2)], d[(1, 2, 2)]), ["B'", "D'", "B"]),
            ]),

            ((d[(2, 1, 2)], d[(4, 2, 1)]), [
                ((d[(2, 1, 2)], d[(4, 2, 1)]), []),
                ((d[(4, 0, 1)], d[(3, 1, 2)]), move_tbl_exchange_edge_r_D_B_inv+["D'"]+move_tbl_exchange_edge_r_D_R),
                ((d[(3, 1, 0)], d[(5, 0, 1)]), move_tbl_exchange_edge_r_D_L_inv+["D2"]+move_tbl_exchange_edge_r_D_R),
                ((d[(5, 2, 1)], d[(2, 1, 0)]), move_tbl_exchange_edge_r_D_F_inv+["D"]+move_tbl_exchange_edge_r_D_R),

                ((d[(4, 2, 1)], d[(2, 1, 2)]), move_tbl_exchange_edge_l_D_F_inv+["D"]+move_tbl_exchange_edge_r_D_R),
                ((d[(3, 1, 2)], d[(4, 0, 1)]), move_tbl_exchange_edge_l_D_R_inv+move_tbl_exchange_edge_r_D_R),
                ((d[(5, 0, 1)], d[(3, 1, 0)]), move_tbl_exchange_edge_l_D_B_inv+["D'"]+move_tbl_exchange_edge_r_D_R),
                ((d[(2, 1, 0)], d[(5, 2, 1)]), move_tbl_exchange_edge_r_D_F_inv+move_tbl_exchange_edge_l_D_F),
                
                ((d[(1, 0, 1)], d[(2, 2, 1)]), ["D"]+move_tbl_exchange_edge_r_D_R),
                ((d[(1, 1, 2)], d[(4, 1, 2)]), move_tbl_exchange_edge_r_D_R),
                ((d[(1, 2, 1)], d[(3, 0, 1)]), ["D'"]+move_tbl_exchange_edge_r_D_R),
                ((d[(1, 1, 0)], d[(5, 1, 0)]), ["D2"]+move_tbl_exchange_edge_r_D_R),

                ((d[(2, 2, 1)], d[(1, 0, 1)]), move_tbl_exchange_edge_l_D_F),
                ((d[(4, 1, 2)], d[(1, 1, 2)]), ["D'"]+move_tbl_exchange_edge_l_D_F),
                ((d[(3, 0, 1)], d[(1, 2, 1)]), ["D2"]+move_tbl_exchange_edge_l_D_F),
                ((d[(5, 1, 0)], d[(1, 1, 0)]), ["D"]+move_tbl_exchange_edge_l_D_F),
            ]),

            ((d[(4, 0, 1)], d[(3, 1, 2)]), [
                ((d[(4, 0, 1)], d[(3, 1, 2)]), []),
                ((d[(3, 1, 0)], d[(5, 0, 1)]), move_tbl_exchange_edge_r_D_L_inv+["D'"]+move_tbl_exchange_edge_r_D_B),
                ((d[(5, 2, 1)], d[(2, 1, 0)]), move_tbl_exchange_edge_r_D_F_inv+["D2"]+move_tbl_exchange_edge_r_D_B),

                ((d[(3, 1, 2)], d[(4, 0, 1)]), move_tbl_exchange_edge_l_D_R_inv+["D"]+move_tbl_exchange_edge_r_D_B),
                ((d[(5, 0, 1)], d[(3, 1, 0)]), move_tbl_exchange_edge_l_D_B_inv+move_tbl_exchange_edge_r_D_B),
                ((d[(2, 1, 0)], d[(5, 2, 1)]), move_tbl_exchange_edge_l_D_L_inv+["D'"]+move_tbl_exchange_edge_r_D_B),
                
                ((d[(1, 0, 1)], d[(2, 2, 1)]), ["D2"]+move_tbl_exchange_edge_r_D_B),
                ((d[(1, 1, 2)], d[(4, 1, 2)]), ["D"]+move_tbl_exchange_edge_r_D_B),
                ((d[(1, 2, 1)], d[(3, 0, 1)]), move_tbl_exchange_edge_r_D_B),
                ((d[(1, 1, 0)], d[(5, 1, 0)]), ["D'"]+move_tbl_exchange_edge_r_D_B),

                ((d[(2, 2, 1)], d[(1, 0, 1)]), ["D"]+move_tbl_exchange_edge_l_D_R),
                ((d[(4, 1, 2)], d[(1, 1, 2)]), move_tbl_exchange_edge_l_D_R),
                ((d[(3, 0, 1)], d[(1, 2, 1)]), ["D'"]+move_tbl_exchange_edge_l_D_R),
                ((d[(5, 1, 0)], d[(1, 1, 0)]), ["D2"]+move_tbl_exchange_edge_l_D_R),
            ]),

            ((d[(3, 1, 0)], d[(5, 0, 1)]), [
                ((d[(3, 1, 0)], d[(5, 0, 1)]), []),
                ((d[(5, 2, 1)], d[(2, 1, 0)]), move_tbl_exchange_edge_r_D_F_inv+["D'"]+move_tbl_exchange_edge_r_D_L),

                ((d[(5, 0, 1)], d[(3, 1, 0)]), move_tbl_exchange_edge_l_D_B_inv+["D"]+move_tbl_exchange_edge_r_D_L),
                ((d[(2, 1, 0)], d[(5, 2, 1)]), move_tbl_exchange_edge_l_D_L_inv+move_tbl_exchange_edge_r_D_L),
                
                ((d[(1, 0, 1)], d[(2, 2, 1)]), ["D'"]+move_tbl_exchange_edge_r_D_L),
                ((d[(1, 1, 2)], d[(4, 1, 2)]), ["D2"]+move_tbl_exchange_edge_r_D_L),
                ((d[(1, 2, 1)], d[(3, 0, 1)]), ["D"]+move_tbl_exchange_edge_r_D_L),
                ((d[(1, 1, 0)], d[(5, 1, 0)]), move_tbl_exchange_edge_r_D_L),

                ((d[(2, 2, 1)], d[(1, 0, 1)]), ["D2"]+move_tbl_exchange_edge_l_D_B),
                ((d[(4, 1, 2)], d[(1, 1, 2)]), ["D"]+move_tbl_exchange_edge_l_D_B),
                ((d[(3, 0, 1)], d[(1, 2, 1)]), move_tbl_exchange_edge_l_D_B),
                ((d[(5, 1, 0)], d[(1, 1, 0)]), ["D'"]+move_tbl_exchange_edge_l_D_B),
            ]),

            ((d[(5, 2, 1)], d[(2, 1, 0)]), [
                ((d[(5, 2, 1)], d[(2, 1, 0)]), []),

                ((d[(2, 1, 0)], d[(5, 2, 1)]), move_tbl_exchange_edge_l_D_L_inv+["D"]+move_tbl_exchange_edge_r_D_F),
                
                ((d[(1, 0, 1)], d[(2, 2, 1)]), move_tbl_exchange_edge_r_D_F),
                ((d[(1, 1, 2)], d[(4, 1, 2)]), ["D'"]+move_tbl_exchange_edge_r_D_F),
                ((d[(1, 2, 1)], d[(3, 0, 1)]), ["D2"]+move_tbl_exchange_edge_r_D_F),
                ((d[(1, 1, 0)], d[(5, 1, 0)]), ["D"]+move_tbl_exchange_edge_r_D_F),

                ((d[(2, 2, 1)], d[(1, 0, 1)]), ["D'"]+move_tbl_exchange_edge_l_D_L),
                ((d[(4, 1, 2)], d[(1, 1, 2)]), ["D2"]+move_tbl_exchange_edge_l_D_L),
                ((d[(3, 0, 1)], d[(1, 2, 1)]), ["D"]+move_tbl_exchange_edge_l_D_L),
                ((d[(5, 1, 0)], d[(1, 1, 0)]), move_tbl_exchange_edge_l_D_L),
            ]),

            ((d[(1, 0, 1)], d[(1, 1, 2)], d[(1, 2, 1)], d[(1, 1, 0)]), [
                ((d[(1, 0, 1)], d[(2, 2, 1)]), move_tbl_exchange_edge_r_D_F),
                ((d[(1, 1, 2)], d[(4, 1, 2)]), ["D'"]+move_tbl_exchange_edge_r_D_F),
                ((d[(1, 2, 1)], d[(3, 0, 1)]), ["D2"]+move_tbl_exchange_edge_r_D_F),
                ((d[(1, 1, 0)], d[(5, 1, 0)]), ["D"]+move_tbl_exchange_edge_r_D_F),

                ((d[(2, 2, 1)], d[(1, 0, 1)]), ["D'"]+move_tbl_exchange_edge_l_D_L),
                ((d[(4, 1, 2)], d[(1, 1, 2)]), ["D2"]+move_tbl_exchange_edge_l_D_L),
                ((d[(3, 0, 1)], d[(1, 2, 1)]), ["D"]+move_tbl_exchange_edge_l_D_L),
                ((d[(5, 1, 0)], d[(1, 1, 0)]), move_tbl_exchange_edge_l_D_L),
            ]),
        ]

        # TODO: add the asserts and solving algo for the 3x3x3 cube!
        f = self.fields
        cells_lst = []
        for cells, moving_lst in moving_table_solve_3x3:
            for pos_col, moves_lst in moving_lst:
                is_pos_col = True
                for pos, cell in zip(pos_col, cells):
                    if f[pos]!=cell[0]:
                        is_pos_col = False
                        break

                if is_pos_col:
                    print("cells: {}, pos_col: {}, moves_lst: {}".format(cells, pos_col, moves_lst))
                    self.apply_move_lst(moves_lst)
                    break

            try:
                for cell in cells:
                    # print("assertion for: f[cell]: {}, cell: {}".format(f[cell], cell))
                    assert f[cell]==cell[0]
            except Exception as e:
                print(e)

                print("for cell: {}, cells: {}".format(cell, cells))
                self.print_field()
                sys.exit(-1)


            # for cells in cells_lst:
            #     for cell in cells:
            #         assert f[cell]==cell[0]
            for cell in cells_lst:
                assert f[cell]==cell[0]
            
            cells_lst.extend(cells)

        print("finished solving:")
        self.print_field()
        print("")


    def count_finished_pairs(self):
        f = self.fields
        s = 0

        s += (0+np.all(f[0, 0, 1]==f[0, 0, 1:-1])+np.all(f[3, -1, 1]==f[3, -1, 1:-1]))==2
        s += (0+np.all(f[0, 1, 0]==f[0, 1:-1, 0])+np.all(f[5, 1, -1]==f[5, 1:-1, -1]))==2
        s += (0+np.all(f[0, -1, 1]==f[0, -1, 1:-1])+np.all(f[2, 0, 1]==f[2, 0, 1:-1]))==2
        s += (0+np.all(f[0, 1, -1]==f[0, 1:-1, -1])+np.all(f[4, 1, 0]==f[4, 1:-1, 0]))==2
        
        s += (0+np.all(f[1, 0, 1]==f[1, 0, 1:-1])+np.all(f[2, -1, 1]==f[2, -1, 1:-1]))==2
        s += (0+np.all(f[1, 1, 0]==f[1, 1:-1, 0])+np.all(f[5, 1, 0]==f[5, 1:-1, 0]))==2
        s += (0+np.all(f[1, -1, 1]==f[1, -1, 1:-1])+np.all(f[3, 0, 1]==f[3, 0, 1:-1]))==2
        s += (0+np.all(f[1, 1, -1]==f[1, 1:-1, -1])+np.all(f[4, 1, -1]==f[4, 1:-1, -1]))==2
        
        s += (0+np.all(f[2, 1, -1]==f[2, 1:-1, -1])+np.all(f[4, -1, 1]==f[4, -1, 1:-1]))==2
        s += (0+np.all(f[3, 1, -1]==f[3, 1:-1, -1])+np.all(f[4, 0, 1]==f[4, 0, 1:-1]))==2
        s += (0+np.all(f[3, 1, 0]==f[3, 1:-1, 0])+np.all(f[5, 0, 1]==f[5, 0, 1:-1]))==2
        s += (0+np.all(f[2, 1, 0]==f[2, 1:-1, 0])+np.all(f[5, -1, 1]==f[5, -1, 1:-1]))==2

        return s


if __name__ == "__main__":
    rc = RubiksCube(4)
    print("rc: {}".format(rc))

    print("Test some rotations!")
    orig_fields = rc.fields.copy()

    # moves_lst = [
    #     ["U", "R", "U'", "R'"]*6,
    #     ["R", "U'", "R", "U", "R", "U", "R", "U'", "R'", "U'", "R2"]*3,
    #     ["Rw", "Uw'", "Rw", "Uw", "Rw", "Uw", "Rw", "Uw'", "Rw'", "Uw'", "Rw2"],
    # ]

    moves = ["U", "F", "R", "L", "D", "B"]+['r', 'l', 'd', 'b', 'f']+["Uw'", "Dw'", "Fw2", "Lw2", "Bw'", "Rw"]
    moves_lst = moves+[m if "2" in m else m+"'" if not "'" in m else m.replace("'", "") for m in moves[::-1]]

    # for moves in moves_lst:
    #     print("moves: {}".format(moves))
    #     for m in moves:
    #         rc.moving_table[m]()

    #     assert np.all(orig_fields==rc.fields)

    # moves_lst = ["D", "f", "r", "U", "D", "f2"]
    # rc.apply_move_lst(moves_lst)
    # print("moves_lst: {}".format(moves_lst))
    # rc.print_field()

    while True:
    # for i in range(0, 100):
    #     print("i: {}".format(i))
        rc.mix_cube()
        rc.solve_cube()
        # break
    rc.print_field()

    # # for moves in moves_lst:
    #     # print("moves: {}".format(moves))
    # for m in moves_lst:
    #     rc.moving_table[m]()

    # # assert np.all(orig_fields==rc.fields)
