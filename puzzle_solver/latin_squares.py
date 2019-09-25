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
            "U2": {"U": "U", "D": "D", "F": "B", "R": "L", "B": "F", "L": "R"},
            "F":  {"U": "R", "D": "L", "F": "F", "R": "D", "B": "B", "L": "U"},
            "F'": {"U": "L", "D": "R", "F": "F", "R": "U", "B": "B", "L": "D"},
            "F2": {"U": "D", "D": "U", "F": "F", "R": "L", "B": "B", "L": "R"},
            "R":  {"U": "B", "D": "F", "F": "U", "R": "R", "B": "D", "L": "L"},
            "R'": {"U": "F", "D": "B", "F": "D", "R": "R", "B": "U", "L": "L"},
            "R2": {"U": "D", "D": "U", "F": "B", "R": "R", "B": "F", "L": "L"},
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

        mt = self.moving_table
        mt["x"] = lambda: mt["Uw"]()+mt["Dw'"]()
        mt["x'"] = lambda: mt["Uw'"]()+mt["Dw"]()
        mt["x2"] = lambda: mt["Uw2"]()+mt["Dw2"]()
        mt["y"] = lambda: mt["Fw"]()+mt["Bw'"]()
        mt["y'"] = lambda: mt["Fw'"]()+mt["Bw"]()
        mt["y2"] = lambda: mt["Fw2"]()+mt["Bw2"]()
        mt["z"] = lambda: mt["Rw"]()+mt["Lw'"]()
        mt["z'"] = lambda: mt["Rw'"]()+mt["Lw"]()
        mt["z2"] = lambda: mt["Rw2"]()+mt["Lw2"]()

        self.possible_moves = np.array([k for k in self.moving_table])


    def _reset_fields(self):
        self.fields = np.array([[[i]*self.n]*self.n for i in range(0, 6)])


    def __rotate_field_counter_clockwise_4x4(self, field):
        field[self.tpl] = field[self.tpl_new_clockwise]


    def __rotate_field_clockwise_4x4(self, field):
        field[self.tpl] = field[self.tpl_new_counter_clockwise]


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
        return "u{}".format(layer)


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
        return "f{}".format(layer)


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
        return "r{}".format(layer)


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
        return "ui{}".format(layer)


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
        return "fi{}".format(layer)


if __name__ == "__main__":
    rc = RubiksCube(4)

    with open(PATH_ROOT_DIR+"../sat_problems/found_valid_4x4_no_neightbor.txt", "r") as f:
        lines = f.readlines()
    lines = list(map(lambda x: list(map(int, x.replace("\n", ""))), lines))
    arr = np.array(lines).reshape((-1, 6, 4, 4))

    solvable_fields = []
    not_solvable_fields = []

    for f_orig in arr[:10]:
        fields_str = "".join(map(str, f_orig.reshape((-1, )).tolist()))
        rc.fields = f_orig.copy()

        is_solvable = False
        try:
            rc.solve_cube()
            is_solvable = True
        except:
            pass
            print("Cannot solve for field: '{}'".format(fields_str))
            not_solvable_fields.append(fields_str)

        if is_solvable:
            print("Solvable cube: '{}'".format(fields_str))
            solvable_fields.append(fields_str)

    print("len(solvable_fields): {}".format(len(solvable_fields)))
    print("len(not_solvable_fields): {}".format(len(not_solvable_fields)))

    def set_one_solvable_field():
        rc.fields = np.array(list(map(int, solvable_fields[0]))).reshape((6, 4, 4))
        rc.print_field()

    set_one_solvable_field()
