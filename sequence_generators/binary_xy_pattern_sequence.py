#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import functools
import sys
import time

import numpy as np

from indexed import IndexedOrderedDict

from PIL import Image, ImageTk

import multiprocessing as mp
from multiprocessing import Process, Pipe, Lock

from dotmap import DotMap

import utils_sequence

class BinaryFieldCell(Exception):
    def __init__(self, y, x, num, mod, p_childs):
        self.y = y
        self.x = x
        self.p = (y, x)
        self.num = num
        self.mod = mod

        self.p_childs = p_childs


    def add_one(self):
        self.num = (self.num+1) % self.mod
        return self.num


class BinaryField(Exception):
    def __init__(self, mod):
        self.cell_start = (0, 0)
        self.mod = mod
        self.field_cells = {}
        
        self.sequence = []
        self.xs = []
        self.ys = []
        self.idx = 1
        # self.field_cells = {self.cell_start: BinaryFieldCell(0, 0, 0, mod, [(0, -1), (0, 1)])}


    def add_one_to_field(self):
        self.p_reached = []
        self._add_one(self.cell_start)
        # print("self.field_cells.keys(): {}".format(self.field_cells.keys()))
        self.print_field_2d()


    def _add_one(self, p):
        if p in self.p_reached:
            return
        self.p_reached.append(p)

        if not p in self.field_cells:
            y = p[0]
            x = p[1]
            # p_childs = [(y, x+1), (y+1, x+1)]
            # p_childs = [(y, x+1), (y+1, x)] #, (y+1, x+1)]
            # p_childs = [(y, x+1), (y+1, x), (y+1, x+1)]
            p_childs = [(y, x+1), (y+1, x), (y-1, x), (y, x-1)]
            # t = (x+y) % 2
            # if t == 0:
            #     p_childs = [(y, x-1), (y, x+1)]
            # else:
            #     p_childs = [(y-1, x), (y+1, x)]
            self.field_cells[p] = BinaryFieldCell(y, x, 0, self.mod, p_childs)

        cell = self.field_cells[p]
        v = cell.add_one()
        if v == 0:
            p_childs = cell.p_childs
            for p_child in p_childs:
                self._add_one(p_child)

        # print("v: {}".format(v))


    def print_field_2d(self):
        def get_field(p):
            if p in get_field.field or not p in get_field.field_cells:
                return
            for p_child in get_field.field_cells[p].p_childs:
                get_field.field[p] = get_field.field_cells[p].num
                get_field(p_child)

        cell_start = self.cell_start
        field_cells = self.field_cells
        get_field.field = {}
        get_field.field_cells = field_cells
        get_field(cell_start)

        # print("get_field.field:\n{}".format(get_field.field))

        field = get_field.field
        lst = list(zip(*list(field.items())))
        ps = lst[0]
        ys, xs = list(map(np.array, list(zip(*ps))))
        vs = np.array(lst[1])
        # print("")
        # print("ys: {}".format(ys))
        # print("xs: {}".format(xs))
        # print("vs: {}".format(vs))

        get_min_max = lambda arr: (np.min(arr), np.max(arr))
        y_min, y_max = get_min_max(ys)
        x_min, x_max = get_min_max(xs)
        # print("y_min: {}, y_max: {}".format(y_min, y_max))
        # print("x_min: {}, x_max: {}".format(x_min, x_max))

        height = y_max - y_min + 1
        width = x_max - x_min + 1

        arr_field = np.zeros((height, width), dtype=np.int)
        ys -= y_min
        xs -= x_min

        for y, x, v in zip(ys, xs, vs):
            arr_field[y, x] = v

        self.arr_field = arr_field
        self.sequence.append((self.idx, np.sum(arr_field)))
        
        self.xs.append(self.idx)
        self.ys.append(np.sum(arr_field))
        
        self.idx += 1

        print("arr_field:\n{}".format(arr_field))

        # v_min, v_max = get_min_max(vs)

if __name__ == "__main__":
    binary_field = BinaryField(2)

    for i in range(1, 63):
        print("\ni: {}".format(i))
        binary_field.add_one_to_field()

    # print("binary_field.sequence:\n{}".format(binary_field.sequence))
    print("binary_field.xs: {}".format(binary_field.xs))
    print("binary_field.ys: {}".format(binary_field.ys))
    # for s in binary_field.sequence:
    #     print("s: {}".format(s))
