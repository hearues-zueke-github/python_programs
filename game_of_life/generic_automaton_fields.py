#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import gzip
import os
import sys

# import tempfile
from memory_tempfile import MemoryTempfile
tempfile = MemoryTempfile()

from collections import defaultdict
from copy import deepcopy
from dotmap import DotMap
from operator import itemgetter

from pprint import pprint

from os.path import expanduser

import multiprocessing as mp

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = tempfile.gettempdir()+"/"

from PIL import Image

import numpy as np

sys.path.append("../")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj
import global_object_getter_setter


def range_gen(g, n):
    i = 0
    while (v := next(g)) != None:
        yield v
        i += 1
        if i == n:
            break


class BitAutomaton(Exception):
    __slot__ = [
        'h', 'w',
        'frame', 'frame_wrap',
        'field_size', 'field',
        'field_frame_size', 'field_frame',
        'd_vars', 'd_func', 'funcs_str',
    ]

    def __init__(self, h, w, frame, frame_wrap, funcs_str):
        self.h = h
        self.w = w
        
        self.frame = frame
        self.frame_wrap = frame_wrap
        
        self.field_size = (h, w)
        self.field = np.zeros(self.field_size, dtype=np.uint8)

        self.field_frame_size = (h+frame*2, w+frame*2)
        self.field_frame = np.zeros(self.field_frame_size, dtype=np.uint8)

        self.d_vars = {}
        self.d_vars['n'] = self.field_frame[frame:-frame, frame:-frame]
        l_up = [('u', i, -i) for i in range(frame, 0, -1)]
        l_down = [('d', i, i) for i in range(1, frame+1)]
        l_left = [('l', i, -i) for i in range(frame, 0, -1)]
        l_right = [('r', i, i) for i in range(1, frame+1)]
        l_empty = [('', 0, 0)]

        for direction, amount, i in l_up+l_down:
            self.d_vars[direction+str(amount)] = self.field_frame[frame+i:frame+h+i, frame:frame+w]
            self.d_vars[direction*amount] = self.field_frame[frame+i:frame+h+i, frame:frame+w]
        for direction, amount, i in l_left+l_right:
            self.d_vars[direction+str(amount)] = self.field_frame[frame:frame+h, frame+i:frame+i+w]
            self.d_vars[direction*amount] = self.field_frame[frame:frame+h, frame+i:frame+i+w]

        for direction_y, amount_y, i_y in l_up+l_empty+l_down:
            for direction_x, amount_x, i_x in l_left+l_empty+l_right:
                self.d_vars[direction_y+str(amount_y)+direction_x+str(amount_x)] = self.field_frame[frame+i_y:frame+i_y+h, frame+i_x:frame+i_x+w]
                self.d_vars[direction_y*amount_y+direction_x*amount_x] = self.field_frame[frame+i_y:frame+i_y+h, frame+i_x:frame+i_x+w]

        self.d_func = {}

        self.funcs_str = funcs_str

        exec(funcs_str, self.d_vars, self.d_func)

        assert 'rng' in self.d_func.keys()

        # test if each function starting with 'fun_' is returning a boolean array!
        l_func_name = []
        for func_name in self.d_func.keys():
            if func_name[:4] != 'fun_':
                continue

            l_func_name.append(func_name)

            # print("func_name: {}".format(func_name))
            v = self.d_func[func_name]()
            assert v.dtype == np.bool
            assert v.shape == self.field_size

        # check, if every function name is appearing starting from 0 to upwards in ascending order!
        assert np.all(np.diff(np.sort([int(v.replace('fun_', '')) for v in l_func_name])) == 1)


    def fill_field_frame(self):
        self.field_frame[self.frame:-self.frame, self.frame:-self.frame] = self.field

        if self.frame_wrap == False:
            self.field_frame[:, :self.frame] = 0
            self.field_frame[:, -self.frame:] = 0
            self.field_frame[:self.frame, self.frame:-self.frame] = 0
            self.field_frame[-self.frame:, self.frame:-self.frame] = 0
        else:
            # do the right part copy to left
            self.field_frame[self.frame:-self.frame, :self.frame] = self.field_frame[self.frame:-self.frame, -self.frame*2:-self.frame]
            # do the left part copy to right
            self.field_frame[self.frame:-self.frame, -self.frame:] = self.field_frame[self.frame:-self.frame, self.frame:self.frame*2]
            # do the bottom part copy to top
            self.field_frame[:self.frame] = self.field_frame[-self.frame*2:-self.frame]
            # do the top part copy to bottom
            self.field_frame[-self.frame:] = self.field_frame[self.frame:self.frame*2]


if __name__ == '__main__':
#     func_str = """
# def f():
#     a = 4 + b
#     return a
# """

#     d_loc = {}
#     d_glob = {'b': 5}

#     exec(func_str, d_glob, d_loc)

#     b = d_loc['f']()
#     print("b: {}".format(b))
    
#     d_glob = {'b': 4}
#     b = d_loc['f']()
#     print("b: {}".format(b))

#     sys.exit()


    h = 3
    w = 4

    frame = 2
    # frame_wrap = False
    frame_wrap = True
    
    funcs_str = """
def rng(seed=0):
    a = seed
    while True:
        a = ((a + 19) ^ 0x2343) % 15232
        yield a % 13
fun_0 = lambda: (u + dl) % 2 == 0
def fun_1():
    return (u + dl) % 3 == 0
def fun_2():
    return ((u + dl + urr) % 2 == 0) ^ ((dr + dll + uurr) % 2 == 0)
def fun_3():
    a = u+d+l+r+ur+ul+dr+dl
    return (a==2)|(a==3)
"""
    
    bit_automaton = BitAutomaton(h=h, w=w, frame=frame, frame_wrap=frame_wrap, funcs_str=funcs_str)

    bit_automaton.field

    # field_size = (h, w)
    # field = np.zeros(field_size, dtype=np.uint8)
    # # field[:] = np.arange(0, np.multiply(*field_size)).reshape(field_size)
    # field[:] = np.random.randint(0, 2, field_size)

    # field_frame_size = (h+frame*2, w+frame*2)
    # field_frame = np.zeros(field_frame_size, dtype=np.uint8)
    
    # def fill_field_frame(field_frame, field_size, field, frame, frame_wrap):
    #     field_frame[:] = np.random.randint(0, 10, field_frame_size)
    #     field_frame[frame:-frame, frame:-frame] = field

    #     if frame_wrap == False:
    #         field_frame[:, :frame] = 0
    #         field_frame[:, -frame:] = 0
    #         field_frame[:frame, frame:-frame] = 0
    #         field_frame[-frame:, frame:-frame] = 0
    #     else:
    #         # do the right part copy to left
    #         field_frame[frame:-frame, :frame] = field_frame[frame:-frame, -frame*2:-frame]
    #         # do the left part copy to right
    #         field_frame[frame:-frame, -frame:] = field_frame[frame:-frame, frame:frame*2]
    #         # do the bottom part copy to top
    #         field_frame[:frame] = field_frame[-frame*2:-frame]
    #         # do the top part copy to bottom
    #         field_frame[-frame:] = field_frame[frame:frame*2]

    # def fill():
    #     fill_field_frame(field_frame, field_size, field, frame, frame_wrap)

    # fill()

    # d_vars = {}
    # d_vars['n'] = field_frame[frame:-frame, frame:-frame]
    # l_up = [('u', i, -i) for i in range(frame, 0, -1)]
    # l_down = [('d', i, i) for i in range(1, frame+1)]
    # l_left = [('l', i, -i) for i in range(frame, 0, -1)]
    # l_right = [('r', i, i) for i in range(1, frame+1)]
    # l_empty = [('', 0, 0)]

    # for direction, amount, i in l_up+l_down:
    #     d_vars[direction+str(amount)] = field_frame[frame+i:frame+h+i, frame:frame+w]
    #     d_vars[direction*amount] = field_frame[frame+i:frame+h+i, frame:frame+w]
    # for direction, amount, i in l_left+l_right:
    #     d_vars[direction+str(amount)] = field_frame[frame:frame+h, frame+i:frame+i+w]
    #     d_vars[direction*amount] = field_frame[frame:frame+h, frame+i:frame+i+w]

    # for direction_y, amount_y, i_y in l_up+l_empty+l_down:
    #     for direction_x, amount_x, i_x in l_left+l_empty+l_right:
    #         d_vars[direction_y+str(amount_y)+direction_x+str(amount_x)] = field_frame[frame+i_y:frame+i_y+h, frame+i_x:frame+i_x+w]
    #         d_vars[direction_y*amount_y+direction_x*amount_x] = field_frame[frame+i_y:frame+i_y+h, frame+i_x:frame+i_x+w]

    # d_func = {}

    # exec(funcs_str, d_vars, d_func)

    # assert 'rng' in d_func.keys()

    # # test if each function starting with 'fun_' is returning a boolean array!
    # l_func_name = []
    # for func_name in d_func.keys():
    #     if func_name[:4] != 'fun_':
    #         continue

    #     l_func_name.append(func_name)

    #     print("func_name: {}".format(func_name))
    #     v = d_func[func_name]()
    #     assert v.dtype == np.bool
    #     assert v.shape == field_size

    # # check, if every function name is appearing starting from 0 to upwards in ascending order!
    # assert np.all(np.diff(np.sort([int(v.replace('fun_', '')) for v in l_func_name])) == 1)

    # # g = d_func['rng']()
