#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.8.6 -i

# -*- coding: utf-8 -*-

import dill
import gzip
import os
import string
import sys
import inspect
import textwrap

from typing import List, Dict, Set, Mapping, Any, Tuple

# import tempfile
from memory_tempfile import MemoryTempfile
tempfile = MemoryTempfile()

from collections import defaultdict
from copy import deepcopy
from dotmap import DotMap
from operator import itemgetter

from pprint import pprint

from os.path import expanduser

import itertools

import multiprocessing as mp

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = tempfile.gettempdir()+"/"

from PIL import Image

import numpy as np

sys.path.append("../")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj
import global_object_getter_setter
import utils
from utils_function import copy_function

from bit_automaton import BitAutomaton

def range_gen(g, n):
    i = 0
    while (v := next(g)) != None:
        yield v
        i += 1
        if i == n:
            break

def prepare_functions(funcs_str, frame):
    bit_automaton = BitAutomaton(h=10, w=8, frame=frame, frame_wrap=True, l_func=[], func_inv=None, func_rng=None)
    
    d_vars = bit_automaton.d_vars
    field_size = bit_automaton.field_size
    d_func = {}
    exec(funcs_str, d_vars, d_func)
    print("d_func.keys(): {}".format(d_func.keys()))

    d_func_keys = d_func.keys()
    assert 'rng' in d_func_keys
    assert 'inv' in d_func_keys
    assert 'l_func' in d_func_keys

    func_inv = copy_function(d_func['inv'])

    d_vars['inv'] = func_inv

    # test if each function starting with 'fun_' is returning a boolean array!
    l_func_name = []
    for func_name in d_func.keys():
        if func_name[:4] != 'fun_':
            continue

        l_func_name.append(func_name)

        # print("func_name: {}".format(func_name))
        v = d_func[func_name]()
        assert v.dtype == np.bool
        assert v.shape == field_size

    # check, if every function name is appearing starting from 0 to upwards in ascending order!
    assert np.all(np.diff(np.sort([int(v.replace('fun_', '')) for v in l_func_name])) == 1)

    l_func = [d_func[func_name] for func_name in l_func_name]
    # s_func_nr = set(range(0, len(l_func_name)))
    func_inv = d_func['inv']
    func_rng = d_func['rng']

    return l_func, func_inv, func_rng


if __name__ == '__main__':
    frame = 2
    frame_wrap = True
    
    bit_automaton = BitAutomaton(h=10, w=8, frame=frame, frame_wrap=True, l_func=[], func_inv=None, func_rng=None)

    funcs_str_inv = "def inv(x):\n return ~x\n"
    funcs_str_rng = """
def rng(seed=0):
    a = seed
    while True:
        a = ((a + 19) ^ 0x2343) % 15232
        yield a % 13

"""

    l_digits = string.digits
    l_var = [v for v in bit_automaton.d_vars.keys() if not any([c in l_digits for c in v])]
    print("l_var: {}".format(l_var))

    def get_random_function_str_body_dm(func_nr : int, l_var : List[str], n_and : int, n_or : int) -> str:
        l_vars_inv = [f'inv({v})' for v in l_var]

        arr_var = np.vstack((l_var, l_vars_inv))
        arr_var_i = np.arange(0, arr_var.shape[1])
        arr_var_row = np.arange(0, 2)

        func_name = f'fun_{func_nr}'
        # s = f'def {func_name}():\n'
        s = ''
        l_new_var = []
        l_l_chosen_var = []
        l_formel_and = []
        for i_or in range(0, n_or):
            new_var = f'a_{i_or}'
            l_new_var.append(new_var)

            arr_chosen_var = np.unique(np.random.choice(arr_var_i, (n_and, )))
            arr_inv_var = np.random.choice(arr_var_row, (arr_chosen_var.shape[0], ))

            formel_and = ' & '.join(np.sort(arr_var[arr_inv_var, arr_chosen_var]))
            l_formel_and.append(formel_and)

        l_former_and_sorted = sorted(set(l_formel_and))
        for new_var, formel_and in zip(l_new_var, l_formel_and):
            s += f' {new_var} = {formel_and}\n'
            l_l_chosen_var.append((arr_chosen_var, arr_inv_var))

        last_formel = ' | '.join(l_new_var)
        s += f' return {last_formel}\n'
        return DotMap(locals(), dynamic_=None)

    n_and = 5
    n_or = 10
    n_func = 12

    l_dm_local = [get_random_function_str_body_dm(func_nr=i, l_var=l_var, n_and=n_and, n_or=n_or) for i in range(0, n_func)]
    l_func_str_body = [dm.s for dm in l_dm_local]

    l_func_str_body_sorted  = sorted(set(l_func_str_body))
    l_func_name = [dm.func_name for dm in l_dm_local[:len(l_func_str_body_sorted)]]
    l_func_str_sorted = [f'def {func_name}():\n{s}' for func_name, s in zip(l_func_name, l_func_str_body_sorted)]

    func_str = ', '.join([func_name for func_name in l_func_name])
    funcs_str_func = '\n'.join(l_func_str_sorted)
    
    func_list_str = 'l_func = [{}]\n'.format(func_str)

    funcs_str = '\n'.join([
        inspect.cleandoc(funcs_str_rng)+'\n',
        funcs_str_inv,
        funcs_str_func,
        func_list_str,
    ])

    # sys.exit()

    # TODO: create some random functions too!

    l_func, func_inv, func_rng = prepare_functions(funcs_str=funcs_str, frame=frame)


    # path_images = PATH_ROOT_DIR + 'images/'

    # file_name = 'nature_trees_river_84761_800x600.jpg'
    # image_path = path_images + file_name

    # img = Image.open(image_path)
    # pix = np.array(img)

    pix = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)

    h, w = pix.shape[:2]

    path_images_save = os.path.join(TEMP_DIR, 'save_images/test_other')
    # path_images_save = os.path.join(TEMP_DIR, 'save_images/{}/'.format(file_name.split('.')[0]))
    utils.mkdirs(path_images_save)

    file_path_funcs = os.path.join(path_images_save, 'test_functions_python.py')
    with open(file_path_funcs, 'w') as f:
        f.write(funcs_str)

    arr_bits = np.array([[list(itertools.chain(*[list(map(int, bin(b)[2:].zfill(8))) for b in v])) for v in row] for row in pix], dtype=np.uint8).transpose(2, 0, 1)

    amount_bit_automaton = arr_bits.shape[0]

    l_bit_automaton = [BitAutomaton(h=h, w=w, frame=frame, frame_wrap=frame_wrap, l_func=l_func, func_inv=func_inv, func_rng=func_rng) for _ in range(0, amount_bit_automaton)]

    for bit_automaton, bits in zip(l_bit_automaton, arr_bits):
        bit_automaton.set_field(bits.astype(np.bool))

    def convert_bit_field_to_pix(l_bit_automaton):
        assert len(l_bit_automaton) == 24 # 24 bits

        arr = np.array([np.sum([bit_automaton<<j for j, bit_automaton in zip(range(7, -1, -1), l_bit_automaton[8*i:8*(i+1)])], axis=0) for i in range(0, 3)], dtype=np.uint8)
        return arr.transpose(1, 2, 0)

    pix2 = convert_bit_field_to_pix(l_bit_automaton)
    assert np.all(pix2 == pix)

    print("i: {}".format(0))
    file_path = os.path.join(path_images_save, '{:04}.png'.format(0))
    Image.fromarray(pix2).save(file_path)

    iterations_amount = 200
    l_func_nr = list(range(0, len(l_func)))
    amount_function_mod = len(l_func_nr)
    rng = func_rng(seed=0)
    for i in range(1, iterations_amount):
    # for i in range(1, 100):
        func_nr = l_func_nr[next(rng) % amount_function_mod]
        print("i: {}, func_nr: {}".format(i, func_nr))

        for bit_automaton in l_bit_automaton:
            # bit_automaton.execute_func(5)
            bit_automaton.execute_func(func_nr)
        pix2 = convert_bit_field_to_pix(l_bit_automaton)
        file_path = os.path.join(path_images_save, '{:04}.png'.format(i))
        Image.fromarray(pix2).save(file_path)
