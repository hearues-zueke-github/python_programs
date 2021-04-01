#! /usr/bin/env -S /usr/bin/time /usr/bin/pypy3.7 -i

# -*- coding: utf-8 -*-

#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.8.6 -i

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

from datetime import datetime
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
from utils_multiprocessing_manager import MultiprocessingManager

sys.path.append("../clustering/")
import utils_cluster

from bit_automaton import BitAutomaton

def range_gen(g, n):
    i = 0
    v = next(g)
    while v != None:
        yield v
        v = next(g)
        i += 1
        if i == n:
            break

# def range_gen(g, n):
#     i = 0
#     while (v := next(g)) != None:
#         yield v
#         i += 1
#         if i == n:
#             break


def prepare_functions(funcs_str, frame) -> Tuple[List[Any], Any, Any, int]:
    bit_automaton = BitAutomaton().init_vals(h=10, w=8, frame=frame, frame_wrap=True, l_func=[], func_inv=None, func_rng=None)
    
    d_vars = bit_automaton.d_vars
    field_size = bit_automaton.field_size
    d_func = {}
    exec(funcs_str, d_vars, d_func)
    print("d_func.keys(): {}".format(d_func.keys()))

    d_func_keys = d_func.keys()
    assert 'rng' in d_func_keys
    assert 'inv' in d_func_keys
    assert 'l_func' in d_func_keys
    assert 'start_seed' in d_func_keys

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

    start_seed = d_func['start_seed']

    return l_func, func_inv, func_rng, start_seed


def create_new_automaton():
    frame = 1
    frame_wrap = True
    
    bit_automaton = BitAutomaton().init_vals(h=10, w=8, frame=frame, frame_wrap=True, l_func=[], func_inv=None, func_rng=None)

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

    def get_random_function_str_body_dm(
        func_nr : int,
        l_var : List[str],
        n_and_min : int,
        n_and_max : int,
        n_or_min : int,
        n_or_max : int,
    ) -> str:
        l_vars_inv = [f'inv({v})' for v in l_var]

        arr_var = np.vstack((l_var, l_vars_inv))
        arr_var_i = np.arange(0, arr_var.shape[1])
        arr_var_row = np.arange(0, 2)

        func_name = f'fun_{func_nr}'
        # s = f'def {func_name}():\n'
        s = ''
        l_new_var = []
        l_l_chosen_var = []
        l_arr_var_names = []
        l_formel_and = []
        n_or = np.random.randint(n_or_min, n_or_max + 1)
        for i_or in range(0, n_or):
            new_var = f'a_{i_or}'
            l_new_var.append(new_var)

            n_and = np.random.randint(n_and_min, n_and_max + 1)
            arr_chosen_var = np.unique(np.random.choice(arr_var_i, (n_and, )))
            arr_inv_var = np.random.choice(arr_var_row, (arr_chosen_var.shape[0], ))

            arr_var_names = np.sort(arr_var[arr_inv_var, arr_chosen_var])
            l_arr_var_names.append(arr_var_names)

            formel_and = ' & '.join(arr_var_names)
            l_formel_and.append(formel_and)

        l_former_and_sorted = sorted(set(l_formel_and))
        for new_var, formel_and in zip(l_new_var, l_formel_and):
            s += f' {new_var} = {formel_and}\n'
            l_l_chosen_var.append((arr_chosen_var, arr_inv_var))

        last_formel = ' | '.join(l_new_var)
        s += f' return {last_formel}\n'
        return DotMap(locals(), dynamic_=None)

    # TODO: save the data to a file too!
    n_func_min = 1
    n_func_max = 1
    # n_func_min = 2
    # n_func_max = 4
    n_or_min = 2
    n_or_max = 10
    n_and_min = 2
    n_and_max = 6

    n_func = np.random.randint(n_func_min, n_func_max + 1)
    l_dm_local = [
        get_random_function_str_body_dm(
            func_nr=i,
            l_var=l_var,
            n_and_min=n_and_min,
            n_and_max=n_and_max,
            n_or_min=n_or_min,
            n_or_max=n_or_max,
        ) for i in range(0, n_func)]
    l_func_str_body = [dm.s for dm in l_dm_local]

    l_func_str_body_sorted  = sorted(set(l_func_str_body))
    l_func_name = [dm.func_name for dm in l_dm_local[:len(l_func_str_body_sorted)]]
    l_func_str_sorted = [f'def {func_name}():\n{s}' for func_name, s in zip(l_func_name, l_func_str_body_sorted)]

    func_str = ', '.join([func_name for func_name in l_func_name])
    funcs_str_func = '\n'.join(l_func_str_sorted)
    
    func_list_str = 'l_func = [{}]'.format(func_str)

    funcs_str = '\n'.join([
        inspect.cleandoc(funcs_str_rng)+'\n',
        funcs_str_inv,
        funcs_str_func,
        func_list_str,
        'start_seed = 0',
        '\n',
    ])

    # sys.exit()

    # TODO: create some random functions too!

    l_func, func_inv, func_rng, start_seed = prepare_functions(funcs_str=funcs_str, frame=frame)


    # path_images = PATH_ROOT_DIR + 'images/'

    # file_name = 'nature_trees_river_84761_800x600.jpg'
    # image_path = path_images + file_name

    # img = Image.open(image_path)
    # pix = np.array(img)

    pix_orig = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)

    def extract_bits_from_pix(pix : np.ndarray) -> np.ndarray:
        pix_bits = np.array([(channel>>i)&0x1 for channel in pix.transpose(2, 0, 1) for i in range(7, -1, -1)], dtype=np.uint8)
        return pix_bits
        # return pix_bits.transpose(1, 2, 0)


    # def extract_rgb_from_pix(pix : np.ndarray) -> np.ndarray:
    #     return pix.transpose(2, 0, 1)

    pix_bits = extract_bits_from_pix(pix_orig)

    pix = pix_bits[0]*255
    # pix = pix_bits[:1]


    h, w = pix_bits.shape[1:]
    # h, w = pix_bits.shape[:2]
    # sys.exit()


    # folder_name = 'test_other'
    folder_name_suffix = '{}_{}'.format(
        datetime.strftime(datetime.now(), '%Y-%m-%d_%H:%M:%S'),
        ''.join(np.random.choice(list(string.hexdigits), (8, ))).upper(),
    )
    folder_name = f'test_other_{folder_name_suffix}'
    dir_path_save_images = os.path.join(TEMP_DIR, 'save_images/')
    dir_path_images = os.path.join(dir_path_save_images, folder_name)
    # dir_path_images = os.path.join(TEMP_DIR, 'save_images/{}/'.format(file_name.split('.')[0]))
    utils.mkdirs(dir_path_images)
    
    dir_path_combined_images = os.path.join(dir_path_save_images, 'combined_images')
    utils.mkdirs(dir_path_combined_images)

    dir_path_combined_images_xor = os.path.join(dir_path_save_images, 'combined_images_xor')
    utils.mkdirs(dir_path_combined_images_xor)

    file_path_funcs = os.path.join(dir_path_images, 'test_functions_python.py')
    with open(file_path_funcs, 'w') as f:
        f.write(funcs_str)

    arr_bits = pix_bits[:1]
    # arr_bits = np.array([[list(itertools.chain(*[list(map(int, bin(b)[2:].zfill(8))) for b in v])) for v in row] for row in pix], dtype=np.uint8).transpose(2, 0, 1)

    amount_bit_automaton = arr_bits.shape[0]

    l_bit_automaton = [BitAutomaton().init_vals(h=h, w=w, frame=frame, frame_wrap=frame_wrap, l_func=l_func, func_inv=func_inv, func_rng=func_rng) for _ in range(0, amount_bit_automaton)]

    for bit_automaton, bits in zip(l_bit_automaton, arr_bits):
        bit_automaton.set_field(bits.astype(np.bool))

    def convert_bit_field_to_pix_1_bit(l_bit_automaton):
        assert len(l_bit_automaton) == 1 # 1 bits

        arr = (l_bit_automaton[0]<<0).astype(np.uint8) * 255
        return arr


    def convert_bit_field_to_pix_8_bit(l_bit_automaton):
        assert len(l_bit_automaton) == 8 # 8 bits

        arr = np.sum([bit_automaton<<j for j, bit_automaton in zip(range(7, -1, -1), l_bit_automaton)], axis=0).astype(np.uint8)
        return arr


    def convert_bit_field_to_pix_24_bit(l_bit_automaton):
        assert len(l_bit_automaton) == 24 # 24 bits

        arr = np.array([np.sum([bit_automaton<<j for j, bit_automaton in zip(range(7, -1, -1), l_bit_automaton[8*i:8*(i+1)])], axis=0) for i in range(0, 3)], dtype=np.uint8)
        return arr.transpose(1, 2, 0)

    convert_bit_field_to_pix = convert_bit_field_to_pix_1_bit
    # convert_bit_field_to_pix = convert_bit_field_to_pix_8_bit
    # convert_bit_field_to_pix = convert_bit_field_to_pix_24_bit

    pix2 = convert_bit_field_to_pix(l_bit_automaton[:1])
    assert np.all(pix2 == pix)


    print("i: {}".format(0))
    file_path = os.path.join(dir_path_images, '{:04}.png'.format(0))
    Image.fromarray(pix2).save(file_path)
    
    l_pix = [pix2]

    cols = 5
    rows = 10
    iterations_amount = cols * rows
    l_func_nr = list(range(0, len(l_func)))
    amount_function_mod = len(l_func_nr)
    rng = func_rng(seed=start_seed)
    for i in range(1, iterations_amount):
    # for i in range(1, 100):
        func_nr = l_func_nr[next(rng) % amount_function_mod]
        print("i: {}, func_nr: {}".format(i, func_nr))

        for bit_automaton in l_bit_automaton:
            # bit_automaton.execute_func(5)
            bit_automaton.execute_func(func_nr)
        pix2 = convert_bit_field_to_pix(l_bit_automaton)
        file_path = os.path.join(dir_path_images, '{:04}.png'.format(i))
        Image.fromarray(pix2).save(file_path)

        l_pix.append(pix2)

    arr_pixs = np.array(l_pix) // 255

    amount_historic_numbers = (frame * 2 + 1)**2 + 1
    # e.g.: frame = 2 -> (2*2+1)**2 = 25, also zero included -> 26

    def calculate_pix_erosion_dilation_4_neighborhood_sum_ranges_unique(frame, pix):
        pix_sum_ranges = np.zeros(pix.shape, dtype=np.int_)
        
        for move_y, move_x in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]:
            pix_sum_ranges += np.roll(np.roll(pix, move_x, axis=1), move_y, axis=0)
        
        pix_erosion = (pix_sum_ranges == 5).astype(np.uint8)
        pix_dilation = (pix_sum_ranges == 5).astype(np.uint8)

        pix_erosion_sum_ranges = np.zeros(pix1.shape, dtype=np.int_)
        pix_dilation_sum_ranges = np.zeros(pix2.shape, dtype=np.int_)
        for move_y in range(-frame, frame+1, 1):
            for move_x in range(-frame, frame+1, 1):
                pix_erosion_sum_ranges += np.roll(np.roll(pix1, move_x, axis=1), move_y, axis=0)
                pix_dilation_sum_ranges += np.roll(np.roll(pix2, move_x, axis=1), move_y, axis=0)

        u_e, c_e = np.unique(pix_erosion_sum_ranges, return_counts=True)
        u_d, c_d = np.unique(pix_dilation_sum_ranges, return_counts=True)

        return u_e, c_e, u_d, c_d


    def calculate_pix_sum_ranges_unique(frame, pix1, pix2, arr_row):
        pix1_sum_ranges = np.zeros(pix1.shape, dtype=np.int_)
        pix2_sum_ranges = np.zeros(pix2.shape, dtype=np.int_)
        pix1xor2 = pix1 ^ pix2
        pix1xor2_sum_ranges = np.zeros(pix2.shape, dtype=np.int_)
        for move_y in range(-frame, frame+1, 1):
            for move_x in range(-frame, frame+1, 1):
                pix1_sum_ranges += np.roll(np.roll(pix1, move_x, axis=1), move_y, axis=0)
                pix2_sum_ranges += np.roll(np.roll(pix2, move_x, axis=1), move_y, axis=0)
                pix1xor2_sum_ranges += np.roll(np.roll(pix1xor2, move_x, axis=1), move_y, axis=0)

        u1, c1 = np.unique(pix1_sum_ranges, return_counts=True)
        u2, c2 = np.unique(pix2_sum_ranges, return_counts=True)
        u1xor2, c1xor2 = np.unique(pix1xor2_sum_ranges, return_counts=True)

        u1_e, c1_e, u1_d, c1_d = calculate_pix_erosion_dilation_4_neighborhood_sum_ranges_unique(frame, pix1)
        u2_e, c2_e, u2_d, c2_d = calculate_pix_erosion_dilation_4_neighborhood_sum_ranges_unique(frame, pix2)
        u1xor2_e, c1xor2_e, u1xor2_d, c1xor2_d = calculate_pix_erosion_dilation_4_neighborhood_sum_ranges_unique(frame, pix1xor2)

        arr_row[u1 + amount_historic_numbers*0] = c1
        arr_row[u2 + amount_historic_numbers*1] = c2
        arr_row[u1xor2 + amount_historic_numbers*2] = c1xor2

        arr_row[u1_e + amount_historic_numbers*3] = c1_e
        arr_row[u1_d + amount_historic_numbers*4] = c1_d
        arr_row[u2_e + amount_historic_numbers*5] = c2_e
        arr_row[u2_d + amount_historic_numbers*6] = c2_d
        arr_row[u1xor2_e + amount_historic_numbers*7] = c1xor2_e
        arr_row[u1xor2_d + amount_historic_numbers*8] = c1xor2_d

    arr_historic_ranges = np.zeros((arr_pixs.shape[0]-1, 9*amount_historic_numbers), dtype=np.int_)

    for i, (pix1, pix2) in enumerate(zip(arr_pixs[:-1], arr_pixs[1:]), 0):
        calculate_pix_sum_ranges_unique(frame, pix1, pix2, arr_historic_ranges[i])

    # TODO: find dynamic_ in other files too! correct this in every files!
    dm_obj = DotMap(_dynamic=None)
    dm_obj['frame'] = frame
    dm_obj['frame_wrap'] = frame_wrap
    dm_obj['l_bit_automaton'] = l_bit_automaton
    dm_obj['arr_pixs'] = arr_pixs
    dm_obj['arr_historic_ranges'] = arr_historic_ranges
    dm_obj['func_str'] = l_func_str_sorted[0]
    dm_obj['_version'] = utils_cluster.__version__

    with gzip.open(os.path.join(dir_path_images, utils_cluster.dm_obj_file_name), 'wb') as f:
        dill.dump(dm_obj, f)

    # sys.exit()

    def combine_all_pix(l_pix, w_space_horizontal=10, h_space_vertical=10):
        h_space_horizontal = l_pix[0].shape[0]
        # w_space_horizontal = 10
        arr_space_horizontal = np.zeros((h_space_horizontal, w_space_horizontal), dtype=np.uint8) + 0x80

        def combine_l_pix_horizontal(l_pix_part : List[np.ndarray]) -> np.ndarray:
            l = [l_pix_part[0]]
            for pix in l_pix_part[1:]:
                l.append(arr_space_horizontal)
                l.append(pix)
            return np.hstack(l)

        l_pix_horizontal = [combine_l_pix_horizontal(l_pix[cols*i:cols*(i+1)]) for i in range(0, rows)]

        w_space_vertical = l_pix_horizontal[0].shape[1]
        # h_space_vertical = 10
        arr_space_vertical = np.zeros((h_space_vertical, w_space_vertical), dtype=np.uint8) + 0x80

        def combine_l_pix_vertical(l_pix_part : List[np.ndarray]) -> np.ndarray:
            l = [l_pix_part[0]]
            for pix in l_pix_part[1:]:
                l.append(arr_space_vertical)
                l.append(pix)
            return np.vstack(l)
        
        pix_vertical = combine_l_pix_vertical(l_pix_horizontal)

        return pix_vertical

    pix_combine = combine_all_pix(l_pix=l_pix)
    file_path_combine = os.path.join(dir_path_combined_images, '{}.png'.format(folder_name))
    Image.fromarray(pix_combine).save(file_path_combine)

    l_pix_xor = [np.zeros(l_pix[0].shape, dtype=np.uint8) + 0x40] + [pix1 ^ pix2 for pix1, pix2 in zip(l_pix[:-1], l_pix[1:])]
    pix_combine_xor = combine_all_pix(l_pix=l_pix_xor)
    file_path_combine_xor = os.path.join(dir_path_combined_images_xor, '{}.png'.format(folder_name))
    Image.fromarray(pix_combine_xor).save(file_path_combine_xor)


if __name__ == '__main__':
    # with gzip.open('/run/user/1000/save_images/test_other_2021-01-18_09:18:00_1C04BE34/dm_obj.pkl.gz', 'rb') as f:
    #     dm_obj = dill.load(f)
    
    # create_new_automaton()
    # sys.exit()

    def create_many_new_automaton(n: int) -> None:
        np.random.seed()
        for i in range(0, n):
            print("i: {}".format(i))
            create_new_automaton()

    mult_proc_mng = MultiprocessingManager(cpu_count=mp.cpu_count())

    print('Define new Function!')
    mult_proc_mng.define_new_func('func_create_many_new_automaton', create_many_new_automaton)

    print('Do the jobs!!')
    l_arguments = []
    l_ret = mult_proc_mng.do_new_jobs(
        ['func_create_many_new_automaton']*mult_proc_mng.cpu_count,
        [(30, )]*mult_proc_mng.cpu_count,
    )
    print("len(l_ret): {}".format(len(l_ret)))

    del mult_proc_mng
