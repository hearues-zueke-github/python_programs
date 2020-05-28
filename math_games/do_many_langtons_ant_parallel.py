#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import gzip
import os
import random
import sys
import subprocess

# from pysat.solvers import Glucose3

from copy import deepcopy

sys.path.append("../combinatorics/")
import different_combinations as combinations
sys.path.append("../math_numbers/")
from prime_numbers_fun import get_primes

import multiprocessing as mp
from multiprocessing import Process

from time import time
from functools import reduce

from PIL import Image

import numpy as np

from generate_LR_combinations import get_all_moves_string


sys.path.append("../")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

if __name__=='__main__':
    l_move_string = get_all_moves_string(max_base_length=5, max_depth=6, remove_duplicate_moves=True)
    print("len(l_move_string): {}".format(len(l_move_string)))            

    l_move_string = np.random.permutation(l_move_string).tolist()
    l_move_string = l_move_string[:2000]
    print("l_move_string: {}".format(l_move_string))

    CPU_COUNT = mp.cpu_count()-2
    length = len(l_move_string)
    length_part = length//CPU_COUNT
    l_idxs = [length_part*i for i in range(0, CPU_COUNT)]+[length]

    MAX_ITERS = 30000000

    def do_many_langtons_ant(l_move_string, proc_num):
        for i_move, move in enumerate(l_move_string, 1):
            print("proc_num: {}, i_move: {}, move: {}, MAX_ITERS: {}".format(proc_num, i_move, move, MAX_ITERS))
            p = subprocess.Popen(["./langtons_ant_multiple_machines.py", move, str(MAX_ITERS), ">", "output_nr_{}.txt".format(proc_num)])
            p.wait()

    l_proc = [Process(target=do_many_langtons_ant, args=(l_move_string[l_idxs[proc_num]:l_idxs[proc_num+1]], proc_num)) for proc_num in range(0, CPU_COUNT)]

    for proc in l_proc:
        proc.start()

    for proc in l_proc:
        proc.join()
