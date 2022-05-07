#! /usr/bin/python3.10

# pip installed libraries
import dill
import glob
import gzip
import keyboard
import os
import requests
import sh
import string
import subprocess
import sys
import time
import tty
import yaml

import multiprocessing as mp
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from copy import deepcopy
from io import StringIO
from memory_tempfile import MemoryTempfile
from multiprocessing import Pool
from PIL import Image
from typing import Dict

from numpy.random import Generator, PCG64
from scipy.sparse import csr_matrix, coo_matrix

HOME_DIR = os.path.expanduser("~")
PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = MemoryTempfile().gettempdir()
PYTHON_PROGRAMS_DIR = os.path.join(HOME_DIR, 'git/python_programs')

# set the relative/absolute path where the utils_load_module.py file is placed!
sys.path.append(PYTHON_PROGRAMS_DIR)
from utils_load_module import load_module_dynamically

var_glob = globals()
load_module_dynamically(**dict(var_glob=var_glob, name='utils', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='utils_multiprocessing_manager', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils_multiprocessing_manager.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='utils_serialization', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils_serialization.py")))

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

get_pkl_gz_obj = utils_serialization.get_pkl_gz_obj
save_pkl_gz_obj = utils_serialization.save_pkl_gz_obj
load_pkl_gz_obj = utils_serialization.load_pkl_gz_obj

save_pkl_obj = utils_serialization.save_pkl_obj
load_pkl_obj = utils_serialization.load_pkl_obj

import utils_tetris
from neuronal_network import NeuralNetwork
from utils_multiprocessing_parallel import MultiprocessingParallelManager

TETRIS_TEMP_DIR = os.path.join(TEMP_DIR, "tetris_temp")
mkdirs(TETRIS_TEMP_DIR)

if __name__ == '__main__':
	print("Hello World!")

	field_w = 6
	field_h = 20

	d_pieces = utils_tetris.prepare_pieces(field_w=field_w)	

	arr_tetris_pieces_rotate = d_pieces['arr_tetris_pieces_rotate']
	df_piece_positions = d_pieces['df_piece_positions']
	arr_tetris_pieces_base = d_pieces['arr_tetris_pieces_base']

	n_pieces = arr_tetris_pieces_base.shape[0]
