#! /usr/bin/python3

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import dill
import gzip
import os
import pdb
import re
import sys
import traceback

import multiprocessing as mp
import numpy as np
import pandas as pd
from z3 import *

from copy import deepcopy, copy
from dotmap import DotMap
from functools import reduce
from memory_tempfile import MemoryTempfile
from pprint import pprint
from shutil import copyfile

sys.path.append('..')
from utils import mkdirs
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

from pysat.solvers import Glucose3

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"
HOME_DIR = os.path.expanduser("~")+"/"
TEMP_DIR = MemoryTempfile().gettempdir()+"/"

OBJS_DIR_PATH = PATH_ROOT_DIR+'objs/'
mkdirs(OBJS_DIR_PATH)

if __name__ == '__main__':
    print('Hello World!')
    # sys.exit()

    AMOUNT_FACES = 6

    n = 3
    l_cells = [[IntVector('cell_{}_{}'.format(face_nr, row), n) for row in range(0, n)] for face_nr in range(0, AMOUNT_FACES)]
    
    s = Solver()

    for l_face in l_cells:
        for l_row in l_face:
            for var in l_row:
                s.add(var >= 0, var < AMOUNT_FACES)

    # add centers, if n % 2 != 0
    if n % 2 == 1:
        for i, l_face in enumerate(l_cells, 0):
            var = l_face[n//2][n//2]
            s.add(var == i)

    # add constraint for having a specific color combo for edges and corners
    # also add a constraint that only one unique edge and one unique corner do exist!

    print('Starting the check for the solution!')
    print(s.check())
    print(s.model())

    m = s.model()

    l_cells_val = [[[(lambda d: int(d.sexpr()))(m[v]) for v in l_row] for l_row in l_face] for l_face in l_cells]
    print("l_cells_val: {}".format(l_cells_val))

    sys.exit()

    # l_male = IntVector('male', NUM)
    # l_deg = IntVector('deg', NUM)
    # l_married = IntVector('married', NUM)
    # l_age = RealVector('age', NUM)
    # l_sal = RealVector('salery', NUM)

    l_age_sort = RealVector('l_age_sort', NUM)
    l_sal_sort = RealVector('l_sal_sort', NUM)
    
    l_age_male_sort_age = RealVector('l_age_male_sort_age', NUM)
    l_age_male_sort_male = IntVector('l_age_male_sort_male', NUM)
    l_age_ndeg_sort_age = RealVector('l_age_ndeg_sort_age', NUM)
    l_age_ndeg_sort_ndeg = IntVector('l_age_ndeg_sort_ndeg', NUM)
    l_age_single_sort_age = RealVector('l_age_single_sort_age', NUM)
    l_age_single_sort_single = IntVector('l_age_single_sort_single', NUM)

    s = Solver()
    
    for l in [l_male, l_deg, l_married]:
        for v in l:
            s.add(Or(v == 0, v == 1))

    # s.add(Sum([v for v in l_male]) == NUM_MALE)
    # s.add(Sum([v for v in l_deg]) == NUM_DEG)
    # s.add(Sum([v for v in l_married]) == NUM_MARRIED)

    # l_male_deg_single = []
    # for v_male, v_deg, v_married in zip(l_male, l_deg, l_married):
    #     l_male_deg_single.append(And(v_male==1, v_deg==1, v_married==0))
    # s.add(Sum([If(v, 1, 0) for v in l_male_deg_single]) == NUM_MALE_DEG_SINGLE)

    # l_fem_deg_single = []
    # for v_male, v_deg, v_married in zip(l_male, l_deg, l_married):
    #     l_fem_deg_single.append(And(v_male==0, v_deg==1, v_married==0))
    # s.add(Sum([If(v, 1, 0) for v in l_fem_deg_single]) == NUM_FEM_DEG_SINGLE)

    # l_fem_age_g50 = []
    # for v_male, v_age in zip(l_male, l_age):
    #     l_fem_age_g50.append(And(v_male==0, v_age > 50))
    # s.add(Sum([If(v, 1, 0) for v in l_fem_age_g50]) == NUM_FEM_AGE_G50)

    # l_single_male_deg_age_l50 = []
    # for v_male, v_deg, v_married, v_age in zip(l_male, l_deg, l_married, l_age):
    #     l_single_male_deg_age_l50.append(And(v_male==1, v_deg==1, v_married==0, v_age <= 50))
    # s.add(Sum([If(v, 1, 0) for v in l_single_male_deg_age_l50]) == NUM_SINGLE_MALE_DEG_AGE_L50)

    # s.add(Sum(l_age) / NUM == MEAN_AGE)
    # s.add(Sum([If(v_male==1, v_age, 0) for v_age, v_male in zip(l_age, l_male)]) / NUM_MALE == MEAN_AGE_MALE)
    # s.add(Sum([If(v_deg==1, v_age, 0) for v_age, v_deg in zip(l_age, l_deg)]) / NUM_DEG == MEAN_AGE_DEG)
    # s.add(Sum([If(v_married==1, v_age, 0) for v_age, v_married in zip(l_age, l_married)]) / NUM_MARRIED == MEAN_AGE_MARRIED)
    # s.add(Sum([If(And(v_male==1, v_deg==1, v_married==0), v_age, 0) for v_age, v_male, v_deg, v_married in zip(l_age, l_male, l_deg, l_married)]) / NUM_MALE_DEG_SINGLE == MEAN_AGE_MALE_DEG_SINGLE)
    # s.add(Sum([If(And(v_male==0, v_deg==1, v_married==0), v_age, 0) for v_age, v_male, v_deg, v_married in zip(l_age, l_male, l_deg, l_married)]) / NUM_FEM_DEG_SINGLE == MEAN_AGE_FEM_DEG_SINGLE)

    # s.add(Sum(l_sal) / NUM == MEAN_SAL)
    # s.add(Sum([If(v_male==0, v_sal, 0) for v_sal, v_male in zip(l_sal, l_male)]) / (NUM - NUM_MALE) == MEAN_SAL_FEM)
    # s.add(Sum([If(v_deg==0, v_sal, 0) for v_sal, v_deg in zip(l_sal, l_deg)]) / (NUM - NUM_DEG) == MEAN_SAL_NDEG)
    # s.add(Sum([If(v_married==0, v_sal, 0) for v_sal, v_married in zip(l_sal, l_married)]) / (NUM - NUM_MARRIED) == MEAN_SAL_SINGLE)
    # s.add(Sum([If(And(v_male==0, v_age <= 50), v_sal, 0) for v_sal, v_age, v_male in zip(l_sal, l_age, l_male)]) / (NUM - NUM_FEM_AGE_G50) == MEAN_SAL_FEM_L50)
    s.add(Sum([If(And(v_male==1, v_deg==1, v_married==0), v_sal, 0) for v_sal, v_male, v_deg, v_married in zip(l_sal, l_male, l_deg, l_married)]) / NUM_SINGLE_MALE_DEG_AGE_L50 == MEAN_SAL_SINGLE_MALE_DEG_L50)

    print('Starting the check for the solution!')
    print(s.check())
    print(s.model())

    m = s.model()

    print('-'*40)
    l_male_val = [(lambda d: int(d.sexpr()))(m[v]) for v in l_male]
    print("l_male_val: {}".format(l_male_val))
    l_deg_val = [(lambda d: int(d.sexpr()))(m[v]) for v in l_deg]
    print("l_deg_val: {}".format(l_deg_val))
    l_married_val = [(lambda d: int(d.sexpr()))(m[v]) for v in l_married]
    print("l_married_val: {}".format(l_married_val))
    
    print(' -'*20)
    l_age_val = [(lambda d: d.numerator().as_long()/d.denominator().as_long())(m[v]) for v in l_age]
    print("l_age_val: {}".format(l_age_val))
    l_sal_val = [(lambda d: d.numerator().as_long()/d.denominator().as_long())(m[v]) for v in l_sal]
    print("l_sal_val: {}".format(l_sal_val))
