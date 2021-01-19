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

from datetime import datetime
from collections import defaultdict
from copy import deepcopy
from dotmap import DotMap
from operator import itemgetter

from pprint import pprint

from os.path import expanduser

import itertools

import matplotlib.pyplot as plt

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

import utils_cluster

if __name__ == '__main__':
    l_n = [30, 100, 84]

    l_mean = [(1, 3), (4, 8), (-1, 5)]
    l_std = [(1., 2.), (1., 0.7), (0.5, 1.25)]

    l_v = [np.random.normal(mean, std, (n, 2)) for n, mean, std in zip(l_n, l_mean, l_std)]

    points = np.vstack(l_v)

    # sys.exit()

    cluster_points, arr_error = utils_cluster.calculate_clusters(points, 4, 100)

    xs, ys = points.T
    xs_c, ys_c = cluster_points.T

    plt.figure()

    plt.plot(xs, ys, color='#0000FF', marker='.', ms=2., ls='')
    plt.plot(xs_c, ys_c, color='#00FF00', marker='.', ms=8., ls='')

    plt.figure()

    plt.plot(np.arange(0, arr_error.shape[0]), arr_error, color='#00FF00', marker='.', ms=8., ls='-')

    plt.show()
