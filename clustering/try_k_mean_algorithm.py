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

if __name__ == '__main__':
    l_n = [30, 100, 84]

    l_mean = [(1, 3), (4, 8), (-1, 5)]
    l_std = [(1., 2.), (1., 0.7), (0.5, 1.25)]

    l_v = [np.random.normal(mean, std, (n, 2)) for n, mean, std in zip(l_n, l_mean, l_std)]

    points = np.vstack(l_v)

    def calculate_clusters(points, cluster_amount, iterations):
        point_dim = points.shape[1]
        cluster_points = points[np.random.permutation(np.arange(0, len(points)))[:cluster_amount]].copy()
        print("before cluster_points:\n{}".format(cluster_points))

        # calc new clusters!
        for i_nr in range(1, iterations + 1):
            print("i_nr: {}".format(i_nr))

            arr_argmin = np.argmin(np.sum((points.reshape((-1, 1, point_dim)) - cluster_points.reshape((1, -1, point_dim)))**2, axis=2), axis=1)

            for i in range(0, cluster_amount):
                arr = points[arr_argmin==i]
                cluster_points[i] = np.mean(arr, axis=0)
            
            print("- after cluster_points:\n{}".format(cluster_points))

        return cluster_points
    # sys.exit()

    cluster_points = calculate_clusters(points, 4, 100)

    xs, ys = points.T
    xs_c, ys_c = cluster_points.T

    plt.figure()

    plt.plot(xs, ys, color='#0000FF', marker='.', ms=2., ls='')
    plt.plot(xs_c, ys_c, color='#00FF00', marker='.', ms=8., ls='')

    plt.show()
