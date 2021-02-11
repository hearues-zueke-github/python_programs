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

def main():
    l_n = [380, 180, 284]

    l_mean = [(1, 3), (4, 8), (-1, 5)]
    l_std = [(1., 2.), (1., 0.7), (0.5, 1.25)]

    l_v = [np.random.normal(mean, std, (n, 2)).astype(np.float128) for n, mean, std in zip(l_n, l_mean, l_std)]

    points = np.vstack(l_v)

    cluster_amount = 3
    iterations = 100

    assert len(utils_cluster.l_color) >= cluster_amount
    # sys.exit()

    cluster_points, l_cluster_points_correspond, arr_error, l_error_cluster, l_cluster = utils_cluster.calculate_clusters(
        points=points,
        cluster_amount=cluster_amount,
        iterations=iterations,
    )

    utils_cluster.get_plots(
        cluster_points=cluster_points,
        l_cluster_points_correspond=l_cluster_points_correspond,
        arr_error=arr_error,
        l_error_cluster=l_error_cluster,
    )

    # xs, ys = points.T

    dm =  utils_cluster.do_clustering_silhouette(points, l_cluster, cluster_amount)
    l_cluster_val_s = dm.l_cluster_val_s

    # l_arr_val_s = np.array([(np.min(arr_val_s), np.median(arr_val_s), np.max(arr_val_s)) for arr_val_s in l_cluster_val_s])
    # pprint(l_arr_val_s)

    l_mean_val_s = [np.mean(l) for l in l_cluster_val_s]
    print('l_mean_val_s:')
    pprint(l_mean_val_s)

    return DotMap(locals(), _dynamic=None)


if __name__ == '__main__':
    dm = main()
