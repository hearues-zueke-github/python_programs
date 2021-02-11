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
import pandas as pd

sys.path.append("../")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj
import global_object_getter_setter
import utils

sys.path.append('../clustering')
import utils_cluster

from bit_automaton import BitAutomaton

if __name__ == '__main__':
    dir_path_save_images = os.path.join(TEMP_DIR, 'save_images/')
    assert os.path.exists(dir_path_save_images)

    dm_obj_file_name = utils_cluster.dm_obj_file_name

    frame = 1

    l_arr_historic_ranges : List[np.ndarray] = []
    l_func_str : List[str] = []
    amount : int = 0
    for root, dirs, files in os.walk(dir_path_save_images):
        if dm_obj_file_name in files:
            print("root: {}, files: {}".format(root, files))
            print("files: {}".format(files))
            with gzip.open(os.path.join(root, dm_obj_file_name), 'rb') as f:
                dm_obj = dill.load(f)
            if dm_obj.frame == frame:
                amount += 1
                print("amount: {}".format(amount))
                l_arr_historic_ranges.append(dm_obj.arr_historic_ranges)
                l_func_str.append(dm_obj.func_str)

    # l_arr_nr = [i for i, arr in enumerate(l_arr_historic_ranges, 0) for _ in range(0, arr.shape[0])]
    arr_historic_ranges_all = np.vstack(l_arr_historic_ranges)
    amount_features = arr_historic_ranges_all.shape[1]

    # print("arr_historic_ranges_all.shape: {}".format(arr_historic_ranges_all.shape))
    arr_histroic_no_dups = np.array(list(set(map(tuple, arr_historic_ranges_all.tolist()))))

    points = arr_histroic_no_dups.astype(np.float128)

    print("points.shape: {}".format(points.shape))
    input('ENTER...')

    # print("points.shape: {}".format(points.shape))
    # input('Enter...')

    l_mean_cluster_silhouette = []
    l_df_stats_cluster_points = []
    l_cluster_points = []

    # for cluster_amount in range(50, 100, 5):
    # for cluster_amount in range(100, 300, 1):
    # for cluster_amount in range(20, 40, 1):
    for cluster_amount in range(10, 22, 3):
        print("cluster_amount: {}".format(cluster_amount))

        # cluster_amount = 15
        iterations=300
  
        cluster_points, l_cluster_points_correspond, arr_error, l_error_cluster, l_cluster = utils_cluster.calculate_clusters(
            points=points,
            cluster_amount=cluster_amount,
            iterations=iterations,
        )

        # sys.exit()

        # utils_cluster.get_plots(
        #     cluster_points=cluster_points,
        #     l_cluster_points_correspond=l_cluster_points_correspond,
        #     arr_error=arr_error,
        #     l_error_cluster=l_error_cluster,
        # )

        # sys.exit()

        arr_stats_cluster_points = np.vstack((
            np.mean(cluster_points, axis=1),
            np.std(cluster_points, axis=1),
            np.min(cluster_points, axis=1),
            np.quantile(cluster_points, 0.25, axis=1),
            np.median(cluster_points, axis=1),
            np.quantile(cluster_points, 0.75, axis=1),
            np.max(cluster_points, axis=1),
        )).T

        df_stats_cluster_points = pd.DataFrame(data=arr_stats_cluster_points, columns=[
            'mean', 'std', 'min', 'quantile', 'median', 'quantile', 'max'
        ])
        l_df_stats_cluster_points.append(df_stats_cluster_points)

        print("arr_stats_cluster_points:\n{}".format(arr_stats_cluster_points))

        l_cluster_points.append(cluster_points)

        # dm =  utils_cluster.do_clustering_silhouette(points, l_cluster, cluster_amount)
        # l_cluster_val_s = dm.l_cluster_val_s

        # l_mean_val_s = np.array([np.mean(l) for l in l_cluster_val_s])
        # print('l_mean_val_s:')
        # pprint(l_mean_val_s)
        # print("np.sum(l_mean_val_s > 0): {}".format(np.sum(l_mean_val_s > 0)))

        # l_mean_cluster_silhouette.append((l_mean_val_s, l_mean_val_s.shape[0], np.sum(l_mean_val_s > 0.)))

        # print('l_mean_cluster_silhouette:')
        # pprint(l_mean_cluster_silhouette)

        # if np.all(l_mean_val_s > 0.):
        #     print('Found!')
        #     break


        # l_arr_val_s = np.array([(np.min(arr_val_s), np.median(arr_val_s), np.max(arr_val_s)) for arr_val_s in l_cluster_val_s])
        # pprint(l_arr_val_s)

        # if np.all(l_arr_val_s[:, 1] > 0):
        #     print('Found!!!')
        #     break

    def cluster_points_sorting(cluster_points):
        return np.array(sorted(cluster_points.tolist()))

    arr1 = cluster_points_sorting(l_cluster_points[0])
    arr2 = cluster_points_sorting(l_cluster_points[1])

    arr_err_sum  = np.sum((arr1.reshape((-1, 1, amount_features))-arr2.reshape((1, -1, amount_features)))**2, axis=2)

    arr_argmin1 = np.argmin(arr_err_sum, axis=1)
    arr_argmin2 = np.argmin(arr_err_sum, axis=0)

    u1, c1 = np.unique(arr_argmin1, return_counts=True)
    u2, c2 = np.unique(arr_argmin2, return_counts=True)

    print("u1: {}".format(u1))
    print("c1: {}".format(c1))
    print("u2: {}".format(u2))
    print("c2: {}".format(c2))
