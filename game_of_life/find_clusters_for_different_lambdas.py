#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.8.6 -i

# -*- coding: utf-8 -*-

# old one
#! /usr/bin/env -S /usr/bin/time /usr/bin/pypy3.7 -i

import dill
import gzip
import os
import string
import sys
import inspect
import textwrap

import matplotlib.pyplot as plt

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
from utils_multiprocessing_manager import MultiprocessingManager

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

            # if amount >= 50:
            # if amount >= 250:
            #     break

    # l_arr_nr = [i for i, arr in enumerate(l_arr_historic_ranges, 0) for _ in range(0, arr.shape[0])]
    arr_historic_ranges_all = np.vstack(l_arr_historic_ranges)
    amount_features = arr_historic_ranges_all.shape[1]

    # print("arr_historic_ranges_all.shape: {}".format(arr_historic_ranges_all.shape))
    arr_histroic_no_dups = np.array(list(set(map(tuple, arr_historic_ranges_all.tolist()))))

    points = arr_histroic_no_dups.astype(np.float128)

    print("points.shape: {}".format(points.shape))
    # input('ENTER...')

    # sys.exit()

    # print("points.shape: {}".format(points.shape))
    # input('Enter...')

    # l_mean_cluster_silhouette = []


    # TODO 2021.03.06: make it multiprocessing possible!

    def calc_many_cluster_amount(l_cluster_amount):
        d = {}

        for cluster_amount in l_cluster_amount:

            print("cluster_amount: {}".format(cluster_amount))

            # cluster_amount = 15
            iterations=300
            try:
                calc_cluster_data = utils_cluster.calculate_clusters(
                    points=points,
                    cluster_amount=cluster_amount,
                    iterations=iterations,
                )

                # d_calc_cluster_data[cluster_amount] = calc_cluster_data

                cluster_points = calc_cluster_data.cluster_points
                l_cluster_points_correspond = calc_cluster_data.l_cluster_points_correspond
                arr_error = calc_cluster_data.arr_error
                l_error_cluster = calc_cluster_data.l_error_cluster
                l_cluster = calc_cluster_data.arr_argmin
                
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
                # d_df_stats_cluster_points[cluster_amount] = df_stats_cluster_points

                print("arr_stats_cluster_points:\n{}".format(arr_stats_cluster_points))

                # l_cluster_points.append(cluster_points)

                d[cluster_amount] = {
                    # 'calc_cluster_data': calc_cluster_data,
                    'arr_error': arr_error,
                    # 'df_stats_cluster_points': df_stats_cluster_points,
                    # 'cluster_points': cluster_points,
                }
            except:
                d[cluster_amount] = {
                    # 'calc_cluster_data': calc_cluster_data,
                    'arr_error': [-1],
                    # 'df_stats_cluster_points': df_stats_cluster_points,
                    # 'cluster_points': cluster_points,
                }


        return d

    # for cluster_amount in range(50, 100, 5):
    # for cluster_amount in range(100, 300, 1):
    # for cluster_amount in range(20, 40, 1):
        

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

    # mult_proc_mng = MultiprocessingManager(cpu_count=mp.cpu_count())
    mult_proc_mng = MultiprocessingManager(cpu_count=mp.cpu_count()-3)

    l_cluster_amount = list(range(2, 350, 20))
    # l_cluster_amount = list(range(2, 50, 1))
    split_parts = mult_proc_mng.worker_amount
    # split_parts = mult_proc_mng.worker_amount*3
    l_split_cluster_amount = [[l_cluster_amount[i] for i in range(cpu_num, len(l_cluster_amount), mult_proc_mng.worker_amount)] for cpu_num in range(0, mult_proc_mng.worker_amount)]

    print('Define new Function!')
    mult_proc_mng.define_new_func('func_calc_many_cluster_amount', calc_many_cluster_amount)

    l_ret = mult_proc_mng.do_new_jobs(
        ['func_calc_many_cluster_amount']*len(l_split_cluster_amount),
        [(l, ) for l in l_split_cluster_amount],
    )
    print("len(l_ret): {}".format(len(l_ret)))
    # print("l_ret: {}".format(l_ret))

    del mult_proc_mng

    d_all = {}
    for d in l_ret:
        d_all.update(d)

    assert set(l_cluster_amount) == set(d_all.keys())

    d_calc_cluster_data = {}
    d_df_stats_cluster_points = {}
    l_cluster_points = []
    d_arr_error = {}

    for cluster_amount in sorted(d_all.keys()):
        d = d_all[cluster_amount]

        d_arr_error[cluster_amount] = d['arr_error']
        
        # d_calc_cluster_data[cluster_amount] = d['calc_cluster_data']
        # d_df_stats_cluster_points[cluster_amount] = d['df_stats_cluster_points']
        # l_cluster_points.append(d['cluster_points'])


    xs = np.array(l_cluster_amount)
    ys = np.array([d_arr_error[cluster_amount][-1] for cluster_amount in l_cluster_amount])
    # ys = np.array([d_calc_cluster_data[cluster_amount].arr_error[-1] for cluster_amount in l_cluster_amount])

    plt.close('all')

    plt.figure()

    plt.xlabel('Cluster amount')
    plt.ylabel('Error (MSE)')

    plt.title('Last error MSE for each cluster amount')

    plt.plot(xs, ys, marker='.', color='#0000FF', markersize=10, linestyle='')

    plt.savefig(os.path.join(dir_path_save_images, 'cluster_amount_error_plot.png'), dpi=500)


    plt.figure()

    plt.xlabel('Cluster amount')
    plt.ylabel('Error (MSE)')

    plt.title('Last error MSE for each cluster amount (diff n)')

    plt.plot(xs, ys / xs, marker='.', color='#0000FF', markersize=10, linestyle='')

    plt.savefig(os.path.join(dir_path_save_images, 'cluster_amount_error_plot_div_n.png'), dpi=500)

    sys.exit()

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
