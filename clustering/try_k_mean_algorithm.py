#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.10 -i

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

import matplotlib.pyplot as plt

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

import utils_cluster

if __name__ == '__main__':
	# create an example of points, where there are actually seeable clusters
	# for this example only one 2d example is shown, but k-means can be used
	# for a n-th dimensional point/vector too!

	l_n_mean_std = [
		(380, (1, 3), (1., 2.), ),
		(280, (4, 8), (1., 0.7), ),
		(384, (-1, 5), (0.5, 1.25), ),
		(450, (2, -1), (0.6, 2.25), ),
	]

	l_v = [np.random.normal(mean, std, (n, 2)).astype(np.float64) for n, mean, std in l_n_mean_std]

	arr_cluster_mean_vec = np.array([v for _, v, _ in l_n_mean_std], dtype=np.float64)
	arr_point = np.vstack(l_v)

	arr_point_tpl = np.core.records.fromarrays(arr_point.T, dtype=[(f'c{i}', 'f8') for i in range(0, arr_point.shape[1])])
	u_arr_point_tpl, c_arr_point_tpl = np.unique(arr_point_tpl, return_counts=True)
	arr_point_unique = u_arr_point_tpl.view((np.float64, len(u_arr_point_tpl.dtype.names)))

	# check if any two points are the same
	if np.any(c_arr_point_tpl > 1):
		arr_point = u_arr_point_tpl[c_arr_point_tpl == 1]

	l_color =  ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255', '#AA4499']

	plt.figure()

	plt.title("An example of the cluster points in 2d")
	for i, arr_v in enumerate(l_v, 0):
		plt.plot(arr_v[:, 0], arr_v[:, 1], linestyle='', marker='o', markersize=1.0, color=l_color[i])
	
	iterations = 100

	min_amount_cluster = 2
	max_amount_cluster = 7

	arr_arr_y_s_mean, arr_best_cluster_amount  = utils_cluster.find_best_fitting_cluster_amount_multiprocessing(
		max_try_nr=8,
		arr_point=arr_point_unique,
		min_amount_cluster=min_amount_cluster,
		max_amount_cluster=max_amount_cluster,
		iterations=iterations,
		amount_proc=mp.cpu_count(),
	)

	print(f"arr_best_cluster_amount: {arr_best_cluster_amount}")

	plt.figure()

	plt.title("Cluster silhouette median values")

	plt.xlabel("Cluster amount")
	plt.ylabel("Silhouette value")

	arr_x_cluster_amount = np.arange(min_amount_cluster, max_amount_cluster+1)
	arr_y_silhoutte_median = np.median(arr_arr_y_s_mean, axis=0)
	plt.plot(arr_x_cluster_amount, arr_y_silhoutte_median, linestyle='', marker='o', label=f"Silhouette median per cluster amount")

	plt.legend()


	cluster_amount_best = arr_best_cluster_amount[0]

	calc_cluster_data = utils_cluster.calculate_clusters(
		cluster_amount=cluster_amount_best,
		arr_point=arr_point,
		iterations=iterations,
	)


	# this part can be used for any a n-th dimension vector too
	arr_true = arr_cluster_mean_vec
	arr_calc = calc_cluster_data.arr_cluster_mean_vec
	arr_diff = arr_true.reshape((1, arr_true.shape[0], arr_true.shape[1])) - arr_calc.reshape((arr_calc.shape[0], 1, arr_calc.shape[1]))
	arr_argsort = np.argsort(np.sqrt(np.sum(arr_diff**2, axis=2)), axis=1)
	arr_best_cluster_idx = arr_argsort[:, 0]
	if np.unique(arr_best_cluster_idx).shape[0] != arr_best_cluster_idx.shape[0]:
		# need more steps for finding the best fitting cluster mean vec!
		assert False and "Not implemented yet!"


	# find the calculated cluster points which are in the true generated clusters and which are not!
	df_cluster_point_true = pd.DataFrame(data={
			'x': np.hstack([v[:, 0] for v in l_v]),
			'y': np.hstack([v[:, 1] for v in l_v]),
			'cluster_nr': np.hstack([np.zeros((v.shape[0], ), dtype=np.int32)+i for i, v in enumerate(l_v, 0)]),
		}, columns=['x', 'y', 'cluster_nr'], dtype=object)

	df_cluster_point_calc = pd.DataFrame(data={
			'x': np.hstack([v[:, 0] for v in calc_cluster_data.l_cluster_points_correspond]),
			'y': np.hstack([v[:, 1] for v in calc_cluster_data.l_cluster_points_correspond]),
			'cluster_nr': np.hstack([np.zeros((v.shape[0], ), dtype=np.int32)+arr_best_cluster_idx[i] for i, v in enumerate(calc_cluster_data.l_cluster_points_correspond, 0)]),
		}, columns=['x', 'y', 'cluster_nr'], dtype=object)

	df_cluster_point_merge = pd.merge(df_cluster_point_true, df_cluster_point_calc, on=['x', 'y'], suffixes=('_true', '_calc'))
	df_cluster_point_merge.sort_values(by=['cluster_nr_calc', 'x', 'y'], inplace=True)
	df_cluster_point_merge.reset_index(drop=True, inplace=True)

	plt.figure()

	plt.title("Differences of true cluster mean points and calc cluster mean points + cluster points")

	color_edge_good = '#00FF00'
	color_edge_bad = '#FF0000'
	l_p = []
	arr_unique_cluster_nr = np.unique(df_cluster_point_merge['cluster_nr_calc'].values)
	arr_correct_guess = np.zeros((arr_unique_cluster_nr.shape[0], 2), dtype=np.uint64)
	for cluster_nr in arr_unique_cluster_nr:
		color = l_color[cluster_nr]
		df_part = df_cluster_point_merge.loc[df_cluster_point_merge['cluster_nr_calc'].values == cluster_nr]

		arr_is_good = df_part['cluster_nr_true'].values == df_part['cluster_nr_calc']

		arr_x_good = df_part['x'].values[arr_is_good]
		arr_y_good = df_part['y'].values[arr_is_good]

		arr_x_bad = df_part['x'].values[~arr_is_good]
		arr_y_bad = df_part['y'].values[~arr_is_good]

		arr_correct_guess[cluster_nr] = [np.sum(arr_is_good), np.sum(~arr_is_good)]

		l_p.append(plt.plot(arr_x_good, arr_y_good, linestyle='', marker='o', markersize=3.0, markerfacecolor=color, markeredgecolor=color_edge_good, label=f'cluster_nr_{cluster_nr:01}_good')[0])
		l_p.append(plt.plot(arr_x_bad, arr_y_bad, linestyle='', marker='o', markersize=3.0, markerfacecolor=color, markeredgecolor=color_edge_bad, label=f'cluster_nr_{cluster_nr:01}_bad')[0])

	# plt.plot(arr_point[:, 0], arr_point[:, 1], linestyle='', marker='o', markersize=0.5)
	p_0 = plt.plot(
		calc_cluster_data.arr_cluster_mean_vec[:, 0],
		calc_cluster_data.arr_cluster_mean_vec[:, 1],
		linestyle='', marker='o', markersize=6.0, markerfacecolor='none', markeredgecolor='#FF0000', label='true_mean_vec',
	)[0]
	p_1 = plt.plot(
		arr_cluster_mean_vec[:, 0],
		arr_cluster_mean_vec[:, 1],
		linestyle='', marker='o', markersize=4.0, color='#0000FF', label='calc_mean_vec',
	)[0]

	plt.legend(handles=l_p+[p_0, p_1])


	print(f"arr_correct_guess:\n{arr_correct_guess}")

	plt.show()
