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

sys.path.append("../")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj
import global_object_getter_setter
import utils

from utils_cluster import calculate_cluster_points_and_silouhette

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

	arr_point = np.vstack(l_v)

	plt.figure()

	plt.title("An example of the cluster points in 2d")
	plt.plot(arr_point[:, 0], arr_point[:, 1], linestyle='', marker='o')

	plt.show(block=True)
	
	iterations = 100

	l_try_nr_arr_x_cluster_amount_arr_y_s_mean = []
	
	for try_nr in range(1, 8+1):
		l_d_data = calculate_cluster_points_and_silouhette(
			arr_point=arr_point,
			min_amount_cluster=2,
			max_amount_cluster=9,
			iterations=iterations,
		)

		l_cluster_amount_s_mean = [(d['cluster_amount'], d['calc_silouhette_data'].val_s_mean) for d in l_d_data]

		arr_x_cluster_amount, arr_y_s_mean = [np.array(list(l)) for l in zip(*l_cluster_amount_s_mean)]
		l_try_nr_arr_x_cluster_amount_arr_y_s_mean.append((try_nr, arr_x_cluster_amount, arr_y_s_mean))

	arr_arr_y_s_mean = np.vstack([arr_y_s_mean for _, _, arr_y_s_mean in l_try_nr_arr_x_cluster_amount_arr_y_s_mean])

	arr_arr_y_s_mean_argsort_inv = np.argsort(arr_arr_y_s_mean, axis=1)
	arr_arr_y_s_mean_argsort = arr_arr_y_s_mean_argsort_inv.copy()
	arr_arange = np.arange(0, arr_arr_y_s_mean.shape[1])
	for i in range(0, arr_arr_y_s_mean.shape[0]):
		arr_arr_y_s_mean_argsort[i, arr_arr_y_s_mean_argsort_inv[i]] = arr_arange

	arr_x_cluster_amount = l_try_nr_arr_x_cluster_amount_arr_y_s_mean[0][1]
	arr_best_cluster_amount = arr_x_cluster_amount[np.argsort(np.sum(arr_arr_y_s_mean_argsort, axis=0))[::-1]]

	print(f"arr_best_cluster_amount: {arr_best_cluster_amount}")

	plt.figure()

	plt.title("l_cluster_amount_s_mean")

	l_p = []
	for try_nr, arr_x_cluster_amount, arr_y_s_mean in l_try_nr_arr_x_cluster_amount_arr_y_s_mean:
		p = plt.plot(arr_x_cluster_amount, arr_y_s_mean, linestyle='-', marker='o', label=f"try_nr: {try_nr}")[0]
		l_p.append(p)

	plt.legend()

	plt.show(block=True)
