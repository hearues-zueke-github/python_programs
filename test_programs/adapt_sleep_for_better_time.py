#! /usr/bin/python3

# -*- coding: utf-8 -*-

import time
import datetime
import os
import sys

from copy import deepcopy

sys.path.append("../combinatorics/")
import different_combinations as combinations
sys.path.append("../math_numbers/")
from prime_numbers_fun import get_primes

# from time import time
from functools import reduce

from PIL import Image

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

sys.path.append("../")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

if __name__ == '__main__':
	print('Hello World!')

	dt_prev = datetime.datetime.now()

	sleep_seconds_should = 0.1
	sleep_seconds = sleep_seconds_should-0.00038123
	cummulative_error_seconds = 0
	time.sleep(sleep_seconds)
	lst_diff = []
	for i in range(0, 1000):
	# while True:
		dt_now = datetime.datetime.now()
		print("sleep_seconds {}".format(sleep_seconds))
		print("dt_prev {}".format(dt_prev))
		print("dt_now {}".format(dt_now))
		diff = (dt_now-dt_prev).total_seconds()
		print("diff {}".format(diff))
		lst_diff.append(diff)
		# if cummulative_error_seconds>0.01:
		# # if diff>sleep_seconds_should:
		# 	sleep_seconds *= 0.999
		# elif cummulative_error_seconds<-0.01:
		# 	sleep_seconds *= 1.001
		cummulative_error_seconds += diff-sleep_seconds_should
		print("cummulative_error_seconds {}".format(cummulative_error_seconds))
		time.sleep(sleep_seconds)
		dt_prev = dt_now

	print("np.mean(lst_diff) {}".format(np.mean(lst_diff)))
