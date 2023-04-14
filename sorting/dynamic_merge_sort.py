#! /usr/bin/python3.10

# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

if __name__ == "__main__":
	n = 20
	arr = np.random.randint(0, 30, (n, ))
	print("arr: {}".format(arr))

	l_check_lt = [1]
	for i in range(0, len(arr) - 1):
		if arr[i] < arr[i + 1]:
			l_check_lt.append(1)
		else:
			l_check_lt.append(0)
	print(f"l_check_lt: {l_check_lt}")

	# TODO: create the first part for the dynamic merge sort
	l_idx = [0]
	v_prev = arr[0]
	idx_prev = 0
	for idx_next in range(1, len(arr)):
		v_next = arr[i]

		if v_prev >= v_next: # !(<) == (>=)
			pass

		v_prev = v_next
