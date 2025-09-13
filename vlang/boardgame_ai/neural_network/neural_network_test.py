#! /usr/bin/python3.13

import numpy as np
import pandas as pd

from arraybyte_serialization.cross_lang_serialization import CrossLangSerialization

if __name__ == "__main__":
	file_path = '/tmp/nn_v_data.arrhex'

	cross_lang_serialization = CrossLangSerialization()

	cross_lang_serialization.load_data_from_file(file_path=file_path)
	cross_lang_serialization.print_object_names_only()

	arr_layer = cross_lang_serialization.d_str_to_arr_i32['arr_layer']

	d_str_to_arr_f64 = cross_lang_serialization.d_str_to_arr_f64
	
	arr_x = d_str_to_arr_f64['arr_x_one_vector'].reshape((-1, arr_layer[0]))
	arr_y = d_str_to_arr_f64['arr_y_one_vector'].reshape((-1, arr_layer[-1]))

	arr_b = [
		d_str_to_arr_f64['b0'],
		d_str_to_arr_f64['b1'],
		d_str_to_arr_f64['b2'],
	]
	arr_w = [
		d_str_to_arr_f64['w0'].reshape((arr_layer[1], arr_layer[0])),
		d_str_to_arr_f64['w1'].reshape((arr_layer[2], arr_layer[1])),
		d_str_to_arr_f64['w2'].reshape((arr_layer[3], arr_layer[2])),
	]

	print(f"arr_x[0]: {arr_x[0]}")
	print(f"arr_y[0]: {arr_y[0]}")

	a0 = arr_x[0]
	a1 = np.dot(arr_w[0], a0) + arr_b[0]
	a2 = np.dot(arr_w[1], a1) + arr_b[1]
	a3 = np.dot(arr_w[2], a2) + arr_b[2]

	print(f"a0: {a0}")
	print(f"a1: {a1}")
	print(f"a2: {a2}")
	print(f"a3: {a3}")

	def leaked_relu(x: np.float64):
		if x < 0:
			return x * 0.01
		return x

	vfunc_leaked_relu = np.vectorize(leaked_relu)

	def f(x, arr_b, arr_w):
		a = x.copy()
		for b, w in zip(arr_b, arr_w):
			a = vfunc_leaked_relu(np.dot(w, a) + b)
		return a

	arr_y_calc = np.array([f(x=x, arr_b=arr_b, arr_w=arr_w) for x in arr_x])
	assert np.all(np.abs(arr_y_calc - arr_y) < 10**-14)
