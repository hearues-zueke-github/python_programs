#! /usr/bin/python3.10
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import pylab
import matplotlib.ticker as tkr

import dill
import gzip
import os
import select
import sys
import time

import multiprocessing as mp
import numpy as np

from colorama import Fore, Style
from copy import deepcopy
from dotmap import DotMap
from multiprocessing import Pipe, Process
from numpy.random import Generator, PCG64

class NeuralNetwork(Exception):
	def f_sig(self, X):
		return 1. / (1 + np.exp(-X))
	def fd_sig(self, X):
		return self.f_sig(X) * (1 - self.f_sig(X))
	def f_relu(self, x):
		return np.vectorize(lambda x: 0.2*x if x < 0. else x)(x)
	def fd_relu(self, x):
		return np.vectorize(lambda x: 0.2 if x < 0. else 1)(x)
	def f_tanh(self, x, pow_nr):
		return np.tanh(x*self.scalar_hidden_multiply**pow_nr)
	def fd_tanh(self, x, pow_nr):
		return (1 - np.tanh(x*0.1)**2)*self.scalar_hidden_multiply**pow_nr

	def f_rmse(self, Y, T):
		# return np.sqrt(np.mean(np.sum((Y-T)**2, axis=1)))
		return np.mean(np.sum((Y-T)**2, axis=1))
	def f_cecf(self, Y, T):
		return np.sum(np.nan_to_num(-T*np.log(Y)-(1-T)*np.log(1-Y)))
		# return np.sum(np.sum(np.vectorize(lambda y, t: -np.log(y) if t==1. else -np.log(1-y))(Y, T), axis=1))

	def f_prediction_regression_vector(self, Y, T):
		return np.sum(np.abs(Y-T) < 0.1, axis=1)==Y.shape[1]
	def f_prediction_regression(self, Y, T):
		return np.sum(self.f_prediction_regression_vector(Y, T))
	def f_missclass_regression_vector(self, Y, T):
		return np.sum(np.abs(Y-T) < 0.1, axis=1)!=Y.shape[1]
	def f_missclass_regression(self, Y, T):
		return np.sum(self.f_missclass_regression_vector(Y, T))

	def f_prediction_vector(self, Y, T):
		return np.sum(np.vectorize(lambda x: 0 if x < 0.5 else 1)(Y).astype(np.uint8)==T.astype(np.uint8), axis=1)==Y.shape[1]
	def f_prediction(self, Y, T):
		return np.sum(self.f_prediction_vector(Y, T))
	def f_missclass_vector(self, Y, T):
		return np.sum(np.vectorize(lambda x: 0 if x < 0.5 else 1)(Y).astype(np.uint8)==T.astype(np.uint8), axis=1)!=Y.shape[1]
	def f_missclass(self, Y, T):
		return np.sum(self.f_missclass_vector(Y, T))

	def f_prediction_onehot_vector(self, Y, T):
		return np.argmax(Y, axis=1)==np.argmax(T, axis=1)
	def f_prediction_onehot(self, Y, T):
		return np.sum(np.argmax(Y, axis=1)==np.argmax(T, axis=1))
	def f_missclass_onehot_vector(self, Y, T):
		return np.argmax(Y, axis=1)!=np.argmax(T, axis=1)
	def f_missclass_onehot(self, Y, T):
		return np.sum(np.argmax(Y, axis=1)!=np.argmax(T, axis=1))

	def f_diff(self, arr_bw, arr_bwd, eta):
		return map(lambda i, a, b: a-b*i**1.5*eta, zip(np.arange(len(arr_bw), 0, -1.), arr_bw, arr_bwd))
	def f_deriv_prev_bigger(self, ps, ds):
		return np.sum(map(lambda p, d: np.sum(np.abs(p)>np.abs(d)), zip(ps, ds)))
	def f_deriv_prev_bigger_per_layer(self, ps, ds):
		return map(lambda p, d: np.sum(np.abs(p)>np.abs(d)), zip(ps, ds))

	def get_random_bws(self, l_node_amount):
		arr_bw = np.array([0]*(len(l_node_amount) - 1), dtype=object)
		arr_bw[:] = [self.rnd.uniform(-1./np.sqrt(n), 1./np.sqrt(n), (m+1, n)).astype(np.float64) for m, n in zip(l_node_amount[:-1], l_node_amount[1:])]
		return arr_bw
	
	def __init__(self, l_seed=[]):
		self.dpi_quality = 500

		self.bws_1 = None
		self.bws_2 = None
		self.bwsd_1 = None
		self.bwsd_2 = None

		self.prev2_diff_1 = None
		self.prev_diff_1 = None
		self.diff_1 = None

		self.arr_bw = None
		self.arr_bw_fixed = []
		self.l_node_amount = None
		self.nl_str = None
		self.nl_whole = []
		self.etas = []
		self.costs_train = []
		self.costs_valid = []
		self.missclasses_percent_train = []
		self.missclasses_percent_valid = []
		self.costs_missclasses_percent_test = []

		self.scalar_hidden_multiply = np.float64(0.3)

		self.calc_cost = None
		self.calc_missclass = None
		self.calc_missclass_vector = None

		self.bws_best_early_stopping = 0.
		self.epoch_best_early_stopping = 0
		self.cost_valid_best_early_stopping = 10.
		self.missclass_percent_valid_best_early_stopping = 100.

		self.with_momentum_1_degree = False
		self.with_confusion_matrix = False

		self.trained_depth = 0
		self.trained_depth_prev = 0

		self.seed_main = np.array(l_seed, dtype=np.uint32)
		self.rnd = Generator(bit_generator=PCG64(seed=self.seed_main))

		self.set_hidden_function("tanh")

	def set_hidden_function(self, func_str):
		if func_str == "sig":
			self.f_hidden = self.f_sig
			self.fd_hidden = self.fd_sig
		elif func_str == "relu":
			self.f_hidden = self.f_relu
			self.fd_hidden = self.fd_relu
		elif func_str == "tanh":
			self.f_hidden = self.f_tanh
			self.fd_hidden = self.fd_tanh

	def init_arr_bw(self, l_node_amount):
		self.l_node_amount = list(l_node_amount)
		self.nl_str = "_".join(list(map(str, self.l_node_amount)))
		self.arr_bw = self.get_random_bws(self.l_node_amount)

	# mix_rate: is the percentage of mixing of the values from arr_bw_other into self.arr_bw. some of the values are replaced with the other bw!
	def mix_arr_bw(self, arr_bw_other, mix_rate=0.1):
		for bw, bw_other in zip(self.arr_bw, arr_bw_other):
			arr_is_used = (self.rnd.random(bw.shape) <= mix_rate)
			if np.any(arr_is_used):
				t_idx = np.where(arr_is_used)
				bw[t_idx] = bw_other[t_idx]

	# mix_rate: how many of the 
	def mutate_arr_bw(self, random_rate=0.1, change_rate=0.01): # by adding random values, the weights are beeing changed a bit
		for bw in self.arr_bw:
			arr_is_used = (self.rnd.random(bw.shape) <= random_rate)
			if np.any(arr_is_used):
				t_idx = np.where(arr_is_used)
				bw[t_idx] += (self.rnd.random((np.sum(arr_is_used), ))-0.5)*2*change_rate # TODO: should be changed for each weight matrice indiviual!

	def crossover_and_mutate(self, arr_bw_1, arr_bw_2, mix_rate, random_rate, change_factor):
		for bw, bw1, bw2 in zip(self.arr_bw, arr_bw_1, arr_bw_2):
			arr_is_used = (self.rnd.random(bw.shape) <= mix_rate)
			if np.any(arr_is_used):
				t_idx_1 = np.where(arr_is_used)
				t_idx_2 = np.where(~arr_is_used)
				bw[t_idx_1] = bw1[t_idx_1]
				bw[t_idx_2] = bw2[t_idx_2]
			else:
				bw[:] = bw1

			arr_is_used = (self.rnd.random(bw.shape) <= random_rate)
			if np.any(arr_is_used):
				t_idx = np.where(arr_is_used)
				bw[t_idx] += (self.rnd.random((np.sum(arr_is_used), ))-0.5)*2*change_factor

			bw_abs = np.abs(bw)
			val_max = np.max(bw_abs)
			if val_max > 10.:
				bw[:] = bw * 10 / val_max

	def calc_feed_forward(self, X):
	# def calc_feed_forward(self, X, arr_bw):
		ones = np.ones((X.shape[0], 1))
		Y = X
		amount_bw = len(self.arr_bw) - 1
		for i, bw in enumerate(self.arr_bw[:-1], 0):
			Y = self.f_hidden(np.hstack((ones, Y)).dot(bw), amount_bw - i)
		Y = self.f_sig(np.hstack((ones, Y)).dot(self.arr_bw[-1]))
		return Y

	def calc_feed_forward_hidden_only(self, X, arr_bw):
		ones = np.ones((X.shape[0], 1))
		Y = X
		for bw in arr_bw:
			Y = self.f_hidden(np.hstack((ones, Y)).dot(bw))
		return Y

	def calc_backprop(self, X, T, arr_bw):
		Zs = []
		As = [X]
		arr_ones = np.ones((X.shape[0], 1))
		A = X
		amount_bw = len(arr_bw) - 1
		for i, bw in enumerate(arr_bw[:-1], 0):
		# for i, bw in zip(range(len(arr_bw), 0, -1), arr_bw[:-1]):
			Z = np.hstack((arr_ones, A)).dot(bw)#*self.hidden_multiplier**i;
			Zs.append(Z)
			A = self.f_hidden(Z, amount_bw - i)
			As.append(A)
		Z = np.hstack((arr_ones, A)).dot(arr_bw[-1])
		Zs.append(Z)
		A = self.f_sig(Z)
		# Y_exp = np.exp(Z)
		# A = Y_exp/np.sum(Y_exp, axis=1).reshape((Y_exp.shape[0], 1))
		# A = A / np.sum(A, axis=1).reshape((-1, 1))
		As.append(A)

		arr_bwd = np.array([np.zeros(bw.shape) for bw in arr_bw], dtype=object)

		d = (A-T)
		arr_bwd[-1][:] = np.hstack((arr_ones, As[-2])).T.dot(d)
		# arr_bwd[-1][:] = np.hstack((arr_ones, As[-2])).T.dot(d * self.fd_sig(As[-1]))

		length = len(arr_bw)
		for i in range(2, length + 1):
			d = d.dot(arr_bw[-i+1][1:].T)*self.fd_hidden(Zs[-i], i-2)#*self.hidden_multiplier**i

			# d_abs = np.abs(d)
			# max_num = np.max(d_abs)
			# min_num = np.min(d_abs)
			# if max_num < 1 and max_num > 0.000001:
			#	 d /= max_num
			# if min_num > 0.0001:
			#	 d *= d_abs**0.5
			# else:
			# 	d *= d_abs**0.1

			# if i == length:
			arr_bwd[-i][:] = np.hstack((arr_ones, As[-i-1])).T.dot(d)

		return np.array(arr_bwd)

	def calc_numerical_gradient(self, X, T, arr_bw):
		arr_bwd = deepcopy(arr_bw) # [np.zeros_like(bw) for bw in arr_bw]

		epsilon = 0.00001
		for i, (bw, bwd) in enumerate(zip(arr_bw, arr_bwd)):
			print("calc numerical layer nr. {}".format(i+1))
			for y in range(0, bw.shape[0]):
				print("y: {}".format(y))
				for x in range(0, bw.shape[1]):
					print("x: {}".format(x))

					bw[y, x] += epsilon
					fr = self.calc_cost(self.calc_feed_forward(X, arr_bw), T)
					bw[y, x] -= epsilon*2.
					fl = self.calc_cost(self.calc_feed_forward(X, arr_bw), T)
					bw[y, x] += epsilon
					bwd[y, x] = (fr - fl) / (2.*epsilon)

		return arr_bwd

	def gradient_check(self, X, T):
		arr_bw = self.arr_bw

		bwsd_real = self.calc_backprop(X, T, arr_bw)
		bwsd_numerical = self.calc_numerical_gradient(X, T, arr_bw)

		print("X:\n{}".format(X))
		print("T:\n{}".format(T))
		for i, (bwd, bwsdi_num) in enumerate(zip(bwsd_real, bwsd_numerical)):
			print("i: {}, bwd:\n{}".format(i, bwd))
			print("i: {}, bwsdi_num:\n{}".format(i, bwsdi_num))
			# print("i: {}, bwd > 0 and bwsdi_num > 0:\n{}".format(i, np.logical_or(np.logical_and(bwd>0, bwsdi_num>0), np.logical_and(bwd<=0, bwsdi_num<=0))))
			# print("i: {}, bwsdi_num/bwd:\n{}".format(i, bwsdi_num/bwd))

		# sys.exit(0)
