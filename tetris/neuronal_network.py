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

    def f_diff(self, l_bw, l_bwd, eta):
        return map(lambda i, a, b: a-b*i**1.5*eta, zip(np.arange(len(l_bw), 0, -1.), l_bw, l_bwd))
    def f_deriv_prev_bigger(self, ps, ds):
        return np.sum(map(lambda p, d: np.sum(np.abs(p)>np.abs(d)), zip(ps, ds)))
    def f_deriv_prev_bigger_per_layer(self, ps, ds):
        return map(lambda p, d: np.sum(np.abs(p)>np.abs(d)), zip(ps, ds))

    def get_random_bws(self, l_node_amount):
        l_bw = np.array([0]*(len(l_node_amount) - 1), dtype=object)
        l_bw[:] = [np.random.uniform(-1./np.sqrt(n), 1./np.sqrt(n), (m+1, n)).astype(np.float64) for m, n in zip(l_node_amount[:-1], l_node_amount[1:])]
        return l_bw
    
    def __init__(self, ):
        self.dpi_quality = 500

        self.bws_1 = None
        self.bws_2 = None
        self.bwsd_1 = None
        self.bwsd_2 = None

        self.prev2_diff_1 = None
        self.prev_diff_1 = None
        self.diff_1 = None

        self.l_bw = None
        self.l_bw_fixed = []
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

    def init_l_bw(self, l_node_amount):
        self.l_node_amount = list(l_node_amount)
        self.nl_str = "_".join(list(map(str, self.l_node_amount)))
        self.l_bw = self.get_random_bws(self.l_node_amount)

    def calc_feed_forward(self, X):
    # def calc_feed_forward(self, X, l_bw):
        ones = np.ones((X.shape[0], 1))
        Y = X
        amount_bw = len(self.l_bw) - 1
        for i, bw in enumerate(self.l_bw[:-1], 0):
            Y = self.f_hidden(np.hstack((ones, Y)).dot(bw), amount_bw - i)
        Y = self.f_sig(np.hstack((ones, Y)).dot(self.l_bw[-1]))
        return Y

    def calc_feed_forward_hidden_only(self, X, l_bw):
        ones = np.ones((X.shape[0], 1))
        Y = X
        for bw in l_bw:
            Y = self.f_hidden(np.hstack((ones, Y)).dot(bw))
        return Y

    def calc_backprop(self, X, T, l_bw):
        Zs = []
        As = [X]
        arr_ones = np.ones((X.shape[0], 1))
        A = X
        amount_bw = len(l_bw) - 1
        for i, bw in enumerate(l_bw[:-1], 0):
        # for i, bw in zip(range(len(l_bw), 0, -1), l_bw[:-1]):
            Z = np.hstack((arr_ones, A)).dot(bw)#*self.hidden_multiplier**i;
            Zs.append(Z)
            A = self.f_hidden(Z, amount_bw - i)
            As.append(A)
        Z = np.hstack((arr_ones, A)).dot(l_bw[-1])
        Zs.append(Z)
        A = self.f_sig(Z)
        # Y_exp = np.exp(Z)
        # A = Y_exp/np.sum(Y_exp, axis=1).reshape((Y_exp.shape[0], 1))
        # A = A / np.sum(A, axis=1).reshape((-1, 1))
        As.append(A)

        l_bwd = np.array([np.zeros(bw.shape) for bw in l_bw], dtype=object)

        d = (A-T)
        l_bwd[-1][:] = np.hstack((arr_ones, As[-2])).T.dot(d)
        # l_bwd[-1][:] = np.hstack((arr_ones, As[-2])).T.dot(d * self.fd_sig(As[-1]))

        length = len(l_bw)
        for i in range(2, length + 1):
            d = d.dot(l_bw[-i+1][1:].T)*self.fd_hidden(Zs[-i], i-2)#*self.hidden_multiplier**i

            # d_abs = np.abs(d)
            # max_num = np.max(d_abs)
            # min_num = np.min(d_abs)
            # if max_num < 1 and max_num > 0.000001:
            #     d /= max_num
            # if min_num > 0.0001:
            #     d *= d_abs**0.5
            # else:
            #     d *= d_abs**0.1

            # if i == length:
            l_bwd[-i][:] = np.hstack((arr_ones, As[-i-1])).T.dot(d)

        return np.array(l_bwd)

    def calc_numerical_gradient(self, X, T, l_bw):
        l_bwd = deepcopy(l_bw) # [np.zeros_like(bw) for bw in l_bw]

        epsilon = 0.00001
        for i, (bw, bwd) in enumerate(zip(l_bw, l_bwd)):
            print("calc numerical layer nr. {}".format(i+1))
            for y in range(0, bw.shape[0]):
                print("y: {}".format(y))
                for x in range(0, bw.shape[1]):
                    print("x: {}".format(x))

                    bw[y, x] += epsilon
                    fr = self.calc_cost(self.calc_feed_forward(X, l_bw), T)
                    bw[y, x] -= epsilon*2.
                    fl = self.calc_cost(self.calc_feed_forward(X, l_bw), T)
                    bw[y, x] += epsilon
                    bwd[y, x] = (fr - fl) / (2.*epsilon)

        return l_bwd

    def gradient_check(self, X, T):
        l_bw = self.l_bw

        bwsd_real = self.calc_backprop(X, T, l_bw)
        bwsd_numerical = self.calc_numerical_gradient(X, T, l_bw)

        print("X:\n{}".format(X))
        print("T:\n{}".format(T))
        for i, (bwd, bwsdi_num) in enumerate(zip(bwsd_real, bwsd_numerical)):
            print("i: {}, bwd:\n{}".format(i, bwd))
            print("i: {}, bwsdi_num:\n{}".format(i, bwsdi_num))
            # print("i: {}, bwd > 0 and bwsdi_num > 0:\n{}".format(i, np.logical_or(np.logical_and(bwd>0, bwsdi_num>0), np.logical_and(bwd<=0, bwsdi_num<=0))))
            # print("i: {}, bwsdi_num/bwd:\n{}".format(i, bwsdi_num/bwd))

        # sys.exit(0)
