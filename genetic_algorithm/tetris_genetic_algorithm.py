#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import functools
import os
import sys
import time

from copy import deepcopy

import numpy as np

from PIL import Image, ImageTk

import matplotlib.pyplot as plt

from memory_tempfile import MemoryTempfile
tempfile = MemoryTempfile()

from GeneticAlgorithmTetris import GeneticAlgorithmTetris

from utils_tetris import load_Xs_Ts_full
import utils_objects

import platform
print("platform.system(): {}".format(platform.system()))

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

def plot_graph(d_using_params, ga, l_cecf_sum_all=None):
    if l_cecf_sum_all is None:
        d = utils_objects.load_dict_object('xs_Ys')
        xs = d['xs']
        Ys = d['Ys']
    else:
        Ys = np.array(l_cecf_sum_all).T
        xs = np.arange(0, len(l_cecf_sum_all))
    
    color_str_template = '#'+'{:02X}'*3

    plt.figure()

    for ys in Ys:
        plt.plot(xs, ys, 'o', markersize=2., color=color_str_template.format(*np.random.randint(32, 198, (3, )).tolist()))

    plt.title('Rows: {}, Cols: {}, Popul. size: {}, Epochs: {}\n'.format(d_using_params['rows'], d_using_params['cols'], d_using_params['population_size'], d_using_params['epochs'])+
        'Hidden nodes: {}, Using Pieces: {}, Block Cells: {}'.format(ga.hidden_nodes, d_using_params['using_pieces'], d_using_params['block_cells']))

    plt.xlabel('Epochs')
    plt.ylabel("sum of {} cecf's".format(d_using_params['amount_pieces']))

    plt.show(block=False)


if __name__ == "__main__":
    # plot_graph()
     # sys.exit(0)

    rows = 20
    cols = 5
    proc_num = 6
    iterations_multi_processing = 100
    using_pieces = 4
    block_cells = [4]

    population_size = 5
    epochs = 400
    
    hidden_nodes = [300, 200, 100]

    # TODO: create a database, where the best saved neural networks are saved!!!

    Xs_full, Ts_full, d_basic_data_info = load_Xs_Ts_full(
        rows=rows,
        cols=cols,
        proc_num=proc_num,
        iterations_multi_processing=iterations_multi_processing,
        using_pieces=using_pieces,
        block_cells=block_cells,
    )

    print("Xs_full[0].shape: {}".format(Xs_full[0].shape))

    l_shapes_1 = [X.shape for X in Xs_full]
    print("l_shapes_1: {}".format(l_shapes_1))

    for i in range(0, len(Xs_full)):
        X = Xs_full[i].copy()
        T = Ts_full[i]

        X[X==-1.] = 0
        X = X.astype(np.uint8)

        rows, cols = X.shape
        X = np.hstack((X, np.zeros((rows, 8-cols%8 if cols%8!=0 else 0), dtype=np.uint8)))
        cols_new = X.shape[1]
        X = X.reshape((rows, cols_new//8, -1))
        X_row_sum = np.sum(X*(2**np.arange(0, 8)), axis=-1, dtype=np.uint8)
        X_row_sum_dtype = X_row_sum.reshape((-1, )).view(dtype=[(f'x{i}', 'u1') for i in range(0, X_row_sum.shape[1])])

        u, c = np.unique(X_row_sum_dtype, return_counts=True)
        u_c_gt_1 = u[c>1]

        l_idxs_not_remove = []
        for v in u_c_gt_1:
            idxs = np.where(X_row_sum_dtype==v)[0]
            l_idxs_not_remove.append(idxs[0])
            ts = T[idxs]
        
        idxs = np.isin(X_row_sum_dtype, u_c_gt_1)
        idxs[l_idxs_not_remove] = False
        idxs = ~idxs

        assert np.sum(np.isin(X_row_sum_dtype[idxs], u_c_gt_1))==u_c_gt_1.shape[0]

        Xs_full[i] = Xs_full[i][idxs]
        Ts_full[i] = Ts_full[i][idxs]

    l_len_Xs_full = [len(X) for X in Xs_full]
    print("new l_len_Xs_full: {}".format(l_len_Xs_full))
    print("Xs_full[0].shape: {}".format(Xs_full[0].shape))

    l_shapes_ = [X.shape for X in Xs_full]
    print("l_shapes_: {}".format(l_shapes_))

    # sys.exit(0)

    ga = GeneticAlgorithmTetris(
        d_basic_data_info=d_basic_data_info,
        population_size=population_size,
        Xs=Xs_full,
        Ts=Ts_full,
        is_single_instance_only=False,
    )
    ga.simple_genetic_algorithm_training(epochs=epochs)

    l_cecf_all, l_cecf_sum_all = ga.l_cecf_all, ga.l_cecf_sum_all

    d_using_params = {
        'rows': rows,
        'cols': cols,
        'proc_num': proc_num,
        'iterations_multi_processing': iterations_multi_processing,
        'using_pieces': using_pieces,
        'block_cells': block_cells,
        'population_size': population_size,
        'epochs': epochs,
        'amount_pieces': d_basic_data_info['amount_pieces'],
    }

    plot_graph(d_using_params, ga, l_cecf_sum_all=l_cecf_sum_all)
