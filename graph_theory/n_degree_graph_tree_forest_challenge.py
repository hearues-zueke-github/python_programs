#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.8.6 -i

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import dill
import gzip
import os
import io
import re
import pdb
import sys
import string
import traceback
import inspect
from itertools import chain

# Needed for excel tables
import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.utils.cell import column_index_from_string
from openpyxl.styles import Alignment, borders, Font

from subprocess import check_call

# Needed for DataFrame (databases etc.)
import numpy as np
import pandas as pd
import multiprocessing as mp

# For ploting stuff
import matplotlib.lines as mlines
import matplotlib.text as mtext
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from dotmap import DotMap
from memory_tempfile import MemoryTempfile
from collections import defaultdict
from pprint import pprint

from datetime import datetime, timedelta, time
from dateutil.relativedelta import relativedelta
from time import sleep

import zipfile
from zipfile import ZipFile

from typing import List, Set, Dict, Tuple, Optional, Any

sys.path.append("../")
from utils_multiprocessing_manager import MultiprocessingManager
from global_object_getter_setter import save_object, load_object, do_object_exist, delete_object

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"
HOME_DIR = os.path.expanduser("~") + "/"
TEMP_DIR = MemoryTempfile().gettempdir() + "/"

import utils
import utils_numpy as unp

if __name__ == '__main__':
    def create_new_d_node_to_node(n, degree):
        nr_tries = 0
        while True:
            nr_tries += 1
            print("nr_tries: {}".format(nr_tries))
            try:
                l_node = list(range(0, n))
                l_edges = [(node1, node2) for i1, node1 in enumerate(l_node, 0) for i2, node2 in enumerate(l_node[i1+1:], i1+1)]
                arr_edges = np.empty((len(l_edges), ), dtype=object)
                arr_edges[:] = l_edges

                arr_edges_mix = arr_edges[np.random.permutation(np.arange(0, len(l_edges)))]

                d_node_to_node = {node: set() for node in l_node}

                l_used_edges = []
                for node1, node2 in arr_edges_mix:
                    if len(d_node_to_node[node1]) >= degree or len(d_node_to_node[node2]) >= degree:
                        continue

                    d_node_to_node[node1].add(node2)
                    d_node_to_node[node2].add(node1)
                    l_used_edges.append((node1, node2))


                assert all([len(s) == degree for s in d_node_to_node.values()])

                node_current = l_node[0]
                s_available_nodes = set(l_node) - set([node_current])
                l_next_nodes = list(d_node_to_node[node_current])
                while l_next_nodes:
                    l_next_nodes_new = []
                    for node in l_next_nodes:
                        for node_next in d_node_to_node[node]:
                            if node_next in s_available_nodes:
                                s_available_nodes.remove(node_next)
                                for node_next_next in d_node_to_node[node_next]:
                                    if node_next_next in s_available_nodes:
                                        l_next_nodes_new.append(node_next_next)
                    l_next_nodes = l_next_nodes_new
                assert len(s_available_nodes) == 0
            except:
                continue
            else:
                break
        print('d_node_to_node:')
        pprint(d_node_to_node)

        return l_node, d_node_to_node, l_used_edges

    # create the graph dot file!
    def create_dot_graph(file_path, l_node, l_used_edges):
        with open(file_path, 'w') as f:
            f.write('graph graphname {\n')

            f.write('  overlap = false;\n')
            for node in l_node:
                f.write('  x{}[label="{}"];\n'.format(node, node))

            for node1, node2 in l_used_edges:
                f.write("  x{} -- x{};\n".format(node1, node2))

            f.write('}\n')

        check_call(['dot', '-Tpng', file_path, '-o', file_path.replace('.dot', '.png')])


    def get_cicles_node_cycles_amount(l_node, d_node_to_node):
        l_s_cicles = []
        d_s_cicles_len = {}
        for node_first in sorted(d_node_to_node.keys()):
            s_cicles = set()
            # node_first = l_node[0]
            def do_recursive_cicles(s_cicles, s_edges, l_node_seq, node_last):
                s_next_node = d_node_to_node[node_last]
                for node_next in s_next_node:
                    t = (node_last, node_next)
                    if t in s_edges:
                        continue
                    if len(l_node_seq) > 0 and l_node_seq[-1] == node_next:
                        continue
                    if node_next == node_first:
                        # t_seq = tuple(l_node_seq)
                        t_seq = tuple(l_node_seq + [node_last])
                        t_seq_rev = t_seq[:1] + t_seq[1:][::-1]
                        if t_seq in s_cicles or t_seq_rev in s_cicles:
                            continue

                        s_cicles.add(t_seq)
                        # print("l_node_seq: {}".format(l_node_seq))
                        # input('Enter...')
                        
                        # globals()['loc2'] = locals()
                        # sys.exit(-123)
                        
                        continue
                    s_edges_new = s_edges.copy()
                    s_edges_new.add(t)
                    do_recursive_cicles(s_cicles, s_edges_new, l_node_seq + [node_last], node_next)

            do_recursive_cicles(s_cicles, set(), [], node_first)
            print("node_first: {}, len(s_cicles): {}".format(node_first, len(s_cicles)))
            # print("s_cicles: {}".format(s_cicles))
            l_s_cicles.append(s_cicles)

            l_s_len = [len(t) for t in s_cicles]
            d = {length: [] for length in set(l_s_len)}
            for t in s_cicles:
                d[len(t)].append(t)
            d_s_cicles_len[node_first] = d

        d_node_cycles_amount = {node: {k: len(d[k]) for k in d.keys()} for node, d in d_s_cicles_len.items()}
        s_cicles_all = set().union(*l_s_cicles)

        l_tpl_node_cycles_amount = [tuple(sorted([(k, v) for k, v in d_node_cycles_amount[node].items()])) for node in l_node]
        d_node_cycles_amount_len = {t: [] for t in set(l_tpl_node_cycles_amount)}
        for node, tpl_node_cycles_amount in zip(l_node, l_tpl_node_cycles_amount):
            d_node_cycles_amount_len[tpl_node_cycles_amount].append(node)
        d_node_cycles_amount_len = {k: len(v) for k, v in d_node_cycles_amount_len.items()}

        l_s_cicle_len = [len(s_cicle) for s_cicle in l_s_cicles]
        d_cycle_nodes = {k: [] for k in set(l_s_cicle_len)}
        for node, s_cicle_len in zip(l_node, l_s_cicle_len):
            d_cycle_nodes[s_cicle_len].append(node)
        print("d_cycle_nodes: {}".format(d_cycle_nodes))

        d_cycle_nodes_amount = {k: len(v) for k, v in d_cycle_nodes.items()}
        print("d_cycle_nodes_amount: {}".format(d_cycle_nodes_amount))

        return d_node_cycles_amount_len, d_node_cycles_amount, d_cycle_nodes_amount
        # return s_cicles_all, d_node_cycles_amount_len, d_node_cycles_amount, d_cycle_nodes_amount

    
    def create_different_graphs(l_res_prev, n_graph):
        np.random.seed()

        l_res = []
        for i_graph in range(0, n_graph):
            print("i_graph: {}".format(i_graph))
            l_node, d_node_to_node, l_used_edges = create_new_d_node_to_node(n, degree)
            d_node_cycles_amount_len, d_node_cycles_amount, d_cycle_nodes_amount = get_cicles_node_cycles_amount(l_node, d_node_to_node)
            # s_cicles_all, d_node_cycles_amount_len, d_node_cycles_amount, d_cycle_nodes_amount = get_cicles_node_cycles_amount(l_node, d_node_to_node)

            does_exist = False
            for _, d_node_cycles_amount_len_ret, _, d_cycle_nodes_amount_ret, _, _ in l_res_prev+l_res:
                if d_cycle_nodes_amount == d_cycle_nodes_amount_ret and d_node_cycles_amount_len == d_node_cycles_amount_len_ret:
                    does_exist = True
                    break

            if not does_exist:
                l_res.append((d_node_to_node, d_node_cycles_amount_len, d_node_cycles_amount, d_cycle_nodes_amount, l_node, l_used_edges))

        return l_res

    n = 16
    degree = 3
    n_graph = 5


    mproc_mng = MultiprocessingManager(cpu_count=mp.cpu_count()-1)
    mproc_mng.define_new_func(name='func_create_different_graphs', func=create_different_graphs)

    l_res_prev = []
    for i in range(0, 5):
        l_res_all = mproc_mng.do_new_jobs(
            ['func_create_different_graphs']*mproc_mng.worker_amount,
            [(l_res_prev, n_graph)]*mproc_mng.worker_amount,
        )
        
        for l_res in l_res_all:
            does_exist = False
            for _, d_node_cycles_amount_len, _, d_cycle_nodes_amount, _, _ in l_res:
                for d_node_cycles_amount_len_ret, d_cycle_nodes_amount_ret in l_res_prev:
                    if d_cycle_nodes_amount == d_cycle_nodes_amount_ret and d_node_cycles_amount_len == d_node_cycles_amount_len_ret:
                        does_exist = True
                        break

                if not does_exist:
                    l_res_prev.append((d_node_cycles_amount_len, d_cycle_nodes_amount))

        print("i: {}, len(l_res_prev): {}".format(i, len(l_res_prev)))
        sleep(5)

    print("len(l_res_prev): {}".format(len(l_res_prev)))

    del mproc_mng
    
    sys.exit()

    for i, (_, _, _, _, _, l_node, l_used_edges) in enumerate(l_res, 0):
        dir_path = os.path.join(TEMP_DIR, os.path.join('n_degree_graph_tree', 'graphs_n_{}_degree_{}'.format(n, degree)))
        utils.mkdirs(dir_path)

        file_path = os.path.join(dir_path, 'n_{}_degree_{}_i_{:03}.dot'.format(n, degree, i))
        create_dot_graph(file_path, l_node, l_used_edges)

    # # l_node_new = l_node[1:] + l_node[:1]
    # l_node_new = [1, 2, 4, 3, 0, 5, 6, 7, 8, 9, 10, 11]
    # d_node_to_node_new = {l_node_new[k]: {l_node_new[node] for node in v} for k, v in d_node_to_node.items()}

    # s_cicles_all_new, d_node_cycles_amount_new, d_cycle_nodes_amount_new = get_cicles_node_cycles_amount(d_node_to_node_new)
