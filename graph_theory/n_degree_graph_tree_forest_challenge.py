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

import zipfile
from zipfile import ZipFile

from typing import List, Set, Dict, Tuple, Optional, Any

sys.path.append("../")
from utils_multiprocessing_manager import MultiprocessingManager
from global_object_getter_setter import save_object, load_object, do_object_exist, delete_object

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"
HOME_DIR = os.path.expanduser("~") + "/"
TEMP_DIR = MemoryTempfile().gettempdir() + "/"

import utils_numpy as unp

if __name__ == '__main__':
    print("Hello World!")

    n = 10
    degree = 3

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

    print('d_node_to_node:')
    pprint(d_node_to_node)

    assert all([len(s) == degree for s in d_node_to_node.values()])

    # create the graph dot file!

    with open('graph.dot', 'w') as f:
        f.write('graph graphname {\n')

        f.write('  overlap = false;\n')
        for node in l_node:
            f.write('  x{}[label="{}"];\n'.format(node, node))

        for node1, node2 in l_used_edges:
            f.write("  x{} -- x{};\n".format(node1, node2))

        f.write('}\n')

     
