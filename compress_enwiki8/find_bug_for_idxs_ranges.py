#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import gzip
import os
import sys

from copy import deepcopy
from dotmap import DotMap

from collections import defaultdict

from os.path import expanduser
PATH_HOME = expanduser("~")+'/'
print("PATH_HOME: {}".format(PATH_HOME))

import numpy as np

sys.path.append("../")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

import utils_compress_enwik8

if __name__ == "__main__":
    #try reconstruct the fail by random!

    # amount of pattern/spaces idxs
    for k in range(0, 100000):
        print("k: {}".format(k))
        n1, n2 = np.random.randint(5, 31, (2, ))

        arr_space_1 = np.random.randint(0, 8, (n1, ))
        arr_pat_1 = np.random.randint(2, 7, (n1, ))

        arr_space_2 = np.random.randint(0, 8, (n2, ))
        arr_pat_2 = np.random.randint(2, 7, (n2, ))

        idxs_ranges_1 = list(map(tuple, np.cumsum(np.vstack((arr_space_1, arr_pat_1)).T.flatten()).reshape((-1, 2)).tolist()))
        idxs_ranges_2 = list(map(tuple, np.cumsum(np.vstack((arr_space_2, arr_pat_2)).T.flatten()).reshape((-1, 2)).tolist()))

        idxs_merged = utils_compress_enwik8.do_merge_idxs_ranges(idxs_ranges_1, idxs_ranges_2)

        print("idxs_ranges_1: {}".format(idxs_ranges_1))
        print("idxs_ranges_2: {}".format(idxs_ranges_2))
        print("idxs_merged: {}".format(idxs_merged))

        a, b = np.array(idxs_merged).T
        which_idxs = np.where(b[:-1]>a[1:])[0]
        print("which_idxs: {}".format(which_idxs))
        idxs1_found = [idxs_ranges_1.index(idxs_merged[i]) for i in which_idxs if idxs_merged[i] in idxs_ranges_1]+[idxs_ranges_1.index(idxs_merged[i]) for i in which_idxs+1 if idxs_merged[i] in idxs_ranges_1]
        idxs2_found = [idxs_ranges_2.index(idxs_merged[i]) for i in which_idxs+1 if idxs_merged[i] in idxs_ranges_2]+[idxs_ranges_2.index(idxs_merged[i]) for i in which_idxs if idxs_merged[i] in idxs_ranges_2]
        print("idxs1_found: {}".format(idxs1_found))
        print("idxs2_found: {}".format(idxs2_found))

        assert which_idxs.shape[0]==0

    sys.exit(0)
    # sys.exit("FAIL Nr. 2!!!")




    with gzip.open('obj.pkl.gz', 'rb') as f:
        dm = dill.load(f)

    idxs1_orig = dm.idxs_ranges_1
    idxs2 = dm.idxs_ranges_2

    '''
    which_idxs: [ 1462  5379 11441 11669]
    idxs1_found: [1446, 5320, 11311, 11533]
    idxs2_found: [59, 254, 555, 568]
    '''

    # idxs2 = idxs2[1200:]
    idxs1 = idxs1_orig[:1410]+idxs1_orig[1600:5200]+idxs1_orig[5500:10000]+idxs1_orig[11200:]
    # idxs1 = idxs1_orig
    print("len(idxs1): {}".format(len(idxs1)))

    idxs_merged = utils_compress_enwik8.do_merge_idxs_ranges(idxs1, idxs2)

    a, b = np.array(idxs_merged).T
    which_idxs = np.where(b[:-1]>a[1:])[0]
    print("which_idxs: {}".format(which_idxs))
    idxs1_found = [idxs1_orig.index(idxs_merged[i]) for i in which_idxs if idxs_merged[i] in idxs1_orig]
    idxs2_found = [idxs2.index(idxs_merged[i]) for i in which_idxs+1 if idxs_merged[i] in idxs2]
    print("idxs1_found: {}".format(idxs1_found))
    print("idxs2_found: {}".format(idxs2_found))
