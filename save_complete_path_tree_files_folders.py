#! /usr/bin/python3

# -*- coding: utf-8 -*-

import datetime
import dill
import gzip
import os
import sys

import matplotlib.pyplot as plt

import numpy as np

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"

def save_tree_of_all_files_folders_approach_1():
    argv = sys.argv
    # should be absolute path!
    STARTING_ROOT_DIR = argv[1]
    STARTING_ROOT_DIR = STARTING_ROOT_DIR.rstrip('/')+'/'
    print("STARTING_ROOT_DIR: {}".format(STARTING_ROOT_DIR))

    SAVE_TO_PATH = argv[2]
    dir_path = '/'.join(SAVE_TO_PATH.split('/')[:-1])
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    fo = open(SAVE_TO_PATH, 'w')

    fo.write('STARTING_ROOT_DIR:{}\n'.format(STARTING_ROOT_DIR))
    for root_dir, folders, files in os.walk(STARTING_ROOT_DIR):
        root_dir_new = root_dir[len(STARTING_ROOT_DIR):]
        print("root_dir_new: {}".format(root_dir_new))

        line = '{root_dir_new};d:{l_dirs};f:{l_files}'.format(
            root_dir_new=root_dir_new,
            l_dirs='|'.join(sorted(folders)),
            l_files='|'.join(sorted(files)),
        )
        fo.write(line+'\n')

    fo.close()


if __name__ == "__main__":
    print('Hello World!')

    argv = sys.argv
    # should be absolute path!
    STARTING_ROOT_DIR = argv[1]
    STARTING_ROOT_DIR = STARTING_ROOT_DIR.rstrip('/')+'/'
    print("STARTING_ROOT_DIR: {}".format(STARTING_ROOT_DIR))

    SAVE_TO_PATH = argv[2]
    dir_path = '/'.join(SAVE_TO_PATH.split('/')[:-1])
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    fo = open(SAVE_TO_PATH, 'w')

    d_complete = {'STARTING_ROOT_DIR': STARTING_ROOT_DIR}
    d_tree = {}

    files_count = 0
    dirs_count = 0
    # first, create the whole tree! each new directory is a new dictionary!
    for root_dir, folders, files in os.walk(STARTING_ROOT_DIR):
        root_dir_new = root_dir[len(STARTING_ROOT_DIR):]
        print("root_dir_new: {}".format(root_dir_new))

        dirs_count += len(folders)
        files_count += len(files)

        print("- dirs_count: {}, files_count: {}".format(dirs_count, files_count))

        l_dir_split = root_dir_new.split('/')
        
        d_node = d_tree
        for dir_name in l_dir_split:
            if not dir_name in d_node:
                d_node[dir_name] = {}
            d_node = d_node[dir_name]

        for folder_name in folders:
            d_node[folder_name] = {}

        d_node['/files/'] = files

    # next, create mapping of names to numbers!
    # TODO next!

    fo.write(str(d_complete)+'\n')
    fo.write(str(d_tree)+'\n')
    fo.close()
