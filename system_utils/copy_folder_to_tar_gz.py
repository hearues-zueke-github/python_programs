#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.9.5 -i

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import dill
import gzip
import os
import pdb
import re
import sys
import traceback
import tarfile

import numpy as np
import pandas as pd
import multiprocessing as mp

from collections import defaultdict
from copy import deepcopy, copy
from dotmap import DotMap
from functools import reduce
from hashlib import sha256
from io import BytesIO
from memory_tempfile import MemoryTempfile
from shutil import copyfile
from pprint import pprint
from typing import List, Set, Tuple, Dict, Union, Any
from PIL import Image
from recordclass import RecordClass, asdict

class RootDirsFiles(RecordClass):
    root: str
    l_dir_name: List[str]
    l_file_name: List[str]

CURRENT_WORKING_DIR = os.getcwd()
PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = '/home/dpmcltrmj'
# HOME_DIR = os.path.expanduser("~")
TEMP_DIR = MemoryTempfile().gettempdir()
PYTHON_PROGRAMS_DIR = os.path.join(HOME_DIR, 'git/python_programs')

# set the relative/absolute path where the utils_load_module.py file is placed!
sys.path.append(PYTHON_PROGRAMS_DIR)
from utils_load_module import load_module_dynamically

var_glob = globals()
load_module_dynamically(**dict(var_glob=var_glob, name='utils', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='utils_multiprocessing_manager', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils_multiprocessing_manager.py")))

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

PLOTS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'plots')
mkdirs(PLOTS_DIR_PATH)

if __name__ == '__main__':
    argv = sys.argv
    src_folder_path = argv[1]
    dst_file_path = argv[2]

    assert os.path.exists(src_folder_path)
    assert os.path.isdir(src_folder_path)
    assert not os.path.islink(src_folder_path)

    assert '.'.join(dst_file_path.split('.')[-2:]) == 'tar.gz'

    if not os.path.exists(dst_file_path):
        with tarfile.open(name=dst_file_path, mode='w:gz') as tar:
            pass

    dst_obj_file_path = dst_file_path[:-6] + 'objs.tar.gz'

    tar = tarfile.open(name=dst_file_path, mode='w:gz')

    l_dir_path_link = []
    l_file_path_link = []
    l_file_path_failed = []

    # tar = tarfile.open(name=dst_file_path, mode='r:gz')
    # members = tar.getmembers()

    # first walk through all the dirs and files
    l_rootdirfile = [RootDirsFiles(*next(os.walk(src_folder_path)))]
    l_rootdirfile_temp = [rootdirfile for rootdirfile in l_rootdirfile]

    iter_nr = 0
    while len(l_rootdirfile_temp) > 0:
        iter_nr += 1
        print("iter_nr: {}".format(iter_nr))

        len_l_rootdirfile_temp = len(l_rootdirfile_temp)
        l_rootdirfile_temp_2 = []
        for i_rootdirfile, rootdirfile in enumerate(l_rootdirfile_temp, 1):
            print("- rootdirfile {:5}/{:5}".format(i_rootdirfile, len_l_rootdirfile_temp))
            root = rootdirfile.root
            for dir_name in rootdirfile.l_dir_name:
                dir_path = os.path.join(root, dir_name)
                if os.path.islink(dir_path):
                    l_dir_path_link.append(dir_path)
                    print('-- path is link! "{}"'.format(dir_path))
                    continue
                l_rootdirfile_temp_2.append(RootDirsFiles(*next(os.walk(dir_path))))

        l_rootdirfile_temp = l_rootdirfile_temp_2
        l_rootdirfile.extend(l_rootdirfile_temp)

        # if iter_nr >= 2:
        #     break


    # then obtain all stats of each dirs and files
    root_first = l_rootdirfile[0].root
    len_root_first = len(root_first) + 1
    df = pd.DataFrame(data=[], columns=['rel_root', 'name', 'type', ' os.stat', 'sha256sum'], dtype=object)
    row_nr = 0

    len_l_rootdirfile = len(l_rootdirfile)
    for i_rootdirfile, rootdirfile in enumerate(l_rootdirfile, 1):
        print("rootdirfile nr. {:5}/{:5}".format(i_rootdirfile, len_l_rootdirfile))
        root = rootdirfile.root
        rel_root = root[len_root_first:]

        for dir_name in rootdirfile.l_dir_name:
            dir_path = os.path.join(root, dir_name)
            stat = os.stat(dir_path)
            df.loc[row_nr] = [rel_root, dir_name, 'd', stat, '']
            row_nr += 1

        for file_name in rootdirfile.l_file_name:
            file_path = os.path.join(root, file_name)
            stat = os.stat(file_path)

            df.loc[row_nr] = [rel_root, file_name, 'f', stat, '']
            row_nr += 1


    # and last step save files in dirs in the tar.gz, with the stats seperated + update the sha256sum
    df_file = df.loc[df['type'].values == 'f']
    len_df_file = len(df_file)
    for i, (row_nr, row) in enumerate(df_file.iterrows(), 1):
        rel_root = row['rel_root']
        name = row['name']
        rel_file_path = os.path.join(rel_root, name)
        file_path = os.path.join(root_first, rel_file_path)

        print("{:6}/{:6} copy {}".format(i, len_df_file, file_path))

        bytes_file = BytesIO()
        try:
            with open(file_path, 'rb') as f:
                bytes_file.write(f.read())
            bytes_file.seek(0)
        except:
            print('- Could not open the file!')
            df.loc[row_nr].sha256sum =  ''
            l_file_path_failed.append(file_path)
            continue

        h = sha256()
        h.update(bytes_file.read())
        df.loc[row_nr]['sha256sum'] =  h.hexdigest()

        try:
            tarinfo = tarfile.TarInfo(name=rel_file_path)
            tarinfo.size = bytes_file.tell()
            bytes_file.seek(0)
            tar.addfile(tarinfo=tarinfo, fileobj=bytes_file)
        except:
            print('- Could not copy the file!')
            l_file_path_failed.append(file_path)
            continue

    tar.close()

    tar = tarfile.open(name=dst_file_path, mode='r:gz')
    members = tar.getmembers()
    
    d_ignores = {
        'l_dir_path_link': l_dir_path_link,
        'l_file_path_link': l_file_path_link,
        'l_file_path_failed': l_file_path_failed,
    }

    l_obj_obj_name = [
        (d_ignores, 'd_ignores.pkl'),
        ([asdict(v) for v in l_rootdirfile], 'l_rootdirfile.pkl'),
        (df, 'df.pkl'),
    ]
    
    tar_obj = tarfile.open(name=dst_obj_file_path, mode='w:gz')
    for obj, obj_name in l_obj_obj_name:
        bytes_file = BytesIO()
        dill.dump(obj, bytes_file)

        tarinfo = tarfile.TarInfo(name=obj_name)
        tarinfo.size = bytes_file.tell()
        bytes_file.seek(0)
        tar_obj.addfile(tarinfo=tarinfo, fileobj=bytes_file)

    tar_obj.close()

    sys.exit()

    # first attempt
    root_first, _, _ = next(os.walk(src_folder_path))
    len_root_first = len(root_first)
    for iter_nr, (root, l_dir_name, l_file_name) in enumerate(os.walk(src_folder_path), 0):
        root_short = root[len_root_first:]

        for file_name in l_file_name:
            src_file_path = os.path.join(root, file_name)
            in_tar_file_path = os.path.join(root_short, file_name).lstrip('/')

            if os.path.islink(src_file_path):
                print('Skip link "{}"'.format(src_file_path))
                l_file_path_link.append(src_file_path)
                continue

            print('copy "{}" -> "{}"'.format(src_file_path, in_tar_file_path))
            try:
                bytes_file = io.BytesIO()
                with open(src_file_path, 'rb') as f:
                    bytes_file.write(f.read())

                tarinfo = tarfile.TarInfo(name=in_tar_file_path)
                tarinfo.size = bytes_file.tell()
                bytes_file.seek(0)
                tar.addfile(tarinfo=tarinfo, fileobj=bytes_file)
                # tar.addfile(tarinfo=tarinfo, fileobj=f)
            except:
                print('- Could not copy the file!')
                l_file_path_failed.append(src_file_path)

        # if iter_nr >= 5:
        #     break

    tar.close()

    tar = tarfile.open(name=dst_file_path, mode='r:gz')
    members = tar.getmembers()
    # tar.close()

    # TODO: create class for root and l_dir_name and l_file_name
