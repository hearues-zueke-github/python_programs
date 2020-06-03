#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import gzip
import os

from memory_tempfile import MemoryTempfile
tempfile = MemoryTempfile()

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

f = tempfile.NamedTemporaryFile(prefix="Test", suffix=".txt")
TEMP_DIR = f.name[:f.name.rfind("/")]+"/"

TEMP_DIR_OBJS = TEMP_DIR+f"python_objects/"
if not os.path.exists(TEMP_DIR_OBJS):
    os.makedirs(TEMP_DIR_OBJS)

def save_dict_object(d, file_name_prefix):
    with gzip.open(TEMP_DIR_OBJS+file_name_prefix+'.pkl.gz', 'wb') as f:
        dill.dump(d, f)


def load_dict_object(file_name_prefix):
    with gzip.open(TEMP_DIR_OBJS+file_name_prefix+'.pkl.gz', 'rb') as f:
        d = dill.load(f)
    return d
