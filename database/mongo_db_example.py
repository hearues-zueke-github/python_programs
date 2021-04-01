#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.8.6 -i

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

import numpy as np
import pandas as pd

from copy import deepcopy, copy
from dotmap import DotMap
from functools import reduce
from memory_tempfile import MemoryTempfile
from shutil import copyfile

from pymongo import MongoClient

import string
import configparser

from io import BytesIO

sys.path.append('..')
from utils import mkdirs

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"
HOME_DIR = os.path.expanduser("~")+"/"
TEMP_DIR = MemoryTempfile().gettempdir()+"/"

OBJS_DIR_PATH = PATH_ROOT_DIR+'objs/'
mkdirs(OBJS_DIR_PATH)

def convert_obj_to_bytearray(obj):
    f = BytesIO()
    dill.dump(obj, f)
    f.seek(0)
    byte_array = f.read()
    return byte_array


def convert_bytearray_to_obj(byte_array):
    f = BytesIO(byte_array)
    obj = dill.load(f)
    return obj


if __name__ == '__main__':
    # client = MongoClient('mongodb://mongoadmin:mongoadmin123#@192.168.1.110:27017')
    
    config = configparser.ConfigParser()

    config_file_path = os.path.join(HOME_DIR, 'Documents/private_files/database/mongo_db_config.ini')

    config.read(config_file_path)
    d_config = dict(config['mongo-db-config-remote'])

    protocol = d_config['protocol']
    user = d_config['user']
    password = d_config['password']
    ip = d_config['ip']
    port = d_config['port']

    url = f'{protocol}://{user}:{password}@{ip}:{port}'
    print("url: {}".format(url))

    client = MongoClient(url)
    db = client['db_python_data']
    
    col = db['some_data']

    l_base64_chars = np.array(list(string.ascii_letters + string.digits + '_-'))

    for i in range(0, 10):
        arr_content = np.random.choice(l_base64_chars, (4, ))
        str_content = ''.join(arr_content)
        col.insert_one({'i': i, 'str_content': str_content, 'arr_content': arr_content.tolist(), 'byte_array': convert_obj_to_bytearray(arr_content)})
