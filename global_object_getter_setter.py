import dill
import gzip
import os
import sys

TEMP_FOLDER_PATH = '/tmp/python_objs/'
if not os.path.exists(TEMP_FOLDER_PATH):
    os.makedirs(TEMP_FOLDER_PATH)

def save_object(obj_name, obj):
    FILE_PATH_ABS = TEMP_FOLDER_PATH+'{}.pkl.gz'.format(obj_name)
    with gzip.open(FILE_PATH_ABS, 'wb') as f:
        dill.dump(obj, f)


def load_object(obj_name):
    FILE_PATH_ABS = TEMP_FOLDER_PATH+'{}.pkl.gz'.format(obj_name)
    with gzip.open(FILE_PATH_ABS, 'rb') as f:
        obj = dill.load(f)
    return obj


def do_object_exist(obj_name):
    FILE_PATH_ABS = TEMP_FOLDER_PATH+'{}.pkl.gz'.format(obj_name)
    return os.path.exists(FILE_PATH_ABS)


def delete_object(obj_name):
    FILE_PATH_ABS = TEMP_FOLDER_PATH+'{}.pkl.gz'.format(obj_name)
    if os.path.exists(FILE_PATH_ABS):
        os.remove(FILE_PATH_ABS)
