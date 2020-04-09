import dill
import gzip
import os
import sys

TEMP_FOLDER_PATH = '/tmp/python_objs/'
if not os.path.exists(TEMP_FOLDER_PATH):
    os.makedirs(TEMP_FOLDER_PATH)

def save_object(obj_name, obj):
    with gzip.open(TEMP_FOLDER_PATH+'{}.pkl.gz', 'wb') as f:
        dill.dump(obj, f)

def load_object(obj_name):
    with gzip.open(TEMP_FOLDER_PATH+'{}.pkl.gz', 'rb') as f:
        obj = dill.load(f)
    return obj
