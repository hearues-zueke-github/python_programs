import dill
import gzip
import os

def get_pkl_obj(func, file_path):
    if not os.path.exists(file_path):
        obj = func()
        with open(file_path, "wb") as f:
            dill.dump(obj, f)
    else:
        with open(file_path, "rb") as f:
            obj = dill.load(f)
    return obj

def save_pkl_obj(obj, file_path):
    with open(file_path, "wb") as f:
        dill.dump(obj, f)

def load_pkl_obj(file_path):
    with open(file_path, "rb") as f:
        obj = dill.load(f)
    return obj

def get_pkl_gz_obj(func, file_path):
    if not os.path.exists(file_path):
        obj = func()
        with gzip.open(file_path, "wb") as f:
            dill.dump(obj, f)
    else:
        with gzip.open(file_path, "rb") as f:
            obj = dill.load(f)
    return obj

def save_pkl_gz_obj(obj, file_path):
    with gzip.open(file_path, "wb") as f:
        dill.dump(obj, f)

def load_pkl_gz_obj(file_path):
    with gzip.open(file_path, "rb") as f:
        obj = dill.load(f)
    return obj
