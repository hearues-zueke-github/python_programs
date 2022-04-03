import dill
import gzip
import os

def save_pkl_gz_obj(obj, file_path):
    with gzip.open(file_path, "wb") as f:
        dill.dump(obj, f)

def load_pkl_gz_obj(file_path):
    with gzip.open(file_path, "rb") as f:
        obj = dill.load(f)
    return obj

def get_pkl_gz_obj(func, file_path):
    if not os.path.exists(file_path):
        obj = func()
        save_pkl_gz_obj(obj=obj, file_path=file_path)
    else:
        obj = load_pkl_gz_obj(file_path=file_path)
    return obj
