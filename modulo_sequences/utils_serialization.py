from PIL import ImageTk
from tkinter import Tk, Label, BOTH
from tkinter.ttk import Frame, Style

import dill
import gzip
import os

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


def time_measure(f, args):
    start_time = time()
    ret = f(*args)
    end_time = time()
    diff_time = end_time-start_time
    return ret, diff_time


class ShowImg(Frame, object):
    def __init__(self, img):
        parent = Tk()
        Frame.__init__(self, parent)
        self.pack(fill=BOTH, expand=1)
        label1 = Label(self)
        label1.photo= ImageTk.PhotoImage(img)
        label1.config(image=label1.photo)
        label1.pack(fill=BOTH, expand=1)
        parent.mainloop()
