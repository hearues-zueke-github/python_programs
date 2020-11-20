import datetime
import string

from time import time

import numpy as np

from PIL import Image, ImageTk
from tkinter import Tk, Label, BOTH
from tkinter.ttk import Frame, Style

all_symbols_16 = np.array(list("0123456789ABCDEF"))
def get_random_str_base_16(n):
    l = np.random.randint(0, 16, (n, ))
    return "".join(all_symbols_16[l])


all_symbols_64 = np.array(list(string.ascii_lowercase+string.ascii_uppercase+string.digits+"-_"))
def get_random_str_base_64(n):
    l = np.random.randint(0, 64, (n, ))
    return "".join(all_symbols_64[l])


def get_date_time_str_full():
    dt = datetime.datetime.now()
    dt_params = (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond)
    return "Y{:04}_m{:02}_d{:02}_H{:02}_M{:02}_S{:02}_f{:06}".format(*dt_params)


def get_date_time_str_full_short():
    dt = datetime.datetime.now()
    dt_params = (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond)
    return "{:04}_{:02}_{:02}_{:02}_{:02}_{:02}_{:06}".format(*dt_params)


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


def int_sqrt(n):
    x_prev = n
    x_now = (n//1+1)//2
    while x_now<x_prev:
        t = (n//x_now+x_now)//2
        x_prev = x_now
        x_now = t
    return x_now
