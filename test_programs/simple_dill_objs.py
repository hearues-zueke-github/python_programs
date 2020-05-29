#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import functools
import os
import sys
import time

from copy import deepcopy

import numpy as np

from PIL import Image, ImageTk

import tkinter as tk
import tkinter.ttk as ttk

import multiprocessing as mp
from multiprocessing import Process, Pipe # , Lock
from recordclass import recordclass, RecordClass

from threading import Lock

import base64
import json

import platform
# print("platform.system(): {}".format(platform.system()))

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

if __name__ == "__main__":
    print('Hello World!')
    PATH_DIR_OBJS = PATH_ROOT_DIR+'objs/'
    if not os.path.exists(PATH_DIR_OBJS):
        os.makedirs(PATH_DIR_OBJS)

    # with open(PATH_DIR_OBJS+'int1.hex', 'wb') as f:
    #     dill.dump(0x123456789ABCEDF0123123, f)

    # with open(PATH_DIR_OBJS+'int2.hex', 'wb') as f:
    #     dill.dump(0x10000000000000, f)

    with open(PATH_DIR_OBJS+'list_01.hex', 'wb') as f:
        dill.dump([0x1000], f)
