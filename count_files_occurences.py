#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import sys

from copy import deepcopy

from time import time
from functools import reduce
from collections import defaultdict

from PIL import Image

import numpy as np

from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

from PIL import ImageTk
from tkinter import Tk, Label, BOTH
from tkinter.ttk import Frame, Style


if __name__ == "__main__":
    count_extensions = defaultdict(int)

    ignore_folders = [".git"]
    for root_dir, dirs, files in os.walk(PATH_ROOT_DIR):
        should_ignore_root_dir = False
        for ignore_folder in ignore_folders:
            if ignore_folder in root_dir:
                should_ignore_root_dir = True
                break

        if should_ignore_root_dir:
            continue

        # print("root_dir: {}".format(root_dir))
        # print("- len(dirs): {}".format(len(dirs)))
        # print("- len(files): {}".format(len(files)))

        only_extensions = list(map(lambda x: None if not "." in x else x.split(".")[-1], files))
        for ext in only_extensions:
            count_extensions[ext] += 1

    count_extensions = dict(count_extensions)
    # print("count_extensions: {}".format(count_extensions))

    del count_extensions[None]

    keys = sorted(list(count_extensions.keys()))
    
    max_len_key = reduce(lambda a, b: a if a > len(b) else len(b), keys, 0)
    max_len_val = reduce(lambda a, b: a if a > len(str(b)) else len(str(b)), count_extensions.values(), 0)

    str_template = "{{:{len_key}}}| {{:{len_val}}}".format(len_key=max_len_key, len_val=max_len_val)

    print("File extensions counts occurences:")
    for key in keys:
        # print("{}: {}".format(key, count_extensions[key]))
        print(str_template.format(key, count_extensions[key]))
