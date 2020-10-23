#! /usr/bin/python3.8

# -*- coding: utf-8 -*-

# Basic imports
import datetime
import getpass
import openpyxl
import os
import pandasql
import pymssql

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import pandas.io.sql as sql

from dotmap import DotMap
from memory_tempfile import MemoryTempfile
from typing import List, Dict

from openpyxl.styles import PatternFill, borders
from openpyxl.utils import get_column_letter

from openpyxl.workbook.workbook import Workbook
from openpyxl.worksheet.worksheet import Worksheet


HOME_DIR = os.path.expanduser("~")+"/"
TEMP_DIR = MemoryTempfile().gettempdir()+"/"
PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"

SQL_ROOT_PATH = PATH_ROOT_DIR + "../sql/"

USERNAME = getpass.getuser()    


def template_plot_nr_1():
    plt.close('all')
    figsize = (10, 8)

    fig = plt.figure(figsize=figsize)

    left = 0.10
    width = 0.70
    bottom = 0.05
    height = 0.40
    ax = fig.add_axes([left, bottom, width, height])

    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([])
    ax.set_yticks([], minor=True)

    plt.show()


if __name__ == '__main__':
    print('Hello World!')

    template_plot_nr_1()
