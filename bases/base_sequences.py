#! /usr/bin/python3

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import dill
import gzip
import os
import sys

import numpy as np

import matplotlib.pyplot as plt

from functools import reduce
from dotmap import DotMap

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"

if __name__=='__main__':
    # base1 = 3, convert num to base2 = 2
        
