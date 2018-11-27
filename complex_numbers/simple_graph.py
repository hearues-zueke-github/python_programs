#! /usr/bin/python3.6

import colorsys
import datetime
import dill
import os
import string
import sys
import traceback

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

if __name__ == "__main__":

    plt.figure()

    x = np.arange(-5, 5.1, 0.1)
    y = x**2-3*x+4

    plt.grid(True)

    plt.plot(x, y, "b-")

    plt.ylim([0-0.5, 20+0.5])

    plt.show()
