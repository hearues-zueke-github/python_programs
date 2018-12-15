#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

from copy import deepcopy

import numpy as np

from PIL import Image, ImageTk

# Simple enough, just import everything from tkinter.
# from tkinter import *
import tkinter as tk

import pygubu

class Application:
    def __init__(self, master):

        #1: Create a builder
        self.builder = builder = pygubu.Builder()

        #2: Load an ui file
        builder.add_from_file('test1.ui')

        #3: Create the widget using a master as parent
        self.mainwindow = builder.get_object('myMainWindow', master)


if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("600x450")
    
    app = Application(root)
    root.mainloop()
