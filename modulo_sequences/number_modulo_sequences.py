#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import sys

from copy import deepcopy

sys.path.append("../combinatorics/")
import different_combinations as combinations
# sys.path.append("../graph_theory/")
# from find_graph_cycles import get_cycles_of_1_directed_graph

from PIL import Image

import numpy as np

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

from PIL import ImageTk
from tkinter import Tk, Label, BOTH
from tkinter.ttk import Frame, Style

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

if __name__ == "__main__":
    # m, n = sys.argv[1:3]
    # m = int(m)
    # n = int(n)

    m = int(sys.argv[1])

    # linear modulos with c = 0
    # x_(i+1) = a*x_i+c

    # a = 

    def get_img(arr):
        arr = arr.astype(np.float64)/(np.max(arr)+1)
        pix = (arr*255.999).astype(np.uint8)
        img = Image.fromarray(pix)
        return img

    # c = 5
    # c = c % m

    # xs = np.arange(0, m).reshape((1, -1))
    # cs = np.arange(0, m).reshape((1, -1))
    # asz = np.arange(0, m).reshape((-1, 1))

    # arr = np.zeros((m, m), dtype=np.int)
    # scale = 15

    for m in range(1, 11):

        vals = np.arange(0, m)
        full_cycles = {}
        
        for i, a in enumerate(vals, 0):
            # print("i: {}, a: {}".format(i, a))
            for j, c in enumerate(vals, 0):
                arr = np.zeros((m, m), dtype=np.int)
                x_iter = vals.copy()
                arr[:, 0] = x_iter
                for l in range(1, m, 1):
                    x_iter = (a*x_iter+c) % m
                    arr[:, l] = x_iter
                # print("j: {}, c: {}, arr:\n{}".format(j, c, arr))
                # img = get_img(arr)
                # img = img.resize((img.width*scale, img.height*scale))
                
                # img.show()
                # ShowImg(img)

                for row in arr:
                    if np.unique(row).shape[0]==m:
                        full_cycles[(a, c, row[0])] = row.tolist()
        # print("full_cycles:\n{}".format(full_cycles))
        print("m: {}, len(full_cycles): {}".format(m, len(full_cycles)))

    # modulo_arr = asz.dot(xs) % m
    # print("modulo_arr:\n{}".format(modulo_arr))

    # convert modulo_arr to an image
    # normalized_arr = modulo_arr.astype(np.float64)/(np.max(modulo_arr)+1)
    # pix = (normalized_arr*255.999).astype(np.uint8)
    # img = Image.fromarray(pix)
    # img = img.resize((img.width*scale, img.height*scale))
    # img.show()
