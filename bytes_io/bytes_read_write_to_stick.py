#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import re
import subprocess
import sys

from copy import deepcopy

from time import time
from functools import reduce

from collections import defaultdict

from PIL import Image

import numpy as np

sys.path.append("..")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

sys.path.append("../encryption")
from utils_encryption import pretty_block_printer

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

from PIL import ImageTk
from tkinter import Tk, Label, BOTH
from tkinter.ttk import Frame, Style

from cmd import Cmd


def is_sudo():
    return os.geteuid()==0

    # try:
    #     os.rename('/etc/foo', '/etc/bar')
    # except IOError as e:
    #     if (e[0] == errno.EPERM):
    #        print >> sys.stderr, "You need root permissions to do this, laterz!"
    #        return False
    # return True


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


class MyPrompt(Cmd):
    promt = "dev_byte_reader> "
    intro = "Welcome to the cmd of dev_byte_reader, where you can read/write the content of different devices!\n"
    
    def __init__(self, dev_path):
        self.dev_path = dev_path
        super(MyPrompt, self).__init__()

        if not os.path.exists(self.dev_path):
            print("File '{}' could not be found!".format(self.dev_path))
            sys.exit(-1)

        self.fw = open(self.dev_path, "wb")
        self.fr = open(self.dev_path, "rb")

        self.max_block_size = int(subprocess.check_output(['blockdev', '--getsz', self.dev_path]).decode('ascii').replace("\n", ""))
        print("self.max_block_size: {}".format(self.max_block_size))


    def do_exit(self, inp):
        print("See ya!")
        self.fw.close()
        self.fr.close()
        return True


    def do_e(self, inp):
        self.do_exit(inp)
        return True


    def convert_str_to_int(self, num_str):
        if len(num_str) > 2 and num_str[:2]=="0x":
            num = int(num_str, 16)
        elif len(num_str) > 2 and num_str[:2]=="0o":
            num = int(num_str, 8)
        elif len(num_str) > 2 and num_str[:2]=="0b":
            num = int(num_str, 2)
        else:
            num = int(num_str)
        return num


    def do_read(self, inp):
        inp_lst = re.sub(' +', ' ', inp).split(" ")

        if len(inp_lst) < 2:
            print("Needed more arguments!")
            print("Usage: read <offset> <amount>")
            return
        try:
            offset_str = inp_lst[0]
            amount_str = inp_lst[1]

            offset = self.convert_str_to_int(offset_str)
            amount = self.convert_str_to_int(amount_str)
        except:
            print("One argument was wrong!")
            return

        if offset+amount > self.max_block_size*BLOCK_SIZE:
            print("offset+amount >= last possible byte position!!!")
            return

        self.fr.seek(offset)
        block_arr = np.fromfile(self.fr, count=amount, dtype=np.uint8)

        print("offset: {0}, 0x{0:0X}; amount: {1}, 0x{1:0X}".format(offset, amount))
        pretty_block_printer(block_arr, 8, block_arr.shape[0])

        self.offset = offset
        self.amount = amount
        self.block_arr = block_arr


    def do_r(self, inp):
        self.do_read(inp)


    def do_read_block(self, inp):
        inp_lst = re.sub(' +', ' ', inp).split(" ")

        if len(inp_lst) < 1:
            print("Needed more arguments!")
            print("Usage: read <offset> <amount>")
            return
        try:
            block_nr_str = inp_lst[0]

            block_nr = self.convert_str_to_int(block_nr_str)

            offset = block_nr*BLOCK_SIZE
            amount = BLOCK_SIZE
        except:
            print("One argument was wrong!")
            return

        if offset+amount > self.max_block_size*BLOCK_SIZE:
            print("offset+amount >= last possible byte position!!!")
            return

        self.fr.seek(offset)
        block_arr = np.fromfile(self.fr, count=amount, dtype=np.uint8)

        print("block_nr: {2}, 0x{2:0X}; offset: {0}, 0x{0:0X}; amount: {1}, 0x{1:0X}".format(offset, amount, block_nr))
        pretty_block_printer(block_arr, 8, block_arr.shape[0])

        self.offset = offset
        self.amount = amount
        self.block_arr = block_arr


    def do_rblk(self, inp):
        self.do_read_block(inp)


    def do_read_block_many(self, inp):
        inp_lst = re.sub(' +', ' ', inp).split(" ")

        if len(inp_lst) < 2:
            print("Needed more arguments!")
            print("Usage: read <offset> <amount>")
            return
        try:
            block_nr_start_str = inp_lst[0]
            block_nr_end_str = inp_lst[1]

            block_nr_start = self.convert_str_to_int(block_nr_start_str)
            block_nr_end = self.convert_str_to_int(block_nr_end_str)

            if block_nr_start >= block_nr_end:
                print("Error: blk_nr_start >= blk_nr_end")
                return
            elif block_nr_start < 0:
                print("Error: blk_nr_start < 0")
                return
            elif block_nr_end > self.max_block_size:
                print("Error: block_nr_end > max_block_size")
                return

            offset = block_nr_start*BLOCK_SIZE
            amount = (block_nr_end-block_nr_start)*BLOCK_SIZE
        except:
            print("One argument was wrong!")
            return

        if offset+amount > self.max_block_size*BLOCK_SIZE:
            print("offset+amount >= last possible byte position!!!")
            return

        start_time = time()
        self.fr.seek(offset)
        block_arr = np.fromfile(self.fr, count=amount, dtype=np.uint8)
        end_time = time()

        print("end_time-start_time: {:.4} s".format(end_time-start_time))

        print("block_nr_start: {2}, 0x{2:0X}; block_nr_end: {3}, 0x{3:0X}; offset: {0}, 0x{0:0X}; amount: {1}, 0x{1:0X}".format(offset, amount, block_nr_start, block_nr_end))
        # pretty_block_printer(block_arr, 8, block_arr.shape[0])

        self.block_arr_many = block_arr.reshape((-1, BLOCK_SIZE))

        self.block_nr_start = block_nr_start
        self.block_nr_end = block_nr_end

        self.offset = offset
        self.amount = amount
        self.block_arr = block_arr


    def do_rblkm(self, inp):
        self.do_read_block_many(inp)


    def do_write_random(self, inp):
        inp_lst = re.sub(' +', ' ', inp).split(" ")

        if len(inp_lst) < 2:
            print("Needed more arguments!")
            print("Usage: read <offset> <amount>")
            return
        try:
            offset_str = inp_lst[0]
            amount_str = inp_lst[1]

            offset = self.convert_str_to_int(offset_str)
            amount = self.convert_str_to_int(amount_str)
        except:
            print("One argument was wrong!")
            return

        if offset+amount > self.max_block_size*BLOCK_SIZE:
            print("offset+amount >= last possible byte position!!!")
            return

        self.fw.seek(offset)
        block_arr = np.random.randint(0, 256, (amount, ), dtype=np.uint8)
        block_arr.tofile(self.fw)

        print("Written data to:")
        print("offset: {0}, 0x{0:0X}; amount: {1}, 0x{1:0X}".format(offset, amount))
        pretty_block_printer(block_arr, 8, block_arr.shape[0])


    def do_wrnd(self, inp):
        self.do_write_random(inp)


    def do_write_zero(self, inp):
        inp_lst = re.sub(' +', ' ', inp).split(" ")

        if len(inp_lst) < 2:
            print("Needed more arguments!")
            print("Usage: read <offset> <amount>")
            return
        try:
            offset_str = inp_lst[0]
            amount_str = inp_lst[1]

            offset = self.convert_str_to_int(offset_str)
            amount = self.convert_str_to_int(amount_str)
        except:
            print("One argument was wrong!")
            return

        if offset+amount > self.max_block_size*BLOCK_SIZE:
            print("offset+amount >= last possible byte position!!!")
            return

        self.fw.seek(offset)
        block_arr = np.zeros((amount, ), dtype=np.uint8)
        block_arr.tofile(self.fw)

        print("Written data to:")
        print("offset: {0}, 0x{0:0X}; amount: {1}, 0x{1:0X}".format(offset, amount))
        # pretty_block_printer(block_arr, 8, block_arr.shape[0])


    def do_wzr(self, inp):
        self.do_write_zero(inp)


    def do_pl(self, inp):
        if not hasattr(self, 'block_arr'):
            print("Please call the read_block or the read command first!")
            return

        try:
            print("Printing last read block:")
            print("offset: {0}, 0x{0:0X}; amount: {1}, 0x{1:0X}".format(self.offset, self.amount))
            pretty_block_printer(self.block_arr, 8, self.block_arr.shape[0])
        except:
            print("Something went wrong!")


    def do_getattr(self, inp):
        if not hasattr(self, inp):
            print("Does not contain attribute '{}'!".format(inp))
            return

        print("{}: {}".format(inp, getattr(self, inp)))


    def do_add(self, inp):
        inp_lst = re.sub(' +', ' ', inp).split(" ")
        print("Adding '{}'".format(inp_lst))


BLOCK_SIZE = 512

if __name__ == "__main__":
    # stick_path = "/dev/sdb"
    stick_path = "/dev/mmcblk0"
    whole_block_size = BLOCK_SIZE*0x1000
    offset = whole_block_size*10

    print("will use '{}' as file!".format(stick_path))

    if not is_sudo():
        print("Start this script as sudo!")
        print("Exit...")
        sys.exit(0)
        
    my_prompt = MyPrompt(dev_path=stick_path)
    # my_prompt.cmdloop()
