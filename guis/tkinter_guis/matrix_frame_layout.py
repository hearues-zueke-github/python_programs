#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import functools
import sys
import time

sys.path.append("../../picture_manipulation")
import approx_random_images

from copy import deepcopy

import numpy as np

from PIL import Image, ImageTk

# Simple enough, just import everything from tkinter.
# from tkinter import *
import tkinter as tk
import tkinter.ttk as ttk

import multiprocessing as mp
from multiprocessing import Process, Pipe
from threading import Lock

import platform
print("platform.system(): {}".format(platform.system()))

class Datas(Exception):
    def __init__(self, other):
        self.other = other

        self.img_w = self.other.img_w
        self.img_h = self.other.img_h

        self.x = None
        self.y = None
        self.idx = None

        self.dm = None
        self.pixs = []
        self.imgs = []
        self.imgtks = []

        self.frm = None
        self.labl = None

        self.lock = Lock()

        self.should_stop_animation = False
        self.animation_time = 200

        self.lablfrm_matrix_lambda_images = other.lablfrm_matrix_lambda_images

        self.labl_width = None
        self.labl_height = None
        self.frm_width = None
        self.frm_height = None


    def create_new_images(self):
        pixs, dm = approx_random_images.create_1_bit_neighbour_pictures(self.img_h, self.img_w, return_pix_array=True, save_pictures=False)

        arr_zero = np.zeros(pixs[0].shape, dtype=np.uint8)

        pixs = [arr_zero+np.array([0x00, 0xFF, 0x00], dtype=np.uint8)]*1+\
               pixs+\
               [arr_zero+np.array([0xFF, 0x00, 0x00], dtype=np.uint8)]*1

        self.pixs = pixs
        self.dm = dm

        for pix in self.pixs:
            img = Image.fromarray(pix)
            imgtk = ImageTk.PhotoImage(img)

            # self.pixs.append(pix)
            self.imgs.append(img)
            self.imgtks.append(imgtk)


    def delete_images(self):
        self.pixs = []
        self.imgs = []
        self.imgtks = []


    def destroy(self):
        while self.lock.acquire(False):
            time.sleep(0.0001)

        self.should_stop_animation = True
        self.lock.release()

        self.delete_images()
        self.frm.destroy()
        self.labl.destroy()


    def update_frame_random(self, idx):
        while self.lock.acquire(False):
            pass
            # time.sleep(0.0001)

        if self.should_stop_animation:
            self.lock.release()
            return

        while self.other.lock.acquire(False):
            pass
            # time.sleep(0.0001)

        frame = self.imgtks[idx]

        self.other.lock.release()
        self.lock.release()
        
        idx += 1
        if idx >= len(self.imgtks):
            idx = 0
        self.labl.configure(image=frame)

        self.other.after(self.animation_time, self.update_frame_random, idx)


    def label_clicked(self, event):
        print("event: {}".format(event))
        self.write_to_text_label(event)
        self.show_matrices_pictures(event)

        # TODO: create the images of the matrices!!!


    def write_to_text_label(self, event):
        txt_box = self.other.txt_box

        txt_box.delete('1.0', tk.END)

        str_line = "idx: {:3}, y: {:2}, x: {:2}\n".format(self.idx, self.y, self.x)
        str_line += "event.y: {:2}, event.x: {:2}\n".format(event.y, event.x)
        str_line += "length: {}\n".format(len(self.pixs)-2)
        # txt_box.insert(tk.END, str_line)

        str_line += "lambdas: {}\n".format(len(self.dm.function_str_lst))

        # print("self.dm.idx_choosen_params_lst:\n{}".format(self.dm.idx_choosen_params_lst))
        # print("self.dm.idx_inv_params_lst:\n{}".format(self.dm.idx_inv_params_lst))
        # txt_box.insert(tk.END, "lambdas:\n")
        for i, (line, idx_choosen_params_lst, idx_inv_params_lst) in enumerate(zip(self.dm.function_str_lst, self.dm.idx_choosen_params_lst, self.dm.idx_inv_params_lst), 0):
            str_line += "i: {}, {}\n".format(i, line)
            str_line += "choosen_params:\n{}\n{}\n".format(i, idx_choosen_params_lst)
            str_line += "inv_params:\n{}\n{}\n".format(i, idx_inv_params_lst)
        # print("str_line:\n{}".format(str_line))
            # txt_box.insert(tk.END, "i: {}, {}\n".format(i, line))
        txt_box.insert(tk.END, str_line)
        txt_box.see("1.0")
        # txt_box.see(tk.END)


    def show_matrices_pictures(self, event):
        lablfrm = self.other.lablfrm_matrix_lambda_images
        lablfrm.delete_matrices_labls()

        dm = self.dm

        ft = dm.ft
        print("ft: {}".format(ft))
        length = ft*2+1

        resize_factor = 20
        length_resize = length*resize_factor

        self.labl_borderwidth = 0

        self.frm_width = length_resize+2+self.labl_borderwidth*2
        self.frm_height = self.frm_width
        self.labl_width = length_resize
        self.labl_height = self.labl_width

        idx_choosen_params_lst = dm.idx_choosen_params_lst
        idx_inv_params_lst = dm.idx_inv_params_lst

        print("self.labl_width: {}".format(self.labl_width))
        print("self.labl_height: {}".format(self.labl_height))

        for y, (choosen_arr, inv_arr) in enumerate(zip(idx_choosen_params_lst, idx_inv_params_lst)):
            print("choosen_arr:\n{},\ninv_arr:\n{}".format(choosen_arr, inv_arr))
            var_arr = choosen_arr+((choosen_arr+inv_arr)==2)
            var_arr_reshape = var_arr.reshape((-1, length, length))
            print("var_arr_reshape:\n{}".format(var_arr_reshape))

            for x, var_arr_matrix in enumerate(var_arr_reshape):
                frm = tk.Frame(lablfrm.interior, height=self.frm_height, width=self.frm_width, bg=lablfrm.bgs[(x+y)%2])
                frm.place(x=self.frm_width*x, y=self.frm_height*y)
                frm.pack_propagate(False)

                arr = np.zeros(var_arr_matrix.shape+(3, ), dtype=np.uint8)
                arr[var_arr_matrix==1] = (0xFF, 0x00, 0x00)
                arr[var_arr_matrix==2] = (0x00, 0x00, 0xFF)
                img = Image.fromarray(arr).resize((self.labl_width, self.labl_height))
                pix = np.array(img)
                imgtk = ImageTk.PhotoImage(img)

                lablfrm.pixs[(y, x)] = pix
                lablfrm.imgs[(y, x)] = img
                lablfrm.imgtks[(y, x)] = imgtk

                labl = tk.Label(frm, bg=lablfrm.bgs_labl[(x+y)%2], borderwidth=self.labl_borderwidth, image=imgtk)
                labl.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
                
                lablfrm.frms[(y, x)] = frm
                lablfrm.labls[(y, x)] = labl

                if lablfrm.max_y < y:
                    lablfrm.max_y = y
                if lablfrm.max_x < x:
                    lablfrm.max_x = x

                lablfrm.idxs_y_x.append((y, x))

        lablfrm.interior.config(width=self.frm_width*(lablfrm.max_x+1), height=self.frm_height*(lablfrm.max_y+1))


class LabelFrameCanvasScrollbar(tk.LabelFrame):
    def __init__(self, parent, text, height, width, y, x, scrbar_x=False, scrbar_y=True):
        self.bgs = ["#55BB00", "#00BB55"]
        self.bgs_labl = ["#22BB00", "#00BB22"]
        # self.bgs_labl = ["#00FF00", "#3300AA"]

        self.pixs = {}
        self.imgs = {}
        
        self.imgtks = {}
        self.frms = {}
        self.labls = {}

        self.idxs_y_x = []

        self.max_x = -1
        self.max_y = -1

        self.scrbar_x = scrbar_x
        self.scrbar_y = scrbar_y

        self.parent = parent
        super(LabelFrameCanvasScrollbar, self).__init__(parent)

        self.config(text=text, height=height, width=width, borderwidth=3, relief="solid")
        self.place(y=y, x=x)
        self.pack_propagate(False)

        self.canvas = tk.Canvas(self)
        
        kwargs = {}
        
        if self.scrbar_x:
            self.scrollbar_x = tk.Scrollbar(self, orient='horizontal')
            kwargs["xscrollcommand"] = self.scrollbar_x.set
            self.scrollbar_x.config(command=self.canvas.xview)
            self.scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        if self.scrbar_y:
            self.scrollbar_y = tk.Scrollbar(self, orient='vertical')
            kwargs["yscrollcommand"] = self.scrollbar_y.set
            self.scrollbar_y.config(command=self.canvas.yview)
            self.scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas.config(background="#004000", **kwargs)
        
        self.canvas.pack(expand=True, fill='both', side='left')

        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

        self.update()
        # self.scrollbar_x.config(width=self.scrollbar_x.winfo_width()-50)

        width = self.canvas.winfo_width()#-self.scrollbar_2.winfo_width()-1
        print("width 2!!!: {}".format(width))
        self.interior = tk.Frame(self.canvas, width=width, height=0, bg="#000000")

        self.interior_id = self.canvas.create_window(0, 0,
                                                     window=self.interior,
                                                     anchor='nw')

        self.interior.bind("<Configure>", self.onFrameConfigure)


    def onFrameConfigure(self, event):
        '''Reset the scroll region to encompass the inner frame'''
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


    def delete_matrices_labls(self):
        for y, x in self.idxs_y_x:
            self.frms[(y, x)].destroy()
            self.labls[(y, x)].destroy()

        self.idxs_y_x = []

        self.pixs = {}
        self.imgs = {}
        self.imgtks = {}

        self.frms = {}
        self.labls = {}

        self.max_x = -1
        self.max_y = -1


# Here, we are creating our class, Window, and inheriting from the Frame
# class. Frame is a class from the tkinter module. (see Lib/tkinter/__init__)
class Window(tk.Frame):

    # Define settings upon initialization. Here you can specify
    def __init__(self, master=None):
        
        # parameters that you want to send through the Frame class. 
        tk.Frame.__init__(self, master)

        #reference to the master widget, which is the tk window                 
        self.master = master

        #with that, we want to then run init_window, which doesn't yet exist

        self.cpu_count = mp.cpu_count()
        print("self.cpu_count: {}".format(self.cpu_count))

        self.pipes_main_proc = [Pipe() for i in range(0, self.cpu_count)]
        self.pipes_proc_main = [Pipe() for i in range(0, self.cpu_count)]
        
        self.pipes_proc_recv = [pipe[0] for pipe in self.pipes_main_proc]
        self.pipes_main_send = [pipe[1] for pipe in self.pipes_main_proc]
        
        self.pipes_main_recv = [pipe[0] for pipe in self.pipes_proc_main]
        self.pipes_proc_send = [pipe[1] for pipe in self.pipes_proc_main]
        
        self.lock = Lock()

        self.ps = [Process(target=self.thread_func, args=(self.pipes_proc_recv[i], self.pipes_proc_send[i])) for i in range(0, self.cpu_count)]
        for p in self.ps: p.start()

        self.init_window()


    def thread_func(self, pipe_recv, pipe_send):
        while True:
            if not pipe_recv.poll():
                time.sleep(0.01)
                continue

            f_enc, args = pipe_recv.recv()
            f = dill.loads(f_enc)
            ret = f(*args)
            pipe_send.send(ret)
            # pipe_send.send(dill.dumps(ret))


    #Creation of init_window
    def init_window(self):

        self.last_clicked_idx = None

        # changing the title of our master widget      
        self.master.title("GUI")

        # allowing the widget to take the full space of the root window
        self.pack(fill=tk.BOTH, expand=1)

        # creating a menu instance
        self.menu = tk.Menu(self.master)
        self.master.config(menu=self.menu)

        # create the file object)
        self.file = tk.Menu(self.menu)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        self.file.add_command(label="Exit", command=self.client_exit_menu_btn)

        #added "file" to our menu
        self.menu.add_cascade(label="File", menu=self.file)

        # create the file object)
        self.edit = tk.Menu(self.menu)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        self.edit.add_command(label="Undo")

        #added "file" to our menu
        self.menu.add_cascade(label="Edit", menu=self.edit)


        self.img_w = 64
        self.img_h = 64

        self.idx_now = 0
        self.idx_max = 400
        self.cols = 12//2
        self.rows = 14//2
        # self.new_labls_amount = 1
        self.new_labls_amount = 12 # 18
        # self.rows = 0
        self.rows_now = 0
        # self.cols_now = self.cols # self.cols


        self.add_new_labl = self.get_func_add_new_labl(self.new_labls_amount)
        self.delete_last_labl = self.get_func_delete_last_labl(self.new_labls_amount)

        self.btnAddLabl = tk.Button(self, text="Add labl", command=self.add_new_labl)#self.get_func_add_new_labl(amount=self.cols))
        self.btnAddLabl.place(x=10, y=20)
        self.btnDeleteLabl = tk.Button(self, text="Delete labl", command=self.delete_last_labl)#self.get_func_delete_last_labl(amount=self.cols))
        self.btnDeleteLabl.place(x=10, y=50)


        self.label_frame = tk.LabelFrame(self, text="My first label frame!", width=400, height=300)
        self.label_frame.place(x=450, y=20)
        self.label_frame.pack_propagate(False)

        self.txt_box = tk.Text(self.label_frame)
        self.txt_box.pack(expand=True, fill='both')

        self.scrollbar = tk.Scrollbar(self.txt_box, command=self.txt_box.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_box.config(yscrollcommand=self.scrollbar.set)


        self.datas = []


        self.frm_width = self.img_w+2
        self.frm_height = self.img_h+2
        self.labl_width = 60
        self.labl_height = 60

        self.label_frame_labls = tk.LabelFrame(self, text="My frame of labels!", width=20+self.cols*self.frm_width, height=self.frm_height*self.rows+24, borderwidth=3, relief="solid")
        self.label_frame_labls.place(x=20, y=100)
        self.label_frame_labls.pack_propagate(False)

        self.canvas = tk.Canvas(self.label_frame_labls)
        # self.canvas = tk.Canvas(self.label_frame_labls, background="#00FF00")
        self.scrollbar_2 = tk.Scrollbar(self.label_frame_labls, orient='vertical')
        
        self.canvas.config(yscrollcommand=self.scrollbar_2.set)
        self.scrollbar_2.config(command=self.canvas.yview)
        
        self.scrollbar_2.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(expand=True, fill='both', side='left')

        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

        self.update()

        width = self.canvas.winfo_width()#-self.scrollbar_2.winfo_width()-1
        print("width: {}".format(width))
        self.interior = tk.Frame(self.canvas, width=width, height=self.frm_height)
        # self.interior = tk.Frame(self.canvas, bg="#FF0000", width=width, height=self.frm_height)
        # self.interior = tk.Frame(self.canvas, bg="#FF0000", width=width, height=(self.rows_now+0)*self.frm_height)
        # self.interior.pack(expand=True, fill=tk.X)
        # self.interior.place(x=1, y=0)

        self.interior_id = self.canvas.create_window(0, 0,
                                                     window=self.interior,
                                                     anchor='nw')

        self.interior.bind("<Configure>", self.onFrameConfigure)

        self.bgs = ["#0000FF", "#00AA33"]
        self.bgs_labl = ["#00FF00", "#3300AA"]
        self.frms = []
        self.labls = []

        self.lablfrm_matrix_lambda_images = LabelFrameCanvasScrollbar(self, "Matrix lambda images", 310, 350, 330, 450, scrbar_x=True, scrbar_y=True)


    def get_func_add_new_labl(self, amount=1):
        def add_new_labl():
            while self.lock.acquire(False):
                time.sleep(0.0001)

            for i in range(0, amount):
                if self.idx_now < self.idx_max:
                    data = Datas(self)

                    data.x = self.idx_now % self.cols
                    data.y = self.idx_now // self.cols
                    data.idx = self.idx_now

                    self.idx_now += 1

                    if data.y+1 != self.rows_now:
                        self.rows_now = data.y+1
                        self.interior.config(height=self.frm_height*self.rows_now)

                    frm = tk.Frame(self.interior, width=self.frm_width, height=self.frm_height, bg=self.bgs[(data.x+data.y)%2])
                    frm.place(x=self.frm_width*data.x, y=self.frm_height*data.y)
                    frm.pack_propagate(False)

                    # self.create_new_image()
                    print("Before create_new_images")
                    data.create_new_images()
                    print("After create_new_images")

                    labl = tk.Label(frm, text="test", bg=self.bgs_labl[(data.x+data.y)%2], image=data.imgtks[-1], borderwidth=0)
                    labl.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
                    labl.bind("<Button-1>", data.label_clicked)

                    data.frm = frm
                    data.labl = labl

                    self.after(0, data.update_frame_random, 0)

                    self.datas.append(data)

            self.lock.release()

        return add_new_labl

    
    def get_func_delete_last_labl(self, amount=1):
        def delete_last_labl():
            self.txt_box.delete("1.0", tk.END)
            for i in range(0, amount):
                if self.idx_now > 0:
                    self.idx_now -= 1

                    x = self.idx_now % self.cols
                    y = self.idx_now // self.cols

                    if x % self.cols == 0:
                        self.rows_now = y
                        self.interior.config(height=self.frm_height*(self.rows_now+0))

                    # self.labls[self.idx_now].destroy()
                    # self.labls.pop(self.idx_now)

                    # self.frms[self.idx_now].destroy()
                    # self.frms.pop(self.idx_now)

                    self.datas[-1].destroy()
                    self.datas.pop()

        return delete_last_labl


    def onFrameConfigure(self, event):
        '''Reset the scroll region to encompass the inner frame'''
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


    def client_exit_menu_btn(self):
        print("Pressed the menu btn button!")
        for p in self.ps:
            p.terminate()
            p.join()

        exit()


if __name__ == "__main__":
    # root window created. Here, that would be the only window, but
    # you can later have windows within windows.
    root = tk.Tk()

    root.geometry("1000x650")

    #creation of an instance
    app = Window(root)

    #mainloop 
    root.mainloop()
