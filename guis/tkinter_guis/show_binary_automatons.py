#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import functools
import sys
import time

# from ..picture_manipulation import approx_random_images
sys.path.append("../../picture_manipulation")
import approx_random_images

from copy import deepcopy

import numpy as np

from PIL import Image, ImageTk

# Simple enough, just import everything from tkinter.
# from tkinter import *
import tkinter as tk

import multiprocessing as mp
from multiprocessing import Process, Pipe # , Lock

from threading import Lock

import platform
print("platform.system(): {}".format(platform.system()))

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
        self.init_window()

        self.cpu_count = mp.cpu_count()
        print("self.cpu_count: {}".format(self.cpu_count))

        # self.pipes = [Pipe() for _ in range(0, self.cpu_count)]
        self.pipes_main_proc = [Pipe() for _ in range(0, self.cpu_count)]
        self.pipes_proc_main = [Pipe() for _ in range(0, self.cpu_count)]

        self.pipes_proc_recv = [pipe[0] for pipe in self.pipes_main_proc]
        self.pipes_main_send = [pipe[1] for pipe in self.pipes_main_proc]
        
        self.pipes_main_recv = [pipe[0] for pipe in self.pipes_proc_main]
        self.pipes_proc_send = [pipe[1] for pipe in self.pipes_proc_main]
        
        self.lock = Lock()

        self.ps = [Process(target=self.thread_func, args=(self.pipes_proc_recv[i], self.pipes_proc_send[i])) for i in range(0, self.cpu_count)]
        for p in self.ps: p.start()


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


    # def update(self, ind):
    #     frame = self.frames[ind]
    #     ind += 1

    #     if self.increment_idx:
    #         ind += self.increment_idx
    #         self.increment_idx = 0
    #     # if ind >= len(self.frames):
    #     #     ind = 0

    #     if self.decrement_idx:
    #         ind -= self.decrement_idx
    #         self.decrement_idx = 0
    #     # if ind < 0:
    #     #     ind = len(self.frames)-1
    #     ind = ind % len(self.frames)
    #     print("ind: {}".format(ind))

    #     self.labl.configure(image=frame)
    #     self.after(100, self.update, ind)

    # def update2(self, ind):
    #     frame = self.frames2[ind]
    #     ind += 1
    #     if ind >= len(self.frames2):
    #         ind = 0
    #     print("ind: {}".format(ind))

    #     self.labl2.configure(image=frame)
    #     self.after(350, self.update2, ind)


    #Creation of init_window
    def init_window(self):

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


        self.quitButton = tk.Button(self, text="Create Image",command=self.create_new_image)
        self.quitButton.place(x=10, y=10)

        self.startAnimationButton = tk.Button(self, text="Start Animation!",command=self.start_animation)
        self.startAnimationButton.place(x=10, y=40)

        self.stopAnimationButton = tk.Button(self, text="Stop Animation!!!",command=self.stop_animation)
        self.stopAnimationButton.place(x=10, y=70)

        self.sortImagesLengthButton = tk.Button(self, text="Sort Animations by length",command=self.sort_animation_by_length)
        self.sortImagesLengthButton.place(x=100, y=40)


        self.pix = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        self.img = Image.fromarray(self.pix)
        self.imgtk = ImageTk.PhotoImage(self.img)

        self.background_colors_outer = ["#8020FF", "#2080FF"]
        self.background_colors_inner = ["#4080FF", "#8040FF"]

        self.frame_outer_size = (70, 70)
        self.frame_outer_y_amount = 7
        self.frame_outer_x_amount = 8

        self.lablFrames = tk.Frame(self,
            width=self.frame_outer_size[0]*self.frame_outer_x_amount,
            height=self.frame_outer_size[1]*self.frame_outer_y_amount,
            bg="#569803")
        self.lablFrames.place(x=10, y=110)

        lablFrames_outer = []
        labls_inner = []
        for y in range(0, self.frame_outer_y_amount):
            lablFrames_outer_row = []
            labls_inner_row = []
            for x in range(0, self.frame_outer_x_amount):
                color_idx = (y+x) % 2
                w = self.frame_outer_size[0]
                h = self.frame_outer_size[1]
                lablFrame = tk.Frame(self.lablFrames, width=w, height=h, bg=self.background_colors_outer[color_idx])
                lablFrame.place(x=w*x, y=h*y)
                lablFrame.pack_propagate(False)

                labl = tk.Label(lablFrame, bg=self.background_colors_inner[color_idx], borderwidth=0, image=self.imgtk)
                labl.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

                lablFrames_outer_row.append(lablFrame)
                labls_inner_row.append(labl)
            lablFrames_outer.append(lablFrames_outer_row)
            labls_inner.append(labls_inner_row)

        self.lablFrames_outer = lablFrames_outer
        self.labls_inner = labls_inner

        self.is_animation_running = False
        self.should_stop_animation = True

        # self.labl2.bind("<Button-1>", self.print_text2)
        # self.labl2.bind("<Button-2>", self.print_text3)

        # self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.x_pos_frame = 580

        self.label_frame_lambdas = tk.LabelFrame(self, text="Lambda functions", width=400, height=150)
        self.label_frame_lambdas.place(x=self.x_pos_frame, y=20)
        self.label_frame_lambdas.pack_propagate(False)

        self.txt_box_lambdas = tk.Text(self.label_frame_lambdas)
        self.txt_box_lambdas.pack(expand=True, fill='both')

        self.scrollbar_lambdas = tk.Scrollbar(self.txt_box_lambdas, command=self.txt_box_lambdas.yview)
        self.scrollbar_lambdas.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_box_lambdas.config(yscrollcommand=self.scrollbar_lambdas.set)


        self.label_frame = tk.LabelFrame(self, text="My first label frame!", width=300, height=300)
        self.label_frame.place(x=self.x_pos_frame, y=170)
        self.label_frame.pack_propagate(False)

        self.txt_box = tk.Text(self.label_frame)
        self.txt_box.pack(expand=True, fill='both')

        self.scrollbar = tk.Scrollbar(self.txt_box, command=self.txt_box.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_box.config(yscrollcommand=self.scrollbar.set)


        self.btnFrame = tk.Frame(self, width = 100, height=80)
        self.btnFrame.place(x=self.x_pos_frame, y=470)
        self.btnFrame.grid_propagate(False)
        self.btnFrame.columnconfigure(0, weight=1)
        self.btnFrame.rowconfigure(0, weight=1)
        self.btnFrame.rowconfigure(1, weight=1)

        self.btnWrite = tk.Button(self.btnFrame, text="Write to text box") # , command=self.write_to_text_box)
        self.btnWrite.bind("<Button-1>", self.write_to_text_box)
        self.btnWrite.grid(row=0, column=0, sticky="news")

        self.btnDelete = tk.Button(self.btnFrame, text="Delete from\ntext box") # , command=self.write_to_text_box)
        self.btnDelete.bind("<Button-1>", self.delete_from_text_box)
        self.btnDelete.grid(row=1, column=0, sticky="news")



    def write_to_text_box(self, event):
        print("event: {}".format(event))
        self.txt_box.insert(tk.END, "TEST! {:03}\n".format(np.random.randint(0, 1000)))
        self.txt_box.see(tk.END)


    def delete_from_text_box(self, event):
        print("event: {}".format(event))
        self.txt_box.delete('1.0', tk.END)

    
    def print_text2(self, event):
        self.increment_idx += 1
        print("Hello button-1!!! It is a label!")
        print("event: {}".format(event))

    def print_text3(self, event):
        self.decrement_idx += 1
        print("Hello button-2!!! It is a label!!!!!")
        print("event: {}".format(event))

    def client_exit_menu_btn(self):
        print("Pressed the menu btn button!")
        for p in self.ps:
            p.terminate()
            p.join()

        exit()


    # def client_exit_btn(self):
    def create_new_image(self):
        print("Pressed 'create_new_image'!")

    
    def start_animation(self):
        if self.should_stop_animation and not self.is_animation_running:
            self.txt_box.delete('1.0', tk.END)

            self.should_stop_animation = False
            self.is_animation_running = True
            print("start animation!")

            rows = len(self.labls_inner)
            cols = len(self.labls_inner[0])
            all_frames = [[None for _ in range(0, cols)] for _ in range(0, rows)]
            all_datas = [[None for _ in range(0, cols)] for _ in range(0, rows)]
            all_lens = [[None for _ in range(0, cols)] for _ in range(0, rows)]

            self.index_table = np.zeros((rows, cols), dtype=np.uint32)+np.arange(0, cols)+(np.arange(0, rows)*cols).reshape((-1, 1))
            print("self.index_table:\n{}".format(self.index_table))
            # sys.exit(-1)

            arr_y = np.zeros((rows, cols), dtype=np.uint32)+np.arange(0, rows).reshape((-1, 1))
            arr_x = np.zeros((rows, cols), dtype=np.uint32)+np.arange(0, cols).reshape((1, -1))
            print("arr_y:\n{}".format(arr_y))
            print("arr_x:\n{}".format(arr_x))
            
            self.arr_idx = np.dstack((arr_y, arr_x)).astype(np.uint32).view("u4,u4").reshape((-1, )) # .reshape((rows, cols))
            print("self.arr_idx:\n{}".format(self.arr_idx))
            
            # arr_idx_2 = arr_idx_1.reshape((-1, 2))
            # print("arr_idx_2:\n{}".format(arr_idx_2))
            
            # arr_idx = arr_idx_2.view("u4,u4")#.reshape((rows, cols))
            # print("arr_idx:\n{}".format(arr_idx))
            # sys.exit(-1)

            self.all_frames = all_frames
            self.all_datas = all_datas
            self.all_lens = all_lens

            def get_pixs_frames(y, x):
                tries = 0
                while True:
                    variables = approx_random_images.get_default_variables()

                    print("Before:")
                    print("variables.height: {}".format(variables.height))
                    print("variables.width: {}".format(variables.width))
                    variables.lambdas_in_picture = False
                    variables.with_resize_image = False
                    variables.bits = 1
                    variables.min_or = 2
                    variables.max_or = 2
                    variables.width = 64
                    variables.height = 64
                    variables.temp_path_lambda_file = 'lambdas.txt'
                    variables.save_pictures = False
                    variables.suffix = ' '

                    print("After:")
                    print("variables.height: {}".format(variables.height))
                    print("variables.width: {}".format(variables.width))

                    dm_params_lambda = approx_random_images.get_dm_params_lambda(variables)
                    dm_params = approx_random_images.get_dm_params(variables)

                    returns = approx_random_images.create_bits_neighbour_pictures(dm_params, dm_params_lambda)
                    pixs, _, dm, _ = returns
                    # pixs, pixs_combined, dm_params, dm_params_lambda = returns
                    # pixs, dm = approx_random_images.create_1_bit_neighbour_pictures(64, 64, return_pix_array=True, save_pictures=False)
                    # approx_random_images
                    # pixs, dm = approx_random_images.create_1_bit_neighbour_pictures(64, 64, return_pix_array=True, save_pictures=False)
                    if len(pixs) >= 5:
                        break
                    
                    tries += 1
                    if tries > 5:
                        break

                print("y: {}, x: {}".format(y, x))
                print("dm.function_str_lst:\not{}".format(dm.function_str_lst))

                arr_zero = np.zeros(pixs[0].shape, dtype=np.uint8)

                pixs = [arr_zero+np.array([0x00, 0xFF, 0x00], dtype=np.uint8)]*1+\
                       pixs+\
                       [arr_zero+np.array([0xFF, 0x00, 0x00], dtype=np.uint8)]*1

                

                return y, x, pixs, dm


            y_x_lst = []
            for y, labls_inner_row in enumerate(self.labls_inner):
                for x, labl in enumerate(labls_inner_row):
                    y_x_lst.append((y, x))

            print("len(y_x_lst): {}".format(len(y_x_lst)))
            get_pixs_frames_enc = dill.dumps(get_pixs_frames)
            # sys.exit(-1)

            def func_send(idx, y, x):
                print("idx: {}, y: {}, x: {}".format(idx, y, x))
                self.pipes_main_send[idx].send((get_pixs_frames_enc, (y, x)))

            def func_recv(idx):
                recv = self.pipes_main_recv[idx%self.cpu_count]
                while not recv.poll():
                    time.sleep(0.01)
                    print("SLEEP!")
                print("NOICE!!!!")
                
                yi, xi, pixs, dm = recv.recv()

                frames = [ImageTk.PhotoImage(
                    Image.fromarray(pix),
                    format = 'gif -index %i' %(i))
                    for i, pix in enumerate(pixs)
                ]
                
                all_frames[yi][xi] = frames
                all_datas[yi][xi] = (pixs, dm)
                all_lens[yi][xi] = len(pixs)-2

                self.labls_inner[yi][xi].bind("<Button-1>", self.get_func_show_lambdas(idx))
                # self.labls_inner[yi][xi].bind("<Button-1>", self.get_func_show_lambdas(yi, xi))
                # labl.bind("<Button-1>", self.get_func_show_lambdas(y, x))

                self.after(0, self.update_frame_random, yi*cols+xi, 0)
                # self.after(0, self.update_frame_random, xi, yi, 0)

                # time.sleep(0.5)
                # print("WAIT!!!")

                str_line = "yi: {:02}, xi: {:02}, amount frames: {:3}\n".format(yi, xi, self.all_lens[yi][xi])
                self.txt_box.insert(tk.END, str_line)
                self.txt_box.see(tk.END)

            cnum = self.cpu_count
            for idx, (y, x) in enumerate(y_x_lst[:cnum], 0):
                func_send(idx%cnum, y, x)
            for idx, (y, x) in enumerate(y_x_lst[cnum:], 0):
                func_recv(idx)
                func_send(idx%cnum, y, x)
            for idx, _ in enumerate(y_x_lst[-cnum:], len(y_x_lst) - cnum):
                func_recv(idx)

            self.all_lens = np.array(all_lens)
            self.idxs = np.arange(0, np.multiply.reduce(self.all_lens.shape))


    def start_animation_old(self):
        if self.should_stop_animation and not self.is_animation_running:
            self.txt_box.delete('1.0', tk.END)

            self.should_stop_animation = False
            self.is_animation_running = True
            print("start animation!")

            all_frames_1d = []
            all_datas_1d = []
            all_lens_1d = []
            for y, _ in enumerate(self.labls_inner):
                for x, _ in enumerate(self.labls_inner[0]):
                    tries = 0
                    while True:
                        pixs, dm = approx_random_images.create_1_bit_neighbour_pictures(64, 64, return_pix_array=True, save_pictures=False)
                        if len(pixs) >= 5:
                            break
                        
                        tries += 1
                        if tries > 5:
                            break

                    print("y: {}, x: {}".format(y, x))
                    print("dm.function_str_lst:\n{}".format(dm.function_str_lst))

                    arr_zero = np.zeros(pixs[0].shape, dtype=np.uint8)

                    pixs = [arr_zero+np.array([0x00, 0xFF, 0x00], dtype=np.uint8)]*1+\
                           pixs+\
                           [arr_zero+np.array([0xFF, 0x00, 0x00], dtype=np.uint8)]*1

                    frames = [ImageTk.PhotoImage(
                        Image.fromarray(pix),
                        format = 'gif -index %i' %(i))
                        for i, pix in enumerate(pixs)
                    ]

                    all_frames_1d.append(frames)
                    all_datas_1d.append((pixs, dm))
                    all_lens_1d.append(len(pixs)-2)


            print("all_lens_1d: {}".format(all_lens_1d))
            arr = np.vstack((all_lens_1d, np.arange(0, len(all_lens_1d)))).T.astype(np.uint32).reshape((-1, )).view("u4,u4")
            print("arr:\n{}".format(arr))
            arr_sort = np.sort(arr, order=["f0", "f1"])
            print("arr_sort:\n{}".format(arr_sort))

            arr_sort_view = arr_sort.view("u4").reshape((-1, 2)).T
            lens_sorted = arr_sort_view[0]
            idxs = arr_sort_view[1]
            print("lens_sorted: {}".format(lens_sorted))
            print("idxs: {}".format(idxs))


            all_frames_1d_sorted = []
            all_datas_1d_sorted = []
            all_lens_1d_sorted = []

            for idx in idxs:
                all_frames_1d_sorted.append(all_frames_1d[idx])
                all_datas_1d_sorted.append(all_datas_1d[idx])
                all_lens_1d_sorted.append(all_lens_1d[idx])

            all_frames_1d = all_frames_1d_sorted
            all_datas_1d = all_datas_1d_sorted
            all_lens_1d = all_lens_1d_sorted


            all_frames = []
            all_datas = []
            all_lens = []

            row_amount = len(self.labls_inner)
            row_len = len(self.labls_inner[0])

            for y in range(0, row_amount):
                all_frames.append(all_frames_1d[row_len*y:row_len*(y+1)])
                all_datas.append(all_datas_1d[row_len*y:row_len*(y+1)])
                all_lens.append(all_lens_1d[row_len*y:row_len*(y+1)])

            self.all_frames = all_frames
            self.all_datas = all_datas
            self.all_lens = all_lens

            for y, labls_inner_row in enumerate(self.labls_inner, 0):
                for x, labl in enumerate(labls_inner_row, 0):
                    labl.bind("<Button-1>", self.get_func_show_lambdas(y, x))

                    self.after(0, self.update_frame_random, x, y, 0)

                    str_line = "y: {:02}, x: {:02}, amount frames: {:3}\n".format(y, x, self.all_lens[y][x])
                    self.txt_box.insert(tk.END, str_line)
                    self.txt_box.see(tk.END)

            # now sort the created index table by the length!


    def update_frame_random(self, idx_y_x, idx):
        # print("in update_frame: label: {}".format(label))
        # print("idx_y_x: {}".format(idx_y_x))
        # idxs = self.arr_idx[idx_y_x]
        while self.lock.acquire(False):
            time.sleep(0.0001)

        y, x = self.arr_idx[idx_y_x]
        # print("idxs: {}".format(idxs))
        # y = idxs[0]
        # x = idxs[1]

        frames = self.all_frames[y][x]
        frame = frames[idx]

        self.lock.release()
        
        idx += 1
        if idx >= len(frames):
            idx = 0
        self.labls_inner[y][x].configure(image=frame)
        if self.should_stop_animation:
            return

        self.after(200, self.update_frame_random, idx_y_x, idx)


    def stop_animation(self):
        # pixs, dm = approx_random_images.create_1_bit_neighbour_pictures(100, 100, return_pix_array=True, save_pictures=False)
        # print("dm.function_str_lst:\n{}".format(dm.function_str_lst))
        
        if self.is_animation_running and not self.should_stop_animation:
            self.is_animation_running = False
            self.should_stop_animation = True

        # def f(a, b):
        #     return a, b, a+b, a-b, a*b

        # for pipe_send in self.pipes_main_send:
        #     a = np.random.randint(-10, 10)
        #     b = np.random.randint(-10, 10)
        #     print("a: {}, b: {}".format(a, b))

        #     pipe_send.send(dill.dumps((f, (a, b))))

        # for pipe_recv in self.pipes_main_recv:
        #     ret = pipe_recv.recv()
        #     print("ret: {}".format(ret))


    def sort_animation_by_length(self):
        print("self.all_lens:\n{}".format(self.all_lens))
        print("self.arr_idx:\n{}".format(self.arr_idx))

        arr = np.vstack((self.all_lens.reshape((-1, )), np.arange(0, np.multiply.reduce(self.all_lens.shape)))).T.astype(np.uint32).reshape((-1, )).view("u4,u4")
        print("arr:\n{}".format(arr))
        arr_sorted = np.sort(arr, order=("f0")).view("u4").reshape((-1, 2)).T
        new_all_lens = arr_sorted[0]
        idxs = arr_sorted[1]
        print("new_all_lens: {}".format(new_all_lens))
        print("idxs: {}".format(idxs))

        self.txt_box.insert(tk.END, "\n")

        rows, cols = self.all_lens.shape
        all_lens = self.all_lens.copy().reshape((-1, ))[idxs].reshape((rows, cols))

        while self.lock.acquire(False):
            time.sleep(0.0001)

        new_arr_idx = self.arr_idx[idxs].copy()
        for i in range(0, new_arr_idx.shape[0]):
            y = i // cols
            x = i % cols
            self.all_lens[y, x] = all_lens[y, x]
            
            str_line = "y: {:02}, x: {:02}, amount frames: {:3}\n".format(y, x, self.all_lens[y, x])
            self.txt_box.insert(tk.END, str_line)
            self.txt_box.see(tk.END)
            
            # if i > 0:
            #     continue

            self.arr_idx[i] = new_arr_idx[i]
        self.lock.release()


    # def get_func_show_lambdas(self, y, x):
    def get_func_show_lambdas(self, idx):
        def show_lambdas(event):
            while self.lock.acquire(False):
                time.sleep(0.0001)

            y, x = self.arr_idx[idx]

            pixs, dm = self.all_datas[y][x]
            function_str_lst = dm.function_str_lst

            self.lock.release()

            self.txt_box_lambdas.delete('1.0', tk.END)
            print("show_lambdas: y: {}, x: {}, event: {}".format(y, x, event))
            self.txt_box_lambdas.insert(tk.END, "\n".join(function_str_lst)+"\n")
            self.txt_box_lambdas.insert(tk.END, "\nidx: {}, y: {}, x: {}\n".format(idx, y, x))
            self.txt_box_lambdas.see(tk.END)

        return show_lambdas

if __name__ == "__main__":
    # root window created. Here, that would be the only window, but
    # you can later have windows within windows.
    root = tk.Tk()

    root.geometry("1000x630")

    #creation of an instance
    app = Window(root)

    #mainloop
    root.mainloop()
