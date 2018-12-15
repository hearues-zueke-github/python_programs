#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import functools
import sys

sys.path.append("../../picture_manipulation")
import approx_random_images

from copy import deepcopy

import numpy as np

from PIL import Image, ImageTk

# Simple enough, just import everything from tkinter.
# from tkinter import *
import tkinter as tk


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


    def update(self, ind):
        frame = self.frames[ind]
        ind += 1

        if self.increment_idx:
            ind += self.increment_idx
            self.increment_idx = 0
        # if ind >= len(self.frames):
        #     ind = 0

        if self.decrement_idx:
            ind -= self.decrement_idx
            self.decrement_idx = 0
        # if ind < 0:
        #     ind = len(self.frames)-1
        ind = ind % len(self.frames)
        print("ind: {}".format(ind))

        self.labl.configure(image=frame)
        self.after(100, self.update, ind)

    def update2(self, ind):
        frame = self.frames2[ind]
        ind += 1
        if ind >= len(self.frames2):
            ind = 0
        print("ind: {}".format(ind))

        self.labl2.configure(image=frame)
        self.after(350, self.update2, ind)


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


        self.pix = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        self.img = Image.fromarray(self.pix)
        self.imgtk = ImageTk.PhotoImage(self.img)

        self.background_colors_outer = ["#8020FF", "#2080FF"]
        self.background_colors_inner = ["#4080FF", "#8040FF"]

        self.frame_outer_size = (70, 70)
        self.frame_outer_y_amount = 5
        self.frame_outer_x_amount = 6

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

        self.label_frame_lambdas = tk.LabelFrame(self, text="Lambda functions", width=400, height=150)
        self.label_frame_lambdas.place(x=450, y=20)
        self.label_frame_lambdas.pack_propagate(False)

        self.txt_box_lambdas = tk.Text(self.label_frame_lambdas)
        self.txt_box_lambdas.pack(expand=True, fill='both')

        self.scrollbar_lambdas = tk.Scrollbar(self.txt_box_lambdas, command=self.txt_box_lambdas.yview)
        self.scrollbar_lambdas.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_box_lambdas.config(yscrollcommand=self.scrollbar_lambdas.set)


        self.label_frame = tk.LabelFrame(self, text="My first label frame!", width=300, height=300)
        self.label_frame.place(x=450, y=170)
        self.label_frame.pack_propagate(False)

        self.txt_box = tk.Text(self.label_frame)
        self.txt_box.pack(expand=True, fill='both')

        self.scrollbar = tk.Scrollbar(self.txt_box, command=self.txt_box.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_box.config(yscrollcommand=self.scrollbar.set)


        self.btnFrame = tk.Frame(self, width = 100, height=80)
        self.btnFrame.place(x=450, y=470)
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

        # TODO: when the frame is clicked, the function text should be displayed on the right side too! (in a text box!)


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


    def update_frame_random(self, x, y, idx):
        # print("in update_frame: label: {}".format(label))
        frames = self.all_frames[y][x]
        frame = frames[idx]
        idx += 1
        if idx >= len(frames):
            idx = 0
        self.labls_inner[y][x].configure(image=frame)
        if self.should_stop_animation:
            return

        self.after(200, self.update_frame_random, x, y, idx)


    def stop_animation(self):
        # pixs, dm = approx_random_images.create_1_bit_neighbour_pictures(100, 100, return_pix_array=True, save_pictures=False)
        # print("dm.function_str_lst:\n{}".format(dm.function_str_lst))
        
        if self.is_animation_running and not self.should_stop_animation:
            self.is_animation_running = False
            self.should_stop_animation = True
        

    def get_func_show_lambdas(self, y, x):
        pixs, dm = self.all_datas[y][x]
        function_str_lst = dm.function_str_lst

        def show_lambdas(event):
            self.txt_box_lambdas.delete('1.0', tk.END)
            print("show_lambdas: y: {}, x: {}, event: {}".format(y, x, event))
            self.txt_box_lambdas.insert(tk.END, "\n".join(function_str_lst)+"\n")
            self.txt_box_lambdas.see(tk.END)

        return show_lambdas

# root window created. Here, that would be the only window, but
# you can later have windows within windows.
root = tk.Tk()

root.geometry("880x580")

#creation of an instance
app = Window(root)

#mainloop 
root.mainloop()
