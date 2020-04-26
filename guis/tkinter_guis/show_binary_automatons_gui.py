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
from recordclass import recordclass, RecordClass

from threading import Lock

import platform
print("platform.system(): {}".format(platform.system()))


class FrameOwn(tk.Frame):
 
    def __init__(self, master, width=None, height=None, background=None):
        self.master = master
        tk.Frame.__init__(self, master, class_='FrameOwn', background=background)

        # canvas width and height!
        self.canvas_width = width
        self.canvas_height = height

        self.canvas = tk.Canvas(self, background=background, highlightthickness=0, width=width, height=height)
        self.canvas.grid(row=0, column=0, sticky=tk.N+tk.E+tk.W+tk.S)

        self.yscrollbar = tk.Scrollbar(self, orient=tk.VERTICAL, width=12)
        self.yscrollbar.grid(row=0, column=1,sticky=tk.N+tk.S)
    
        self.canvas.configure(yscrollcommand=self.yscrollbar.set)
        self.yscrollbar['command']=self.canvas.yview

        self.xscrollbar = tk.Scrollbar(self, orient=tk.HORIZONTAL, width=12)
        self.xscrollbar.grid(row=1, column=0, sticky=tk.E+tk.W)
        
        self.canvas.configure(xscrollcommand=self.xscrollbar.set)
        self.xscrollbar['command']=self.canvas.xview

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        
        self.innerframe = tk.Frame(self.canvas, bg=background)
        # self.innerframe = tk.Frame(self.canvas, bg='#0000FF')
        self.innerframe.pack(anchor=tk.N)

        self.pixel = tk.PhotoImage(master=self, name='test_photo', width=1, height=1)
        
        self.canvas.create_window(0, 0, window=self.innerframe, anchor='nw', tags='inner_frame')

        self.update_viewport()

        self.horizontal_factor = 1
        self.vertical_factor = 1
        self._active_area = None

        def _on_mousewheel(event):
            if self._active_area:
                self._active_area.onMouseWheel(event)

        self.bind_all('<4>', _on_mousewheel,  add='+')
        self.bind_all('<5>', _on_mousewheel,  add='+')
        self.add_support_to(self.canvas, xscrollbar=self.xscrollbar, yscrollbar=self.yscrollbar)


    def update_viewport(self):
        self.update()

        window_width = self.innerframe.winfo_reqwidth()
        window_height = self.innerframe.winfo_reqheight()

        self.canvas.configure(scrollregion="0 0 %s %s" % (window_width, window_height), width=self.canvas_width, height=self.canvas_height)

        self['width'] = self.canvas_width+int(self.yscrollbar['width'])
        self['height'] = self.canvas_height+int(self.xscrollbar['width'])


    def add_support_to(self, widget=None, xscrollbar=None, yscrollbar=None, what="units", horizontal_factor=None, vertical_factor=None):
        def _mousewheel_bind(widget):
            self._active_area = widget

        def _mousewheel_unbind():
            self._active_area = None

        def _make_mouse_wheel_handler(widget, orient, factor = 1, what="units"):
            view_command = getattr(widget, orient+'view')
            
            def onMouseWheel(event):
                if event.num == 4:
                    view_command("scroll", (-1)*factor, what)
                elif event.num == 5:
                    view_command("scroll", factor, what) 
                    view_command("scroll", event.delta, what)
        
            return onMouseWheel
        
        if xscrollbar is None and yscrollbar is None:
            return

        if xscrollbar is not None:
            horizontal_factor = horizontal_factor or self.horizontal_factor

            xscrollbar.onMouseWheel = _make_mouse_wheel_handler(widget, 'x', self.horizontal_factor, what)
            xscrollbar.bind('<Enter>', lambda event, scrollbar=xscrollbar: _mousewheel_bind(scrollbar) )
            xscrollbar.bind('<Leave>', lambda event: _mousewheel_unbind())

        if yscrollbar is not None:
            vertical_factor = vertical_factor or self.vertical_factor

            yscrollbar.onMouseWheel = _make_mouse_wheel_handler(widget, 'y', self.vertical_factor, what)
            yscrollbar.bind('<Enter>', lambda event, scrollbar=yscrollbar: _mousewheel_bind(scrollbar) )
            yscrollbar.bind('<Leave>', lambda event: _mousewheel_unbind())

        main_scrollbar = yscrollbar if yscrollbar is not None else xscrollbar
        
        if widget is not None:
            if isinstance(widget, list) or isinstance(widget, tuple):
                list_of_widgets = widget
                for widget in list_of_widgets:
                    widget.bind('<Enter>', lambda event: _mousewheel_bind(widget))
                    widget.bind('<Leave>', lambda event: _mousewheel_unbind())

                    widget.onMouseWheel = main_scrollbar.onMouseWheel
            else:
                widget.bind('<Enter>', lambda event: _mousewheel_bind(widget))
                widget.bind('<Leave>', lambda event: _mousewheel_unbind())

                widget.onMouseWheel = main_scrollbar.onMouseWheel


# Here, we are creating our class, MainWindow, and inheriting from the Frame
# class. Frame is a class from the tkinter module. (see Lib/tkinter/__init__)
class MainWindow(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.master = master
        # self.init_window()

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

        self.i = 0
        self.i2 = 0

        self.width = 500
        self.height = 450

        w = self.master.winfo_screenwidth()
        h = self.master.winfo_screenheight()
        size = (self.width, self.height)
        print("w: {}, h: {}".format(w, h))
        print("size: {}".format(size))
        x = w/2 - size[0]/2
        y = h/2 - size[1]/2
        self.master.geometry("%dx%d+%d+%d" % (size + (x, y)))

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


    def change_label_color(self):
        self.txt1.config(bg=['#00FF80', '#FF00FF'][self.i])
        print("self.i: {}".format(self.i))
        self.i = (self.i+1)%2


    def change_text(self):
        self.txt3.config(text=['A', 'B'][self.i2])
        self.i2 = (self.i2+1)%2


    def init_window(self):
        # self.master.protocol("WM_DELETE_WINDOW", self.master.destroy)
        self.master.protocol("WM_DELETE_WINDOW", self.client_exit_menu_btn)

        self.master.title("GUI")

        # allowing the widget to take the full space of the root window
        self.pack(fill=tk.BOTH, expand=1)

        self.menu = tk.Menu(self.master)
        self.master.config(menu=self.menu)

        self.file = tk.Menu(self.menu, tearoff=False)
        self.file.add_command(label="Exit", command=self.client_exit_menu_btn)

        self.menu.add_cascade(label="File", menu=self.file)

        self.edit = tk.Menu(self.menu, tearoff=False)
        self.edit.add_command(label="Undo")

        self.menu.add_cascade(label="Edit", menu=self.edit)

        self.pixel = tk.PhotoImage(master=self, name='test_photo', width=1, height=1)

        btn_color_normal = '#40FF80'
        btn_color_hover = '#30FFA0'
        # btn_color_hover = '#FF4080'
        # btn_color_hover = '#FF4080'
        kwargs = dict(bg=btn_color_normal, image=self.pixel, compound=tk.CENTER, highlightthickness=0, bd=1, relief=tk.GROOVE)
        # kwargs = dict(bg=btn_color_normal, image=self.pixel, compound=tk.CENTER, highlightthickness=0, bd=1, relief=tk.GROOVE)
        kwargs_config = dict(width=150-4, height=30-4, padx=0, pady=0)

        def create_functions_for_hover_button_color_change(btn):
            def on_enter(e):
                btn['activebackground'] = btn_color_hover
                # print("btn.keys(): {}".format(btn.keys()))

            def on_leave(e):
                # btn['background'] = btn_color_normal
                pass

            return on_enter, on_leave

        self.btn_increase_x_image = tk.Button(self, text='Increase x images', **kwargs)
        # self.btn_increase_x_image = tk.Button(self, text='Increase x images', bg='#40FF80', image=self.pixel, compound=tk.CENTER, highlightthickness=0, bd=1, relief=tk.FLAT)
        self.btn_increase_x_image.config(**kwargs_config)
        # self.btn_increase_x_image.config(width=150-4, height=30-4, padx=0, pady=0)
        self.btn_increase_x_image.place(x=20+300+20, y=20)

        self.btn_decrease_x_image = tk.Button(self, text='Decrease x images', **kwargs)
        self.btn_decrease_x_image.config(**kwargs_config)
        self.btn_decrease_x_image.place(x=20+300+20, y=50)

        self.btn_increase_y_image = tk.Button(self, text='Increase y images', **kwargs)
        self.btn_increase_y_image.config(**kwargs_config)
        self.btn_increase_y_image.place(x=20+300+20, y=80)

        self.btn_decrease_y_image = tk.Button(self, text='Decrease y images', **kwargs)
        self.btn_decrease_y_image.config(**kwargs_config)
        self.btn_decrease_y_image.place(x=20+300+20, y=110)

        on_enter, on_leave = create_functions_for_hover_button_color_change(self.btn_increase_x_image)
        self.btn_increase_x_image.bind("<Enter>", on_enter)
        self.btn_increase_x_image.bind("<Leave>", on_leave)
        on_enter, on_leave = create_functions_for_hover_button_color_change(self.btn_decrease_x_image)
        self.btn_decrease_x_image.bind("<Enter>", on_enter)
        self.btn_decrease_x_image.bind("<Leave>", on_leave)
        on_enter, on_leave = create_functions_for_hover_button_color_change(self.btn_increase_y_image)
        self.btn_increase_y_image.bind("<Enter>", on_enter)
        self.btn_increase_y_image.bind("<Leave>", on_leave)
        on_enter, on_leave = create_functions_for_hover_button_color_change(self.btn_decrease_y_image)
        self.btn_decrease_y_image.bind("<Enter>", on_enter)
        self.btn_decrease_y_image.bind("<Leave>", on_leave)

        # TODOs at 2019.12.06
        # TODO: add buttons for adding in x and y direction more labels with images!
        # TODO: make id modular, so that it is possible to add any content into to label (as an image of course)
        # TODO: create animated images with change able timing too!

        self.frame_own = FrameOwn(self, width=300, height=300)
        self.frame_own.place(x=20, y=10)

        fo = self.frame_own

        # fill self.frame_own with labels!
        fo.colors = ['#00FF80', '#8080FF']
        fo.w = 100
        fo.h = 100
        fo.rows = 0
        fo.cols = 0

        # fo.lst_lbls = []
        # fo.lst_lbls_inner = []
        fo.lst_lbl_image_animation = []
        
        class LabelImageAnimation:
            __slots__ = (
                'idx_i',  'idx_j', 'str_rnd_seed',
                'x_lbl', 'y_lbl', 'w_lbl', 'h_lbl',
                'x_lbl_inner', 'y_lbl_inner', 'w_lbl_inner', 'h_lbl_inner',
                'idx_img', 'l_imgtk', 'l_img', 'lock', 'lbl', 'lbl_inner',
                'is_animation_on', 'time_elapse', 'len_l_imgtk'
            )

            def __init__(self, idx_i, idx_j,  x_lbl,  y_lbl,  w_lbl,  h_lbl, 
                  x_lbl_inner,  y_lbl_inner,  w_lbl_inner,  h_lbl_inner,  idx_img, 
                  l_img, l_imgtk,  lock=None,  lbl=None,  lbl_inner=None, 
                  is_animation_on=None,  time_elapse=None, ):
                self.str_rnd_seed = hex(np.random.randint(0, 0x100000000))[2:].upper()

                self.idx_i: int = idx_i
                self.idx_j: int = idx_j

                self.x_lbl: int = x_lbl
                self.y_lbl: int = y_lbl
                self.w_lbl: int = w_lbl
                self.h_lbl: int = h_lbl

                self.x_lbl_inner: int = x_lbl_inner
                self.y_lbl_inner: int = y_lbl_inner
                self.w_lbl_inner: int = w_lbl_inner
                self.h_lbl_inner: int = h_lbl_inner

                self.idx_img: int = idx_img
                self.l_img: list = l_img
                self.l_imgtk: list = l_imgtk

                self.lock: Lock = None
                if lock is None:
                    self.lock = Lock()
                else:
                    self.lock = lock

                self.lbl: tk.Label = None
                if lbl is not None:
                    self.lbl = lbl_inner

                self.lbl_inner: tk.Label = None
                if lbl_inner is not None:
                    self.lbl_inner = lbl_inner

                self.is_animation_on: bool = False
                if is_animation_on is not None:
                    self.is_animation_on = is_animation_on

                self.time_elapse: int = 100 # ms
                if time_elapse is not None:
                    self.time_elapse = time_elapse

        fo.w_img = 90
        fo.h_img = 80
        fo.x_img = (fo.w-fo.w_img)//2
        fo.y_img = (fo.h-fo.h_img)//2


        def define_new_def_for_print(x, y, o):
            # o...LabelImageAnimation
            def new_def(event):
                print("event: {}, (x, y): {}".format(event, (x, y)))
                with o.lock:
                    if o.is_animation_on:
                        print('Animation is on, will be set to off!')
                    else:
                        print('Animation is off, will be set to on!')

                    o.is_animation_on = not o.is_animation_on
            return new_def


        def create_button_animation_functions(o):
            def do_the_animation():
                print("o.idx_i: {}, o.idx_j: {}, o.idx_img: {}".format(o.idx_i, o.idx_j, o.idx_img))
                lock = o.lock
                lock.acquire()
                is_animation_on = o.is_animation_on
                lock.release()
                if is_animation_on:
                    o.idx_img = (o.idx_img+1)%o.len_l_imgtk
                    o.lbl_inner.configure(image=o.l_imgtk[o.idx_img])
                    o.lbl_inner.image = o.l_imgtk[o.idx_img]
                    o.lbl_inner.after(o.time_elapse, do_the_animation)


            def toggle_animation(event):
                lock = o.lock
                lock.acquire()
                o.is_animation_on = not o.is_animation_on
                if o.is_animation_on:
                    o.lbl_inner.after(0, do_the_animation)
                lock.release()

            return do_the_animation, toggle_animation


        def create_new_label_with_image(i, j):
            o = LabelImageAnimation(
                idx_i=i, idx_j=j,
                x_lbl=i*fo.w, y_lbl=j*fo.h,
                w_lbl=fo.w, h_lbl=fo.h,
                x_lbl_inner=fo.x_img, y_lbl_inner=fo.y_img,
                w_lbl_inner=fo.w_img, h_lbl_inner=fo.h_img,
                idx_img=0, l_imgtk=[], l_img=[],
            )

            l = tk.Label(
                master=fo.innerframe, text='{}'.format(j*i), bg=fo.colors[(i+j)%2],
                width=o.w_lbl, height=o.h_lbl, image=fo.pixel, fg='#000000', compound='c',
                borderwidth=0,
            )
            l.place(x=o.x_lbl, y=o.y_lbl)

            amount_images = 80
            pix = np.random.randint(0, 256, (o.h_lbl_inner, o.w_lbl_inner), dtype=np.uint8)
            pix[-1] = 0
            for i_iter in range(0, amount_images):
                # pix = np.random.randint(0, 256, (o.h_lbl_inner, o.w_lbl_inner), dtype=np.uint8)
                # pix[i_iter] = 0
                pix = np.roll(pix, 1, axis=0)
                img = Image.fromarray(pix)
                o.l_img.append(img)
                imgtk = ImageTk.PhotoImage(img, name='img_{}_{}_{}_{}'.format(o.str_rnd_seed, i_iter, j, i))
                o.l_imgtk.append(imgtk)
            o.len_l_imgtk = len(o.l_imgtk)
            # print("o.len_l_imgtk: {}".format(o.len_l_imgtk))

            bg_li = '#000000'
            li = tk.Label(master=l, text='inner', bg=bg_li, image=o.l_imgtk[0], borderwidth=0)
            # li = tk.Label(master=l, text='inner', bg=bg_li, image=imgtk, borderwidth=0)
            li.place(x=o.x_lbl_inner, y=o.y_lbl_inner)
            # li.bind("<Button-1>", define_new_def_for_print(i, j, o))

            # TODO: add a function for animation on/off!
            do_the_animation, toggle_animation = create_button_animation_functions(o)
            # def change_image(event):
            #     print('Image changed!')
            #     o.lbl_inner.configure(image=o.l_imgtk[0])
            #     o.lbl.update()

            # li.bind("<Button-1>", change_image)
            li.bind("<Button-1>", toggle_animation)

            o.lbl = l
            o.lbl_inner = li

            return o


        def create_new_col_labels():
            if fo.rows==0:
                assert fo.cols==0
                fo.rows = 1
                assert len(fo.lst_lbl_image_animation)==0

                fo.lst_lbl_image_animation.append([])

            i = fo.cols
            for j in range(0, fo.rows):
                o = create_new_label_with_image(i, j)
                fo.lst_lbl_image_animation[j].append(o)

            fo.cols += 1


        def create_new_row_labels():
            if fo.cols==0:
                assert fo.rows==0
                fo.cols = 1
                assert len(fo.lst_lbl_image_animation)==0

            lst_lbl_image_animation_row = []
            j = fo.rows
            for i in range(0, fo.cols):
                o = create_new_label_with_image(i, j)
                lst_lbl_image_animation_row.append(o)
            fo.lst_lbl_image_animation.append(lst_lbl_image_animation_row)

            fo.rows += 1


        def delete_col_labels():
            if fo.cols==0:
                assert fo.rows==0
                assert len(fo.lst_lbl_image_animation)==0
                return

            fo.cols -= 1
            i = fo.cols
            for j in range(0, fo.rows):
                l_row = fo.lst_lbl_image_animation[j]
                o = l_row.pop()
                lock = o.lock
                lock.acquire()
                o.is_animation_on = False
                lock.release()
                del o

            if fo.cols==0:
                fo.rows = 0
                assert np.all(np.array([len(l) for l in fo.lst_lbl_image_animation])==0)
                fo.lst_lbl_image_animation = []


        def delete_row_labels():
            if fo.cols==0:
                assert fo.rows==0
                assert len(fo.lst_lbl_image_animation)==0
                return

            fo.rows -= 1
            j = fo.rows
            l_objects = fo.lst_lbl_image_animation[j]
            for o in fo.lst_lbl_image_animation.pop():
                lock = o.lock
                lock.acquire()
                o.is_animation_on = False
                lock.release()
                del o

            if fo.rows==0:
                fo.cols = 0
                assert len(fo.lst_lbl_image_animation)==0


        def func_btn_create_new_col(event):
            create_new_col_labels()
            fo.innerframe.config(width=fo.w*fo.cols, height=fo.h*fo.rows)
            fo.update_viewport()


        def func_btn_create_new_row(event):
            create_new_row_labels()
            fo.innerframe.config(width=fo.w*fo.cols, height=fo.h*fo.rows)
            fo.update_viewport()


        def func_btn_delete_col(event):
            delete_col_labels()
            if fo.rows==0:
                fo.innerframe.config(width=1, height=1)
            else:
                fo.innerframe.config(width=fo.w*fo.cols, height=fo.h*fo.rows)
            fo.update_viewport()


        def func_btn_delete_row(event):
            delete_row_labels()
            if fo.cols==0:
                fo.innerframe.config(width=1, height=1)
            else:
                fo.innerframe.config(width=fo.w*fo.cols, height=fo.h*fo.rows)
            fo.update_viewport()


        self.btn_increase_x_image.bind('<Button-1>', func_btn_create_new_col)
        self.btn_increase_y_image.bind('<Button-1>', func_btn_create_new_row)
        self.btn_decrease_x_image.bind('<Button-1>', func_btn_delete_col)
        self.btn_decrease_y_image.bind('<Button-1>', func_btn_delete_row)


        for _ in range(0, 4):
            create_new_row_labels()
        
        for _ in range(0, 3-1):
            create_new_col_labels()

        fo.innerframe.config(width=fo.w*fo.cols, height=fo.h*fo.rows)
        fo.update_viewport()

        print("fo.rows: {}".format(fo.rows))
        print("fo.cols: {}".format(fo.cols))

        # 2019.10.15, TODO: add animation of random pictures first!
        # TODO: print the information of the labels info!

        return

        self.startAnimationButton = tk.Button(self, text="Start Animation!", command=self.start_animation)
        self.startAnimationButton.place(x=10, y=40)

        self.stopAnimationButton = tk.Button(self, text="Stop Animation!!!", command=self.stop_animation)
        self.stopAnimationButton.place(x=10, y=70)

        self.sortImagesLengthButton = tk.Button(self, text="Sort Animations by length", command=self.sort_animation_by_length)
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


    def client_exit_menu_btn(self):
        print("Pressed the menu btn button!")
        for i, p in enumerate(self.ps, 0):
            p.terminate()
            p.join()

        self.master.destroy()

    
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
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()
