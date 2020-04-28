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
import tkinter.ttk as ttk

import multiprocessing as mp
from multiprocessing import Process, Pipe # , Lock
from recordclass import recordclass, RecordClass

from threading import Lock

import platform
print("platform.system(): {}".format(platform.system()))

START_ROWS = 2
START_COLS = 1

# class ReadOnlyText(tk.Frame):
class ReadOnlyText(tk.Text):
    def __init__(self, *args, **kwargs):
        self.frame = ttk.Frame(master=kwargs['master'])
        kwargs['master'] = self.frame
        tk.Text.__init__(self, *args, **kwargs)

        def txtEvent(event):
            if(event.state==12 and event.keysym=='c'):
                return
            else:
                return "break"

        self.bind("<Key>", lambda e: txtEvent(e))

        self.frame.grid_propagate(False)
    # implement stretchability
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)

    # create a Text widget
        # self = tk.Text(self)
        self.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

    # create a Scrollbar and associate it with txt
        self.scrollb_x = ttk.Scrollbar(self.frame, orient=tk.HORIZONTAL, command=self.xview)
        self.scrollb_x.grid(row=1, column=0, sticky='nsew')
        self['xscrollcommand'] = self.scrollb_x.set

        self.scrollb_y = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.yview)
        self.scrollb_y.grid(row=0, column=1, sticky='nsew')
        self['yscrollcommand'] = self.scrollb_y.set


def get_new_binary_automaton_images(w, h):
    variables = approx_random_images.get_default_variables()

    # print("Before:")
    # print("variables.height: {}".format(variables.height))
    # print("variables.width: {}".format(variables.width))
    variables.lambdas_in_picture = False
    variables.with_resize_image = False
    
    variables.bits = 8
    # variables.bits = 24
    
    variables.min_or = 2
    variables.max_or = 3

    variables.min_and = 3
    variables.max_and = 3

    variables.min_n = 2
    variables.max_n = 4

    variables.width = w
    variables.height = h
    variables.temp_path_lambda_file = ''
    variables.with_frame = True
    # variables.with_frame = False

    # variables.temp_path_lambda_file = 'lambdas.txt'
    variables.save_data = False
    variables.save_pictures = False
    variables.save_gif = False
    # variables.suffix = '1112'

    variables.image_by_str = '1'
    # variables.image_by_str = '1'
    
    variables.max_it = 100

    # print("After:")
    # print("variables.height: {}".format(variables.height))
    # print("variables.width: {}".format(variables.width))

    dm_params_lambda = approx_random_images.get_dm_params_lambda(variables)
    dm_params = approx_random_images.get_dm_params(variables)

    # dm_params.return_pix_array = False

    # returns = create_bits_neighbour_pictures(dm_params, dm_params_lambda)
    # dm_params, dm_params_lambda = returns
    # globals()['dm_params'] = dm_params
    # globals()['dm_params_lambda'] = dm_params_lambda
    

    dm_params.return_pix_array = True

    returns = approx_random_images.create_bits_neighbour_pictures(dm_params, dm_params_lambda)
    pixs, pixs_bws, pixs_combined, dm_params, dm_params_lambda = returns
    # globals()['pixs'] = pixs
    # globals()['pixs_combined'] = pixs_combined
    # globals()['dm_params'] = dm_params
    # globals()['dm_params_lambda'] = dm_params_lambda

    return pixs, pixs_bws, dm_params, dm_params_lambda


class LabelImageAnimation:
    __slots__ = (
        'idx_i',  'idx_j', 'str_rnd_seed',
        'x_lbl', 'y_lbl', 'w_lbl', 'h_lbl',
        'x_lbl_inner', 'y_lbl_inner', 'w_lbl_inner', 'h_lbl_inner',
        'idx_img', 'l_imgtk', 'l_img', 'lock', 'lbl', 'lbl_inner',
        'is_animation_on', 'time_elapse', 'len_l_imgtk',
        'func_do_the_animation', 'func_toggle_animation', 'func_reset_index',
        'first_duplicates', 'last_duplicates',
        'dm_params', 'dm_params_lambda',
        'pixs', 'pixs_bws',
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

        self.create_button_animation_functions()


    def create_button_animation_functions(self):
        def func_do_the_animation():
            # print("self.idx_i: {}, self.idx_j: {}, self.idx_img: {}".format(self.idx_i, self.idx_j, self.idx_img))
            # lock = self.lock
            # lock.acquire()
            with self.lock:
                is_animation_on = self.is_animation_on
            # lock.release()
            if is_animation_on:
                self.idx_img = (self.idx_img+1)%self.len_l_imgtk
                self.lbl_inner.configure(image=self.l_imgtk[self.idx_img])
                self.lbl_inner.image = self.l_imgtk[self.idx_img]
                self.lbl_inner.after(self.time_elapse, func_do_the_animation)


        def func_toggle_animation(event, state=None):
            # lock = self.lock
            # lock.acquire()
            # print("state: {}".format(state))
            with self.lock:
                if state is None:
                    self.is_animation_on = not self.is_animation_on
                    if self.is_animation_on:
                        self.lbl_inner.after(0, func_do_the_animation)
                elif state==True and not self.is_animation_on:
                    self.is_animation_on = True
                    self.lbl_inner.after(0, func_do_the_animation)
                elif state==False:
                    self.is_animation_on = False

            # lock.release()


        def func_reset_index(event):
            with self.lock:
                self.idx_img = 0
                self.lbl_inner.configure(image=self.l_imgtk[self.idx_img])
                self.lbl_inner.image = self.l_imgtk[self.idx_img]

        self.func_do_the_animation = func_do_the_animation
        self.func_toggle_animation = func_toggle_animation
        self.func_reset_index = func_reset_index


class FrameOwn(tk.Frame):

    def __init__(self, master, width=None, height=None, background=None):
        self.bg_li = '#000000'
        self.l_lbl_colors = ['#00FF80', '#8080FF']

        self.master = master
        kwargs = dict(highlightthickness=0, bd=1, relief=tk.GROOVE)
        # kwargs = dict(image=master.pixel, compound=tk.CENTER, highlightthickness=0, bd=1, relief=tk.GROOVE)
        tk.Frame.__init__(self, master, class_='FrameOwn', background=background, borderwidth=1, **kwargs)

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


    def create_new_label_with_image(self, i, j):
        obj = LabelImageAnimation(
            idx_i=i, idx_j=j,
            x_lbl=i*self.w, y_lbl=j*self.h,
            w_lbl=self.w, h_lbl=self.h,
            x_lbl_inner=self.x_img, y_lbl_inner=self.y_img,
            w_lbl_inner=self.w_img, h_lbl_inner=self.h_img,
            idx_img=0, l_imgtk=[], l_img=[],
        )

        lbl = tk.Label(
            # master=self.innerframe, text='{}'.format(j*i), bg=self.l_lbl_colors[(i+j)%len(self.l_lbl_colors)],
            master=self.innerframe, bg=self.l_lbl_colors[(i+j)%len(self.l_lbl_colors)],
            width=obj.w_lbl, height=obj.h_lbl, image=self.pixel, fg='#000000', compound='c',
            borderwidth=0,
        )
        lbl.place(x=obj.x_lbl, y=obj.y_lbl)


        print('Generate images for (i, j): {}'.format((i, j)))
        pixs, pixs_bws, dm_params, dm_params_lambda = get_new_binary_automaton_images(w=self.w_img, h=self.h_img)

        obj.pixs = np.array(pixs, dtype=np.uint8)
        obj.pixs_bws = np.array(pixs_bws, dtype=np.uint8)

        # duplicate the first pix and the last pix 2 or more times
        first_duplicates = 2
        last_duplicates = 2
        obj.first_duplicates = first_duplicates
        obj.last_duplicates = last_duplicates
        pixs = [pixs[0]]*first_duplicates+pixs+[pixs[-1]]*last_duplicates

        # amount_images = self.h_img
        # amount_images = 80
        # pix = np.random.randint(0, 256, (obj.h_lbl_inner, obj.w_lbl_inner), dtype=np.uint8)
        # pix[-1] = 0
        for i_iter, pix in enumerate(pixs, 0):
        # for i_iter in range(0, amount_images):
            # pix = np.random.randint(0, 256, (obj.h_lbl_inner, obj.w_lbl_inner), dtype=np.uint8)
            # pix[i_iter] = 0
            # pix = np.roll(pix, 1, axis=0)
            img = Image.fromarray(pix)
            obj.l_img.append(img)
            # imgtk = ImageTk.PhotoImage(master=self, image=img, name='img_{}_{}_{}_{}'.format(obj.str_rnd_seed, i_iter, j, i))
            imgtk = ImageTk.PhotoImage(master=self, image=img)
            obj.l_imgtk.append(imgtk)
        obj.len_l_imgtk = len(obj.l_imgtk)

        obj.dm_params = dm_params
        obj.dm_params_lambda = dm_params_lambda

        # bg_li = '#000000'
        # lbl_inner = tk.Label(master=lbl, text='inner', bg=bg_li, image=obj.l_imgtk[0], borderwidth=0)
        lbl_inner = tk.Label(master=lbl, bg=self.bg_li, image=obj.l_imgtk[0], borderwidth=0)
        lbl_inner.place(x=obj.x_lbl_inner, y=obj.y_lbl_inner)
        lbl_inner.bind("<Button-1>", obj.func_toggle_animation)
        lbl_inner.bind("<Button-3>", obj.func_reset_index)

        def set_new_images(event):
            self.master.func_open_second_window(event=event)
            self.master.win2.add_pictures_to_lbl(obj=obj)

        lbl_inner.bind("<Button-2>", set_new_images)

        obj.lbl = lbl
        obj.lbl_inner = lbl_inner

        return obj


    def create_new_col_labels(self):
        if self.rows==0:
            assert self.cols==0
            self.rows = 1
            assert len(self.lst_lbl_image_animation)==0

            self.lst_lbl_image_animation.append([])

        i = self.cols
        for j in range(0, self.rows):
            o = self.create_new_label_with_image(i, j)
            self.lst_lbl_image_animation[j].append(o)

        self.cols += 1


    def create_new_row_labels(self):
        if self.cols==0:
            assert self.rows==0
            self.cols = 1
            assert len(self.lst_lbl_image_animation)==0

        lst_lbl_image_animation_row = []
        j = self.rows
        for i in range(0, self.cols):
            o = self.create_new_label_with_image(i, j)
            lst_lbl_image_animation_row.append(o)
        self.lst_lbl_image_animation.append(lst_lbl_image_animation_row)

        self.rows += 1


    def delete_col_labels(self):
        if self.cols==0:
            assert self.rows==0
            assert len(self.lst_lbl_image_animation)==0
            return

        self.cols -= 1
        i = self.cols
        for j in range(0, self.rows):
            l_row = self.lst_lbl_image_animation[j]
            o = l_row.pop()
            with o.lock:
                o.is_animation_on = False
            del o

        if self.cols==0:
            self.rows = 0
            assert np.all(np.array([len(l) for l in self.lst_lbl_image_animation])==0)
            self.lst_lbl_image_animation = []


    def delete_row_labels(self):
        if self.cols==0:
            assert self.rows==0
            assert len(self.lst_lbl_image_animation)==0
            return

        self.rows -= 1
        j = self.rows
        l_objects = self.lst_lbl_image_animation[j]
        for o in self.lst_lbl_image_animation.pop():
            with o.lock:
                o.is_animation_on = False
            del o

        if self.rows==0:
            self.cols = 0
            assert len(self.lst_lbl_image_animation)==0


    def func_btn_create_new_col(self, event):
        self.create_new_col_labels()
        self.innerframe.config(width=self.w*self.cols, height=self.h*self.rows)
        self.update_viewport()


    def func_btn_create_new_row(self, event):
        self.create_new_row_labels()
        self.innerframe.config(width=self.w*self.cols, height=self.h*self.rows)
        self.update_viewport()


    def func_btn_delete_col(self, event):
        self.delete_col_labels()
        if self.rows==0:
            self.innerframe.config(width=1, height=1)
        else:
            self.innerframe.config(width=self.w*self.cols, height=self.h*self.rows)
        self.update_viewport()


    def func_btn_delete_row(self, event):
        self.delete_row_labels()
        if self.cols==0:
            self.innerframe.config(width=1, height=1)
        else:
            self.innerframe.config(width=self.w*self.cols, height=self.h*self.rows)
        self.update_viewport()


    def func_btn_reset_all_animation(self, event):
        for l_row in self.lst_lbl_image_animation:
            for obj in l_row:
                with obj.lock:
                    obj.is_animation_on = False
                    # obj.func_toggle_animation(state=True, event=None)
                    obj.idx_img = 0
                    obj.lbl_inner.configure(image=obj.l_imgtk[obj.idx_img])
                    obj.lbl_inner.image = obj.l_imgtk[obj.idx_img]


    def func_btn_all_animation_on(self, event):
        for l_row in self.lst_lbl_image_animation:
            for obj in l_row:
                obj.func_toggle_animation(state=True, event=None)


    def func_btn_all_animation_off(self, event):
        for l_row in self.lst_lbl_image_animation:
            for obj in l_row:
                obj.func_toggle_animation(state=False, event=None)


# example window, where this window can only exists once!
class Win2(tk.Frame):
    def __init__(self, master, main_frame, number):
        tk.Frame.__init__(self, master, class_='Win2')
        self.master = master
        self.master.title('Show One Automaton')
        self.master.protocol("WM_DELETE_WINDOW", self.close_window)
        self.main_frame = main_frame
        # self.master.geometry("400x400+200+200")

        geom_master = main_frame.master.geometry()
        wh_s, x_s, y_s = geom_master.split('+')
        xr = int(x_s)
        yr = int(y_s)
        w_s, h_s = wh_s.split('x')
        w = int(w_s)
        h = int(h_s)
        x = main_frame.master.winfo_x()
        y = main_frame.master.winfo_y()

        fo = main_frame.frame_own

        self.resize = 3
        self.w_img_size = fo.w_img*self.resize
        self.h_img_size = fo.h_img*self.resize

        self.master.geometry('{}x{}+{}+{}'.format(
            self.w_img_size+280, self.h_img_size+10,
            (x-xr)*2+xr+w, yr,
        ))

        self.pack(fill=tk.BOTH, expand=1)
        self.master.update()
        self.update()

        # self.frame = tk.Frame(self.master)
        # self.quit = tk.Button(self.frame, text="Exit this window!", command=self.close_window)
        # self.quit.pack()
        # self.frame.pack()

        # self.lbl = tk.Frame(master=self, width=50, height=40, bg='#00FF00')
        # self.lbl = tk.Label(master=self, text='Hello', width=50, height=40, bg='#00FF00')
        
        # print("self.winfo_width(): {}".format(self.winfo_width()))
        # print("self.winfo_height(): {}".format(self.winfo_height()))
        
        self.lbl = tk.Label(master=self,
            compound='c', image=main_frame.pixel, width=self.w_img_size, height=self.h_img_size, bg='#808080', borderwidth=2,
            highlightcolor='#FF0000',
            relief=tk.SOLID, padx=0, pady=0,
        )
        self.lbl.place(x=3, y=3)

        self.lbl.bind('<Button-1>', self.show_next_pic)
        self.lbl.bind('<Button-3>', self.reset_to_first_pic)


        self.ro_txt = ReadOnlyText(master=self, relief=tk.SOLID, bd=1, highlightthickness=0, wrap='none', font=("Monospace", 6))
        self.ro_txt.frame.place(x=self.w_img_size+10, y=3, width=250, height=80)

        self.is_image_loaded = False


    def show_next_pic(self, event):
        if not self.is_image_loaded:
            return

        self.idx_img = (self.idx_img+1)%self.len_l_imgtk
        self.lbl.configure(image=self.l_imgtk[self.idx_img])

    def reset_to_first_pic(self, event):
        if not self.is_image_loaded:
            return

        self.idx_img = 0
        self.lbl.configure(image=self.l_imgtk[self.idx_img])
 

    def close_window(self):
        self.main_frame.win2 = None
        self.master.destroy()


    def add_pictures_to_lbl(self, obj):
        assert isinstance(obj, LabelImageAnimation)

        self.l_img = [img.resize((img.width*self.resize, img.height*self.resize)) for img in obj.l_img[obj.first_duplicates:-obj.last_duplicates]]
        self.l_imgtk = [ImageTk.PhotoImage(master=self, image=img) for img in self.l_img]

        self.idx_img = 0
        self.len_l_imgtk = len(self.l_imgtk)
        self.lbl.configure(image=self.l_imgtk[self.idx_img])

        # self.ro_txt.delete(0, 'end')
        self.ro_txt.delete('1.0', tk.END)
        self.ro_txt.insert(tk.END, '\n'.join(obj.dm_params_lambda.functions_str_lst))

        self.is_image_loaded = True


# Here, we are creating our class, MainWindow, and inheriting from the Frame
# class. Frame is a class from the tkinter module. (see Lib/tkinter/__init__)
class MainWindow(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master, class_='MainWindow')
        self.master = master

        self.cpu_count = mp.cpu_count()
        # print("self.cpu_count: {}".format(self.cpu_count))

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

        self.width = 620
        self.height = 550

        w = self.master.winfo_screenwidth()
        h = self.master.winfo_screenheight()
        size = (self.width, self.height)
        # print("w: {}, h: {}".format(w, h))
        # print("size: {}".format(size))
        x = w/2 - size[0]/2
        y = h/2 - size[1]/2
        self.master.geometry("%dx%d+%d+%d" % (size + (x, y)))

        self.init_window()

        self.win2 = None


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


    def func_do_debug(self, event):
            import pdb; pdb.set_trace()


    def func_open_second_window(self, event):
        if self.win2 is None:
            self.win2 = Win2(master=tk.Toplevel(self.master), main_frame=self, number=3)


    def init_window(self):
        self.pixel = tk.PhotoImage(master=self, name='photo_pixel_1x1', width=1, height=1)
        self.master.protocol("WM_DELETE_WINDOW", self.client_exit_menu_btn)

        self.master.title("Automaton Binary GUI")

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

        # TODOs at 2019.12.06
        # TODO: add buttons for adding in x and y direction more labels with images!
        # TODO: make id modular, so that it is possible to add any content into to label (as an image of course)
        # TODO: create animated images with change able timing too!


        self.frame_own = FrameOwn(self, width=400, height=500)
        self.frame_own.place(x=20, y=10)
        self.frame_own.update()

        globals()['frame_own'] = self.frame_own
        # import pdb; pdb.set_trace()

        fo = self.frame_own

        # fill self.frame_own with labels!

        fo.w = 50
        fo.h = fo.w
        fo.w_img = fo.w-10
        fo.h_img = fo.h-10

        fo.rows = 0
        fo.cols = 0
        fo.x_img = (fo.w-fo.w_img)//2
        fo.y_img = (fo.h-fo.h_img)//2

        fo.lst_lbl_image_animation = []


        self.l_frame_own_btns = []

        x_start = fo.winfo_x()+fo.cget('width')+20
        y_start = fo.winfo_y()
        
        print("x_start: {}".format(x_start))
        print("y_start: {}".format(y_start))

        btn_color_normal = '#40FF80'
        btn_color_hover = '#30FFA0'
        kwargs = dict(master=self, bg=btn_color_normal, image=self.pixel, compound=tk.CENTER, highlightthickness=0, bd=1, relief=tk.GROOVE)
        btn_w = 150
        btn_h = 30
        kwargs_config = dict(width=btn_w-4, height=btn_h-4, padx=0, pady=0)

        l_buttons_attributes = [
            dict(y_space=0, text='Increase x images', function=fo.func_btn_create_new_col),      
            dict(y_space=0, text='Decrease x images', function=fo.func_btn_delete_col),
            dict(y_space=5, text='Increase y images', function=fo.func_btn_create_new_row),
            dict(y_space=0, text='Decrease y images', function=fo.func_btn_delete_row),
            dict(y_space=5, text='All Anim. On', function=fo.func_btn_all_animation_on),
            dict(y_space=0, text='All Anim. Off', function=fo.func_btn_all_animation_off),
            dict(y_space=0, text='Reset All Anim.', function=fo.func_btn_reset_all_animation),
            dict(y_space=10, text='Do Debug', function=self.func_do_debug),
            dict(y_space=10, text='Second Window', function=self.func_open_second_window),
        ]

        def create_functions_for_hover_button_color_change(btn):
            def on_enter(e):
                btn['activebackground'] = btn_color_hover

            def on_leave(e):
                pass

            return on_enter, on_leave


        for idx_btn, d in enumerate(l_buttons_attributes, 0):
            btn = tk.Button(text=d['text'], **kwargs)
            btn.config(**kwargs_config)
            btn.place(x=x_start, y=y_start+btn_h*idx_btn+d['y_space'])

            on_enter, on_leave = create_functions_for_hover_button_color_change(btn)
            btn.bind("<Enter>", on_enter)
            btn.bind("<Leave>", on_leave)

            btn.bind('<Button-1>', d['function'])

            self.l_frame_own_btns.append(btn)


        for _ in range(0, START_ROWS):
            fo.create_new_row_labels()
        
        for _ in range(0, START_COLS-1):
            fo.create_new_col_labels()

        fo.innerframe.config(width=fo.w*fo.cols, height=fo.h*fo.rows)
        fo.update_viewport()

        print("fo.rows: {}".format(fo.rows))
        print("fo.cols: {}".format(fo.cols))


    def client_exit_menu_btn(self):
        print("Pressed the EXIT button or X!")
        for i, p in enumerate(self.ps, 0):
            p.terminate()
            p.join()

        self.master.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindow(master=root)
    root.mainloop()
