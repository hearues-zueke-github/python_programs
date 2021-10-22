#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import pdb

import numpy as np

from functools import partial

from PIL import Image, ImageTk

try:
    # Tkinter for Python 2.xx
    import Tkinter as tk
except ImportError:
    # Tkinter for Python 3.xx
    import tkinter as tk
 

class FrameOwn(tk.Frame):
 
    def __init__(self, master, width=None, height=None, background=None):
        self.master = master
        tk.Frame.__init__(self, master, class_='FrameOwn', background=background)

        # canvas width and height!
        self.canvas_width = width
        self.canvas_height = height

        self.canvas = tk.Canvas(self, background=background, highlightthickness=0, width=width, height=height)
        # self.canvas.pack()
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

        self.horizontal_factor = 2
        self.vertical_factor = 2
        self._active_area = None

        def _on_mousewheel(event):
            if self._active_area:
                self._active_area.onMouseWheel(event)

        self.bind_all('<4>', _on_mousewheel, add='+')
        self.bind_all('<5>', _on_mousewheel, add='+')
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


def main():
    root = tk.Tk()

    def close():
        print("Application-Shutdown")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)

    root.geometry('+{x}+{y}'.format(x=200, y=150))
    root.geometry('{width}x{height}'.format(width=700, height=550))

    frame_own = FrameOwn(root, width=240, height=300)
    frame_own.place(x=30, y=20)

    # fill frame_own with labels!
    # colors = ['#00FF80', '#8040DF']
    colors = ['#00FF80', '#8080FF']
    w = 60
    h = 60
    rows = 9
    cols = 6
    frame_own.lst_lbls = []
    for j in range(0, rows):
        lst_lbls_row = []
        for i in range(0, cols):
            l = tk.Label(master=frame_own.innerframe, text='{}\nj: {}, i: {}'.format(j*i, j, i), bg=colors[(i+j)%2], width=w, height=h, image=frame_own.pixel, fg='#000000', compound='c')
            l.place(x=i*w, y=j*h)
            lst_lbls_row.append(l)
        frame_own.lst_lbls.append(lst_lbls_row)

    def print_something(event):
        print("print_something: event: {}".format(event))


    frame_own.lst_lbls[0][0].bind('<Button-1>', print_something)

    frame_own.innerframe.config(width=w*cols, height=h*rows)
    frame_own.update_viewport()

    def create_new_frame_cell_click(x, y, cell_width, cell_height, rows, columns):
        # TODO: make is dynamically later!
        frame_width_x = 5
        frame_width_y = 5
        inbetween_space_x = 2
        inbetween_space_y = 2

        frame_own2_width = frame_width_x*2 + (cell_width + inbetween_space_x)*columns - inbetween_space_x
        frame_own2_height = frame_width_y*2 + (cell_height + inbetween_space_y)*rows - inbetween_space_y

        print("frame_own2_width: {}".format(frame_own2_width))
        print("frame_own2_height: {}".format(frame_own2_height))

        frame_own2 = FrameOwn(root, width=frame_own2_width, height=frame_own2_height) # , background='#808080')
        frame_own2.place(x=x, y=y)
        # frame_own2.place(x=30+240+20, y=20)
        arr_field = np.zeros((rows, columns), dtype=np.int64)

        deactive_cell = '#000000'
        active_cell = '#FFFFFF'

        def draw_a_point(event):
            print('Draw a point at: event: {}'.format(event))
            x = event.x
            y = event.y
            print("x: {}".format(x))
            print("y: {}".format(y))

            c_x = (x-frame_width_x) // (cell_width+inbetween_space_x)
            c_y = (y-frame_width_y) // (cell_height+inbetween_space_y)
            print("c_x: {}, c_y: {}".format(c_x, c_y))

            if c_x < 0 or c_y < 0 or c_x >= arr_field.shape[1] or c_y >= arr_field.shape[0]:
                return

            v = arr_field[c_y, c_x]

            print("v: {}".format(v))
            if v == 1:
                arr_field[c_y, c_x] = 0
                c_pos_x = frame_width_x + c_x * (cell_width+inbetween_space_x)
                c_pos_y = frame_width_y + c_y * (cell_height+inbetween_space_y)
                frame_own2.canvas2.create_rectangle(c_pos_x, c_pos_y, c_pos_x+cell_width, c_pos_y+cell_height, fill=deactive_cell, outline="")
            else:
                arr_field[c_y, c_x] = 1
                c_pos_x = frame_width_x + c_x * (cell_width+inbetween_space_x)
                c_pos_y = frame_width_y + c_y * (cell_height+inbetween_space_y)
                frame_own2.canvas2.create_rectangle(c_pos_x, c_pos_y, c_pos_x+cell_width, c_pos_y+cell_height, fill=active_cell, outline="")
            # frame_own2.canvas.create_oval(x, y, x, y, width=0, fill='white')
            # frame_own2.canvas2.create_rectangle(x, y, x+10, y+10, fill="#476042")

        width_canvas2 = frame_own2_width
        height_canvas2 = frame_own2_height

        frame_own2.innerframe.config(width=width_canvas2, height=height_canvas2)
        frame_own2.update_viewport()

        frame_own2.canvas2 = tk.Canvas(frame_own2.innerframe, background='#808080', highlightthickness=0, width=width_canvas2, height=height_canvas2)
        frame_own2.canvas2.place(x=0, y=0)
        frame_own2.canvas2.bind('<Button-1>', draw_a_point)

        frame_own2.xscrollbar.grid_forget()
        frame_own2.yscrollbar.grid_forget()

        for row_idx in range(0, rows):
            for column_idx in range(0, columns):
                x = frame_width_x + (cell_width + inbetween_space_x) * column_idx
                y = frame_width_y + (cell_height + inbetween_space_y) * row_idx

                frame_own2.canvas2.create_rectangle(x, y, x+cell_width, y+cell_height, fill=deactive_cell, outline="")

        frame_own2.update_viewport()

    create_new_frame_cell_click(
        x=30+240+20, y=20,
        cell_width = 20, cell_height = 20,
        rows = 10, columns = 10,
    )

    create_new_frame_cell_click(
        x=30+240+20, y=20+200+50,
        cell_width = 40, cell_height = 40,
        rows = 4, columns = 5,
    )

    def do_debug(event):
        pdb.set_trace()

    pixel = tk.PhotoImage(width=1, height=1)
# button = tk.Button(root, text="", image=pixel, width=100, height=100, compound="c")
    root.button = tk.Button(root, text='debugger', image=pixel, width=100, height=50)
    root.button.place(x=30, y=350)
    root.button.bind('<Button-1>', do_debug)
    root.button.config(text='Test!')

    root.lbl = tk.Label(root.button, text='Yes') # , background='#ff0000')
    root.lbl.place(x=0, y=0)
    root.lbl.update()

    root.lbl.txt_var = tk.StringVar()
    root.lbl.config(textvariable=root.lbl.txt_var)

    def set_new_text(text):
        root.lbl.txt_var.set(text)
        root.lbl.update()

        lbl_x = (root.button.winfo_width()-root.lbl.winfo_width())//2
        lbl_y = (root.button.winfo_height()-root.lbl.winfo_height())//2
        print("lbl_x: {}, lbl_y: {}".format(lbl_x, lbl_y))
        root.lbl.place(x=lbl_x, y=lbl_y)

    set_new_text('Debug')

    root.canvas = tk.Canvas(root, background='#808000', highlightthickness=5, width=150, height=100, highlightbackground="red")
    root.canvas.place(x=530, y=10)
    root.canvas.update()

    def draw_a_pixel_in_canvas(event):
        array = np.random.randint(0, 256, (100, 150), dtype=np.uint8)
        root.canvas.img =  ImageTk.PhotoImage(image=Image.fromarray(array))

        root.canvas.create_image(5, 5, anchor="nw", image=root.canvas.img)
        print("array:\n{}".format(array))

        # for _ in range(0, 10000):
        #     x = np.random.randint(0, 150) + 5
        #     y = np.random.randint(0, 100) + 5
        #     root.canvas.create_rectangle(x, y, x+1, y+1, fill='#000000', outline="", )

    root.canvas.bind('<Button-1>', draw_a_pixel_in_canvas)

    globals()['root'] = root
    globals()['set_new_text'] = set_new_text

    root.mainloop()

    # TODO: use this one for the binary automaton stuff for the future!

  
if __name__ == '__main__':
    main() 
