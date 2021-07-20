#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import pdb

import numpy as np

from pprint import pprint

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

class ButtonOwn(tk.Button):
    def __init__(self, master, x, y, width, height, text, background=None):
        self.master = master

        self.pixel = tk.PhotoImage(width=1, height=1)
        tk.Button.__init__(self, master, width=width, height=height, image=self.pixel, background=background)
        # tk.Button.__init__(self, master, class_='ButtonOwn', width=width, height=height, image=self.pixel, background=background)
        self.place(x=x, y=y)
        self.update()

        self.lbl = tk.Label(self) # , background='#ff0000')
        self.lbl.place(x=0, y=0)
        self.lbl.update()

        self.txt_var = tk.StringVar()
        self.lbl.config(textvariable=self.txt_var)

        self.set_new_text(text=text)

    def set_new_text(self, text):
        self.txt_var.set(text)
        self.lbl.update()

        lbl_x = (self.winfo_width()-self.lbl.winfo_width()) // 2
        lbl_y = (self.winfo_height()-self.lbl.winfo_height()) // 2
        self.lbl.place(x=lbl_x, y=lbl_y)


def main():
    root = tk.Tk()

    def close():
        print("Application-Shutdown")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)

    root.geometry('+{x}+{y}'.format(x=200, y=150))
    root.geometry('{width}x{height}'.format(width=680, height=550))

    def create_new_frame_cell_click(root, x, y, cell_width, cell_height, rows, columns):
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
        arr_field = np.zeros((rows, columns), dtype=np.int64) + 1

        deactive_cell = '#000000'
        neutral_cell = '#808080'
        active_cell = '#FFFFFF'

        def get_c_y_c_x_vals(event):
            print('Draw a point at: event: {}'.format(event))
            y = event.y
            x = event.x
            print("y: {}".format(y))
            print("x: {}".format(x))

            c_y = (y-frame_width_y) // (cell_height+inbetween_space_y)
            c_x = (x-frame_width_x) // (cell_width+inbetween_space_x)
            print("(c_y, c_x): {}".format((c_y, c_x)))

            if c_x < 0 or c_y < 0 or c_x >= arr_field.shape[1] or c_y >= arr_field.shape[0]:
                return None, None

            return c_y, c_x

        def get_new_function(func, arr_field):
            def function(event):
                c_y, c_x = get_c_y_c_x_vals(event)
                if c_x is None:
                    return

                func(c_y, c_x)
            return function

        def add_one(c_y, c_x): # event):
            # c_y, c_x = get_c_y_c_x_vals(event)
            # if c_x is None:
            #     return

            v = arr_field[c_y, c_x]
            print("add_one to v: {}".format(v))

            c_pos_x = frame_width_x + c_x * (cell_width+inbetween_space_x)
            c_pos_y = frame_width_y + c_y * (cell_height+inbetween_space_y)
            if v == 0:
                arr_field[c_y, c_x] = 1
                color = neutral_cell
            elif v == 1:
                arr_field[c_y, c_x] = 2
                color = active_cell
            else:
                return

            frame_own2.canvas2.create_rectangle(c_pos_x, c_pos_y, c_pos_x+cell_width, c_pos_y+cell_height, fill=color, outline="")

        def sub_one(c_y, c_x): # event):
            # c_y, c_x = get_c_y_c_x_vals(event)
            # if c_x is None:
            #     return

            v = arr_field[c_y, c_x]
            print("sub_one to v: {}".format(v))

            c_pos_x = frame_width_x + c_x * (cell_width+inbetween_space_x)
            c_pos_y = frame_width_y + c_y * (cell_height+inbetween_space_y)
            if v == 2:
                arr_field[c_y, c_x] = 1
                color = neutral_cell
            elif v == 1:
                arr_field[c_y, c_x] = 0
                color = deactive_cell
            else:
                return

            frame_own2.canvas2.create_rectangle(c_pos_x, c_pos_y, c_pos_x+cell_width, c_pos_y+cell_height, fill=color, outline="")

        frame_own2.add_one = add_one
        frame_own2.sub_one = sub_one

        width_canvas2 = frame_own2_width
        height_canvas2 = frame_own2_height

        frame_own2.innerframe.config(width=width_canvas2, height=height_canvas2)
        frame_own2.update_viewport()

        frame_own2.canvas2 = tk.Canvas(frame_own2.innerframe, background='#7ac918', highlightthickness=0, width=width_canvas2, height=height_canvas2)
        frame_own2.canvas2.place(x=0, y=0)
        frame_own2.canvas2.bind('<Button-1>', get_new_function(add_one, arr_field))
        frame_own2.canvas2.bind('<Button-3>', get_new_function(sub_one, arr_field))

        frame_own2.xscrollbar.grid_forget()
        frame_own2.yscrollbar.grid_forget()

        y_mid = rows // 2
        x_mid = columns // 2

        y_corner = frame_width_y + (cell_height+inbetween_space_y) * y_mid - inbetween_space_y
        x_corner = frame_width_x + (cell_width+inbetween_space_x) * x_mid - inbetween_space_x

        frame_own2.canvas2.create_rectangle(
            x_corner, y_corner,
            x_corner+cell_width+inbetween_space_x*2,
            y_corner+cell_height+inbetween_space_y*2,
            fill='#ffc800', outline="",
        )
        
        for row_idx in range(0, rows):
            for column_idx in range(0, columns):
                x = frame_width_x + (cell_width + inbetween_space_x) * column_idx
                y = frame_width_y + (cell_height + inbetween_space_y) * row_idx

                frame_own2.canvas2.create_rectangle(x, y, x+cell_width, y+cell_height, fill=neutral_cell, outline="")

        frame_own2.update_viewport()

        frame_own2.arr_field = arr_field

        return frame_own2

    # root.frame_own_1 = create_new_frame_cell_click(
    #     root=root,
    #     x=30+240+20, y=20,
    #     cell_width=20, cell_height=20,
    #     rows=3, columns=3,
    # )

    # root.frame_own_2 = create_new_frame_cell_click(
    #     root=root,
    #     x=30+240+20, y=20+200+50,
    #     cell_width=20, cell_height=20,
    #     rows=5, columns=5,
    # )

    def do_debug(event):
        pdb.set_trace()

    root.pixel = tk.PhotoImage(width=1, height=1)
# button = tk.Button(root, text="", image=pixel, width=100, height=100, compound="c")
    root.button = tk.Button(root, text='debugger', image=root.pixel, width=100, height=50)
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

    root.canvas_framethickness = 5
    root.canvas_width = 150
    root.canvas_height = 100
    root.canvas_x_start = 10
    root.canvas_y_start = 10

    root.l_canvas = []
    for y in range(0, 4):
        l = []
        root.l_canvas.append(l)
        for x in range(0,3):
            canvas = tk.Canvas(root, background='#808000', highlightthickness=root.canvas_framethickness, width=root.canvas_width, height=root.canvas_height, highlightbackground="#C08000")
            canvas.place(
                x=root.canvas_x_start+(root.canvas_width+root.canvas_framethickness)*x,
                y=root.canvas_y_start+(root.canvas_height+root.canvas_framethickness)*y,
            )
            canvas.update()
            l.append(canvas)

            canvas.arr_bit_canvas = np.zeros((root.canvas_height, root.canvas_width), dtype=np.uint8)
            canvas.arr_bit_canvas[:] = np.random.randint(0, 2, canvas.arr_bit_canvas.shape)

            canvas.img =  ImageTk.PhotoImage(image=Image.fromarray(canvas.arr_bit_canvas * 255))
            canvas.create_image(5, 5, anchor="nw", image=canvas.img)

            canvas.pos_x = x
            canvas.pos_y = y

            def bind_command(canvas):
                def func_draw_a_new_image_in_canvas(event):
                    canvas.arr_bit_canvas[:] = np.random.randint(0, 2, canvas.arr_bit_canvas.shape, dtype=np.uint8)
                    canvas.img =  ImageTk.PhotoImage(image=Image.fromarray(canvas.arr_bit_canvas * 255))

                    canvas.create_image(5, 5, anchor="nw", image=canvas.img)
                    print("canvas.pos_y: {}, canvas.pos_x: {}".format(canvas.pos_y, canvas.pos_x))
                    print("canvas.arr_bit_canvas:\n{}".format(canvas.arr_bit_canvas))

                canvas.bind('<Button-1>', func_draw_a_new_image_in_canvas)
                canvas.func_draw_a_new_image_in_canvas = func_draw_a_new_image_in_canvas
            bind_command(canvas)

    root.btn_x_start = 490
    root.btn_y_start = 10
    root.btn_width = 150
    root.btn_height = 50
    root.btn_color = '#008000'
    root.btn_add_new_cell_rule = ButtonOwn(
        master=root,
        x=root.btn_x_start,
        y=root.btn_y_start+root.btn_height*0,
        width=root.btn_width,
        height=root.btn_height,
        text='Add Cell Rule',
        background=root.btn_color,
    )
    root.btn_remove_cell_rule = ButtonOwn(
        master=root,
        x=root.btn_x_start,
        y=root.btn_y_start+root.btn_height*1,
        width=root.btn_width,
        height=root.btn_height,
        text='Remove Cell Rule',
        background=root.btn_color,
    )
    root.btn_execute_cell_rule = ButtonOwn(
        master=root,
        x=root.btn_x_start,
        y=root.btn_y_start+root.btn_height*2,
        width=root.btn_width,
        height=root.btn_height,
        text='Execute Cell rule',
        background=root.btn_color,
    )
    root.btn_randomize_all_images = ButtonOwn(
        master=root,
        x=root.btn_x_start,
        y=root.btn_y_start+root.btn_height*3,
        width=root.btn_width,
        height=root.btn_height,
        text='Randomize All Images',
        background=root.btn_color,
    )

    root.cell_rule_width = 74
    root.cell_rule_height = 74
    root.cell_rule_max_amount = 7

    def func_add_new_cell_rule(event):
        print('Added new cell rule!')
        frame_own_cell_rule = create_new_frame_cell_click(
            root=root.cell_rules_frame.innerframe,
            x=root.x_cell_rule, y=0,
            cell_width=20, cell_height=20,
            rows=3, columns=3,
        )
        root.x_cell_rule += root.cell_rule_width
        root.cell_rules_frame.l_cell_rule.append(frame_own_cell_rule)

        length = len(root.cell_rules_frame.l_cell_rule)
        if length > root.cell_rule_max_amount:
            root.cell_rules_frame.innerframe.config(width=length*root.cell_rule_width)
            root.cell_rules_frame.update_viewport()

    def func_remove_cell_rule(event):
        print('Removed cell rule!')
        if len(root.cell_rules_frame.l_cell_rule) > 1:
            root.x_cell_rule -= root.cell_rule_width
            root.cell_rules_frame.l_cell_rule.pop().destroy()

            length = len(root.cell_rules_frame.l_cell_rule)
            if length > root.cell_rule_max_amount:
                root.cell_rules_frame.innerframe.config(width=length*root.cell_rule_width)
            else:
                root.cell_rules_frame.innerframe.config(width=root.cell_rule_max_amount*root.cell_rule_width)
            root.cell_rules_frame.update_viewport()

    def func_execute_cell_rule(event):
        l_arr_field = root.get_all_arr_field()
        print('l_arr_field:')
        pprint(l_arr_field)
        
        if all([np.all(np.all(arr == 1)) for arr in l_arr_field]):
            return
        
        for l_canvas_row in root.l_canvas:
            for canvas in l_canvas_row:
                arr_bit_canvas_bool = canvas.arr_bit_canvas.astype(np.bool)
                arr_bit_canvas_or = np.zeros(arr_bit_canvas_bool.shape, dtype=np.uint8)
                arr_bit_canvas_and = np.zeros(arr_bit_canvas_bool.shape, dtype=np.uint8)
                
                d_pix_canvas = {}
                for dy in range(0, 3):
                    for dx in range(0, 3):
                        d_pix_canvas[(dy, dx)] = np.roll(np.roll(arr_bit_canvas_bool, dx-1, 1), dy-1, 0)

                arr_bit_canvas_or[:] = 0
                for arr_field in l_arr_field:
                    if np.any(arr_field != 1):
                        arr_bit_canvas_and[:] = 1
                        for dy in range(0, 3):
                            for dx in range(0, 3):
                                v = arr_field[dy, dx]
                                if v == 1:
                                    continue
                                elif v == 0:
                                    arr_bit_canvas_and &= d_pix_canvas[(dy, dx)] == False
                                elif v == 2:
                                    arr_bit_canvas_and &= d_pix_canvas[(dy, dx)] == True
                                else:
                                    assert False
                        arr_bit_canvas_or |= arr_bit_canvas_and

                canvas.arr_bit_canvas[:] = arr_bit_canvas_or

                canvas.img =  ImageTk.PhotoImage(image=Image.fromarray(canvas.arr_bit_canvas*255))
                canvas.create_image(5, 5, anchor="nw", image=canvas.img)

    def func_randomize_all_images(event):
        for l_canvas_row in root.l_canvas:
            for canvas in l_canvas_row:
                canvas.func_draw_a_new_image_in_canvas(None)

    root.btn_add_new_cell_rule.bind('<Button-1>', func_add_new_cell_rule)
    root.btn_remove_cell_rule.bind('<Button-1>', func_remove_cell_rule)
    root.btn_execute_cell_rule.bind('<Button-1>', func_execute_cell_rule)
    root.btn_randomize_all_images.bind('<Button-1>', func_randomize_all_images)

    root.bind('<space>', func_execute_cell_rule)

    root.cell_rules_frame = FrameOwn(root, width=root.cell_rule_max_amount*root.cell_rule_width, height=root.cell_rule_height)
    root.cell_rules_frame.place(x=20, y=450)
    root.cell_rules_frame.yscrollbar.grid_forget()

    root.cell_rules_frame.l_cell_rule = []
    root.cell_rules_frame.innerframe.config(width=root.cell_rule_max_amount*root.cell_rule_width, height=root.cell_rule_height)
    root.cell_rules_frame.update_viewport()

    root.x_cell_rule = 0

    for _ in range(0, 3):
        func_add_new_cell_rule(None)

    # set the values for the cell_rule
    for idx_cell_rule, c_y, c_x, mode in [
        (0, 1, 1, '+'),
        (1, 1, 2, '+'),
        (2, 0, 1, '-'),
    ]:
        cell_rule = root.cell_rules_frame.l_cell_rule[idx_cell_rule]
        if mode == '+':
            cell_rule.add_one(c_y, c_x)
        elif mode == '-':
            cell_rule.sub_one(c_y, c_x)
        else:
            assert False
    
    def get_all_arr_field():
        l_arr_field = []
        for cell_rule in root.cell_rules_frame.l_cell_rule:
            arr_field = cell_rule.arr_field
            l_arr_field.append(arr_field)
        return l_arr_field

    root.get_all_arr_field = get_all_arr_field


    globals()['root'] = root
    globals()['set_new_text'] = set_new_text

    root.mainloop()

    # TODO: use this one for the binary automaton stuff for the future!

  
if __name__ == '__main__':
    main() 
