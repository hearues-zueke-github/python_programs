#!/usr/bin/env python3

# -*- coding: utf-8 -*-
 
from functools import partial
 
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


def main():
    root = tk.Tk()

    def close():
        print("Application-Shutdown")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)

    root.geometry('+{x}+{y}'.format(x=200, y=150))
    root.geometry('{width}x{height}'.format(width=500, height=550))

    frame_own = FrameOwn(root, width=240, height=300)
    frame_own.place(x=30, y=20)

    # fill frame_own with labels!
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
     
    root.mainloop()

    # TODO: use this one for the binary automaton stuff for the future!

  
if __name__ == '__main__':
    main() 
