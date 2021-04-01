import numpy as np

import sys
sys.path.append("../")
from utils_function import copy_function

class BitAutomaton(Exception):
    __slot__ = [
        'h', 'w',
        'frame', 'frame_wrap',
        'field_size', 'field',
        'field_frame_size', 'field_frame',
        'd_vars',
        'l_func', 'func_rng', 's_func_nr',
    ]

    def __init__(self):
        pass


    def init_vals(self, h, w, frame, frame_wrap, l_func=None, func_inv=None, func_rng=None):
        self.h = h
        self.w = w
        
        self.frame = frame
        self.frame_wrap = frame_wrap
        
        self.field_size = (h, w)
        self.field = np.zeros(self.field_size, dtype=np.bool)

        self.field_frame_size = (h+frame*2, w+frame*2)
        self.field_frame = np.zeros(self.field_frame_size, dtype=np.bool)

        self.d_vars = {}
        self.d_vars['n'] = self.field_frame[frame:-frame, frame:-frame]
        l_up = [('u', i, -i) for i in range(frame, 0, -1)]
        l_down = [('d', i, i) for i in range(1, frame+1)]
        l_left = [('l', i, -i) for i in range(frame, 0, -1)]
        l_right = [('r', i, i) for i in range(1, frame+1)]
        l_empty = [('', 0, 0)]

        for direction, amount, i in l_up+l_down:
            self.d_vars[direction+str(amount)] = self.field_frame[frame+i:frame+h+i, frame:frame+w]
            self.d_vars[direction*amount] = self.field_frame[frame+i:frame+h+i, frame:frame+w]
        for direction, amount, i in l_left+l_right:
            self.d_vars[direction+str(amount)] = self.field_frame[frame:frame+h, frame+i:frame+i+w]
            self.d_vars[direction*amount] = self.field_frame[frame:frame+h, frame+i:frame+i+w]

        for direction_y, amount_y, i_y in l_up+l_down:
            for direction_x, amount_x, i_x in l_left+l_empty+l_right:
                self.d_vars[direction_y+str(amount_y)+direction_x+str(amount_x)] = self.field_frame[frame+i_y:frame+i_y+h, frame+i_x:frame+i_x+w]
                self.d_vars[direction_y*amount_y+direction_x*amount_x] = self.field_frame[frame+i_y:frame+i_y+h, frame+i_x:frame+i_x+w]

        if l_func is not None:
            self.l_func = [copy_function(f, self.d_vars) for f in l_func]
            self.s_func_nr = set(range(0, len(l_func)))
        if func_inv is not None:
            self.d_vars['inv'] = copy_function(func_inv)
        if func_rng is not None:
            self.func_rng = copy_function(func_rng, self.d_vars)

        return self


    def set_field(self, field):
        assert isinstance(field, np.ndarray)
        assert field.shape == self.field_size
        assert field.dtype == np.bool
        self.field = field
        self.fill_field_frame()


    def fill_field_frame(self):
        self.field_frame[self.frame:-self.frame, self.frame:-self.frame] = self.field

        if self.frame_wrap == False:
            self.field_frame[:, :self.frame] = 0
            self.field_frame[:, -self.frame:] = 0
            self.field_frame[:self.frame, self.frame:-self.frame] = 0
            self.field_frame[-self.frame:, self.frame:-self.frame] = 0
        else:
            # do the right part copy to left
            self.field_frame[self.frame:-self.frame, :self.frame] = self.field_frame[self.frame:-self.frame, -self.frame*2:-self.frame]
            # do the left part copy to right
            self.field_frame[self.frame:-self.frame, -self.frame:] = self.field_frame[self.frame:-self.frame, self.frame:self.frame*2]
            # do the bottom part copy to top
            self.field_frame[:self.frame] = self.field_frame[-self.frame*2:-self.frame]
            # do the top part copy to bottom
            self.field_frame[-self.frame:] = self.field_frame[self.frame:self.frame*2]


    def execute_func(self, n):
        # print('execute_func: n: {}'.format(n))
        assert n in self.s_func_nr
        self.field = self.l_func[n]()
        self.fill_field_frame()


    def __lshift__(self, n):
        return self.field << n
