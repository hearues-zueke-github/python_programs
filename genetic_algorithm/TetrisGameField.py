import random

import numpy as np

class TetrisGameField(Exception):
    def __init__(self, d_basic_data_info, l_snn, max_piece_nr=1000, using_pieces=3):
        self.rows = d_basic_data_info['rows']
        self.cols = d_basic_data_info['cols']
        self.amount_pieces = d_basic_data_info['amount_pieces']
        self.l_amount_group_pieces = d_basic_data_info['l_amount_group_pieces']
        self.l_group_pieces = d_basic_data_info['l_group_pieces']

        self.arr_pcs_idx = np.arange(0, self.amount_pieces)
        self.arr_pcs_one_hot = np.diag(np.ones((self.amount_pieces, )))

        self.l_group_piece_arr_pos = d_basic_data_info['l_group_piece_arr_pos']
        self.l_group_piece_max_x = d_basic_data_info['l_group_piece_max_x']
        self.l_pcs_idx_l_index_to_group_idx_pos = d_basic_data_info['l_pcs_idx_l_index_to_group_idx_pos']

        self.using_pieces = using_pieces
        self.max_piece_nr = max_piece_nr

        self.l_snn = l_snn

        self.arr_x = np.empty((self.rows*self.cols + self.amount_pieces*self.using_pieces, ))

        self.garbage_lines = 5
        self.with_garbage = False

        self.field_add_rows = 4
        self.field = np.zeros((self.rows+self.field_add_rows, self.cols), dtype=np.uint8)
        self.field_y = self.field.shape[0]
        self.field_x = self.field.shape[1]

        self.games_played = 0


    def __repr__(self):
        return f'TetrisGameField(rows={self.rows}, cols={self.cols}, amount_pieces={self.amount_pieces})'


    def reset_values(self):
        self.field[:] = 0

        self.l_clear_lines = []
        self.l_pieces_used = []

        self.need_field_reset = False
        self.piece_nr = 0
        self.clear_lines = 0

        self.l_next_pcs = np.random.choice(self.arr_pcs_idx, size=(self.using_pieces, )).tolist()
        self.pcs_now = random.choice(self.arr_pcs_idx)

        if self.with_garbage:
            self.add_garbage_lines()


    def add_garbage_lines(self):
        self.field[-self.garbage_lines:] = 1
        ys = np.arange(self.rows+self.field_add_rows-1, self.rows+self.field_add_rows-1-self.garbage_lines, -1)
        xs = np.random.randint(0, self.cols, (self.garbage_lines, ))
        self.field[(ys, xs)] = 0


    def define_next_piece(self):
        self.pcs_now = self.l_next_pcs.pop(0)
        self.l_next_pcs.append(random.choice(self.arr_pcs_idx))
        
        self.piece_posi_y = 1
        
        snn = self.l_snn[self.pcs_now]
        self.arr_x[:] = 0
        self.arr_x[:self.rows*self.cols] = (self.field[self.field_add_rows:].reshape((-1, ))!=0)+0
        self.arr_x[self.rows*self.cols:] = self.arr_pcs_one_hot[[self.pcs_now]+self.l_next_pcs[:-1]].reshape((-1, ))
        self.arr_x[self.arr_x==0] = -1

        argmax = snn.calc_output_argmax(self.arr_x)

        group_idx, piece_posi_x = self.l_pcs_idx_l_index_to_group_idx_pos[self.pcs_now][argmax]

        self.group_idx = group_idx
        self.piece_posi_x = piece_posi_x

        self.piece_positions = self.l_group_piece_arr_pos[self.pcs_now][self.group_idx]


    def do_move_piece_down_instant(self):
        is_move_down_possible = True
        s_pos_now = set()
        for yi_, xi_ in self.piece_positions:
            yi = self.piece_posi_y+yi_
            xi = self.piece_posi_x+xi_
            if yi+1>=self.field_y:
                is_move_down_possible = False
                break
            s_pos_now.add((yi, xi))
        
        if not is_move_down_possible:
            return False

        s_pos_down = set()
        for yi, xi in s_pos_now:
            s_pos_down.add((yi+1, xi))

        s_next_pos = s_pos_down-s_pos_now

        if not all([self.field[yi, xi]==0 for yi, xi in s_next_pos]):
            is_move_down_possible = False
            if self.piece_posi_y < 5:
                self.need_field_reset = True
            return False

        s_pos_down_prev = s_pos_down
        i = 2
        while True:
            s_pos_down = set()
            for yi, xi in s_pos_now:
                s_pos_down.add((yi+i, xi))

            s_next_pos = s_pos_down-s_pos_now

            if any([yi>=self.field_y for yi, _ in s_next_pos]) or any([self.field[yi, xi]!=0 for yi, xi in s_next_pos]):
                break

            s_pos_down_prev = s_pos_down
            i += 1

        s_pos_down = s_pos_down_prev

        s_next_pos = s_pos_down-s_pos_now
        s_prev_pos = s_pos_now-s_pos_down

        for yi, xi in s_prev_pos:
            self.field[yi, xi] = 0
           
        for yi, xi in s_next_pos:
            self.field[yi, xi] = self.pcs_now+1

        self.piece_posi_y += i-1

        return True


    def main_game_loop(self):
        self.games_played += 1

        self.reset_values()

        while True:
            self.define_next_piece()
            is_piece_moved = self.do_move_piece_down_instant()

            if not is_piece_moved and self.need_field_reset:
                return False
            elif self.piece_nr>self.max_piece_nr:
                return True

            idxs_row_full = np.all(self.field!=0, axis=1)
            if np.any(idxs_row_full):
                self.field[idxs_row_full] = 0

                self.clear_lines += np.sum(idxs_row_full)

                all_rows_full = np.where(idxs_row_full)[0]
                all_rows_empty = np.where(np.all(self.field==0, axis=1))[0]
                all_rows = np.arange(self.field_y-1, -1, -1)

                all_rows_rest = all_rows[(~np.isin(all_rows, all_rows_full))&(~np.isin(all_rows, all_rows_empty))]

                all_rows_needed = all_rows[:all_rows_rest.shape[0]]
                idxs_same_rows = np.where(all_rows_rest!=all_rows_needed)[0]
                
                if idxs_same_rows.shape[0]>0:
                    idx_first_row = idxs_same_rows[0]

                    rows_from = all_rows_rest[idx_first_row:]
                    rows_to = all_rows_needed[idx_first_row:]

                    for row_from, row_to in zip(rows_from, rows_to):
                        self.field[row_to] = self.field[row_from]
                        self.field[row_from] = 0

            self.piece_nr += 1