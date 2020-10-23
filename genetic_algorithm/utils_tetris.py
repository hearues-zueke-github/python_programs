import os

import numpy as np

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

def parse_tetris_game_data(file_path):
    with open(file_path, "rb") as f:
        l_bytes = list(f.read())

    rows, cols = l_bytes[:2]
    field_cells = rows*cols

    # extract all fields first!
    amount_pieces = l_bytes[2]

    i = 3
    l_amount_group_pieces = l_bytes[i:i+amount_pieces]

    i = i+amount_pieces

    l_group_pieces = []
    for _ in range(0, sum(l_amount_group_pieces)):
        amount_pos = l_bytes[i]
        l_group_pieces.append(l_bytes[i+1:i+1+amount_pos*2])
        i += 1+amount_pos*2

    l_length = l_bytes[i:i+2]
    length = (l_length[1]<<8)+l_length[0]
    i += 2

    arr_pcs_idx_pos = np.array(l_bytes[i:i+3*length]).reshape((-1, 3))
    i += 3*length

    arr_max_height_per_column = np.array(l_bytes[i:i+cols*length]).reshape((-1, cols))
    i += cols*length

    l_rest = l_bytes[i:]
    len_l_rest = len(l_rest)

    field_size = rows*cols
    amount_fields = len(l_rest)//field_size
    assert len_l_rest==field_size*amount_fields
    assert length*2+2==amount_fields
    arr_fields = np.array(l_rest).reshape((-1, rows, cols))

    l_group_piece_arr_pos = []
    l_group_piece_max_x = []
    group_idx_acc = 0
    for group_amount in l_amount_group_pieces:
        l_group_one_piece_arr_pos = []
        l_group_one_piece_max_x = []
        for _ in range(0, group_amount):
            arr = np.array(l_group_pieces[group_idx_acc]).reshape((-1, 2))
            l_group_one_piece_arr_pos.append(arr.tolist())
            l_group_one_piece_max_x.append(max(arr[:, 1]))
            group_idx_acc += 1
        l_group_piece_arr_pos.append(l_group_one_piece_arr_pos)
        l_group_piece_max_x.append(l_group_one_piece_max_x)

    l_pcs_idx_l_index_to_group_idx_pos = []
    for pcs_idx in np.arange(0, amount_pieces):
        l_index_to_group_idx_pos = []
        for group_idx, max_x in enumerate(l_group_piece_max_x[pcs_idx], 0):
            for pos in range(0, cols-max_x):
                l_index_to_group_idx_pos.append((group_idx, pos))
        l_pcs_idx_l_index_to_group_idx_pos.append(l_index_to_group_idx_pos)

    d_data = {
        'rows': rows,
        'cols': cols,
        'amount_pieces': amount_pieces,
        'l_amount_group_pieces': l_amount_group_pieces,
        'l_group_pieces': l_group_pieces,
        
        'l_group_piece_arr_pos': l_group_piece_arr_pos,
        'l_group_piece_max_x': l_group_piece_max_x,
        'l_pcs_idx_l_index_to_group_idx_pos': l_pcs_idx_l_index_to_group_idx_pos,

        'length': length,
        'arr_pcs_idx_pos': arr_pcs_idx_pos,
        'arr_max_height_per_column': arr_max_height_per_column,
        'arr_fields': arr_fields,
    }

    return d_data


def load_Xs_Ts_from_tetris_data(d_data, using_pieces=3):
    rows = d_data['rows']
    cols = d_data['cols']
    amount_pieces = d_data['amount_pieces']
    l_amount_group_pieces = d_data['l_amount_group_pieces']
    l_group_pieces = d_data['l_group_pieces']

    arr_fields = d_data['arr_fields'][1::2]
    arr_pcs_idx_pos = d_data['arr_pcs_idx_pos']
    # could be used instead of the arr_fields, but lets see
    arr_max_height_per_column = d_data['arr_max_height_per_column']

    # calculate first all possible tuple combinations for the piece + direction + pos!
    l_pcs_group = []
    l_pcs_group_idx = []
    l_pcs_group_max_x = []
    l_pcs_idx_l_tpl_pcs_group_pos = []
    group_idx_acc = 0
    for pcs_idx, amount in enumerate(l_amount_group_pieces, 0):
        l = []
        l_idx = []
        l_max_x = []
        for j in range(0, amount):
            l_pos = l_group_pieces[group_idx_acc]
            max_x = max(l_pos[1::2])

            l.append(l_pos)
            l_idx.append(group_idx_acc)
            l_max_x.append(max_x)

            group_idx_acc += 1
        l_pcs_group.append(l)
        l_pcs_group_idx.append(l_idx)
        l_pcs_group_max_x.append(l_max_x)

        l_pcs_idx_l_tpl_pcs_group_pos.append([(pcs_idx, group_idx, pos) for group_idx, max_x in zip(l_idx, l_max_x) for pos in range(0, cols-max_x)])

    d_pcl_idx_d_tpl_pcs_group_pos_to_index = {
        pcs_idx: {t: i for i, t in enumerate(l_tpl_pcs_group_pos, 0)} for pcs_idx, l_tpl_pcs_group_pos in enumerate(l_pcs_idx_l_tpl_pcs_group_pos, 0)
    }

    s_used_pcs_group_pos = set([tuple(l) for l in arr_pcs_idx_pos.tolist()]) 
    s_all_pcs_group_pos = set([t for l in l_pcs_idx_l_tpl_pcs_group_pos for t in l])

    s_diff = s_all_pcs_group_pos - s_used_pcs_group_pos

    # convert the fields + other data into the learning X and T Matrices!
    Xs = [[] for _ in range(0, amount_pieces)]
    Ts = [[] for _ in range(0, amount_pieces)]

    arr_pcs_one_hot = np.diag(np.ones((amount_pieces, ), dtype=np.int8))

    x_len = rows*cols + using_pieces*amount_pieces
    l_t_len = [len(l) for l in l_pcs_idx_l_tpl_pcs_group_pos]
    for i in range(0, d_data['length']-using_pieces+1):
        arr_field = arr_fields[i]
        arr_pcs_idx_pos_part = arr_pcs_idx_pos[i:i+using_pieces]
        arr_pcs_idx, arr_group_idx, arr_pos = arr_pcs_idx_pos_part.T

        pcs_idx = arr_pcs_idx[0]
        X = Xs[pcs_idx]
        T = Ts[pcs_idx]

        # vector for x
        arr_x = np.zeros((x_len, ), dtype=np.int8)
        arr_t = np.zeros((l_t_len[pcs_idx], ), dtype=np.int8)

        arr_x[:rows*cols] = (arr_field.reshape((-1, ))!=0)+0
        arr_x[rows*cols:] = arr_pcs_one_hot[arr_pcs_idx].reshape((-1, ))

        tpl = tuple(arr_pcs_idx_pos_part[0].tolist())
        # l_tpl_pcs_group_pos = l_pcs_idx_l_tpl_pcs_group_pos[pcs_idx]
        d_tpl_pcs_group_pos = d_pcl_idx_d_tpl_pcs_group_pos_to_index[pcs_idx]
        t_idx = d_tpl_pcs_group_pos[tpl]

        arr_t[t_idx] = 1

        arr_x[arr_x==0] = -1

        X.append(arr_x)
        T.append(arr_t)
        # break

    l_len_Xs = [len(X) for X in Xs]
    # print("l_len_Xs: {}".format(l_len_Xs))


    Xs = [np.array(X, dtype=np.float) for X in Xs]
    Ts = [np.array(T, dtype=np.float) for T in Ts]

    return Xs, Ts


def load_Xs_Ts_full(rows=12, cols=5, proc_num=1, iterations_multi_processing=1, using_pieces=3, block_cells=[4]):
    # TODO: add the amount of needed xs and ts vectors!

    # rows = 12
    # cols = 5
    # proc_num = 1
    # iterations_multi_processing = 1
    l_suffix = ['r{rows}_c{cols}_i{nr}_j{proc}_b{bc}'.format(rows=rows, cols=cols, proc=i, nr=j, bc=''.join(map(str, block_cells)))
        for i in range(1, proc_num+1) for j in range(1, iterations_multi_processing+1)
    ]
    # l_suffix = ['12_5_{}_{}'.format(i, j) for i in range(1, 8) for j in range(1, 2)]
    # l_suffix = ['{:03}_{}'.format(i, j) for i in range(101, 108) for j in range(1, 2)]
    # l_suffix = ['400']
    file_name_template = 'tetris_game_data/data_fields_{suffix}.ttrsfields'
    data_file_path_template = PATH_ROOT_DIR+file_name_template

    print("suffix: {}".format(l_suffix[0]))
    data_file_path = data_file_path_template.format(suffix=l_suffix[0])
    d_data = parse_tetris_game_data(file_path=data_file_path)

    d_basic_data_info = {
        'rows': d_data['rows'],
        'cols': d_data['cols'],
        'amount_pieces': d_data['amount_pieces'],
        'l_amount_group_pieces': d_data['l_amount_group_pieces'],
        'l_group_pieces': d_data['l_group_pieces'],

        'using_pieces': using_pieces,

        'l_group_piece_arr_pos': d_data['l_group_piece_arr_pos'],
        'l_group_piece_max_x': d_data['l_group_piece_max_x'],
        'l_pcs_idx_l_index_to_group_idx_pos': d_data['l_pcs_idx_l_index_to_group_idx_pos'],
    }

    Xs, Ts = load_Xs_Ts_from_tetris_data(d_data=d_data, using_pieces=using_pieces)

    l_Xs = [[X] for X in Xs]
    l_Ts = [[T] for T in Ts]

    for suffix in l_suffix[1:]:
        print("suffix: {}".format(suffix))
        data_file_path = data_file_path_template.format(suffix=suffix)
        d_data = parse_tetris_game_data(file_path=data_file_path)

        d_basic_data_info_new = {
            'rows': d_data['rows'],
            'cols': d_data['cols'],
            'amount_pieces': d_data['amount_pieces'],
            'l_amount_group_pieces': d_data['l_amount_group_pieces'],
            'l_group_pieces': d_data['l_group_pieces'],

            'using_pieces': using_pieces,

            'l_group_piece_arr_pos': d_data['l_group_piece_arr_pos'],
            'l_group_piece_max_x': d_data['l_group_piece_max_x'],
            'l_pcs_idx_l_index_to_group_idx_pos': d_data['l_pcs_idx_l_index_to_group_idx_pos'],
        }

        assert d_basic_data_info==d_basic_data_info_new

        Xs, Ts = load_Xs_Ts_from_tetris_data(d_data=d_data, using_pieces=using_pieces)

        for l_X, l_T, X, T in zip(l_Xs, l_Ts, Xs, Ts):
            l_X.append(X)
            l_T.append(T)

    Xs_full = [np.vstack(l_X) for l_X in l_Xs]
    Ts_full = [np.vstack(l_T) for l_T in l_Ts]

    l_len_Xs_full = [len(X) for X in Xs_full]
    print("l_len_Xs_full: {}".format(l_len_Xs_full))

    return Xs_full, Ts_full, d_basic_data_info
