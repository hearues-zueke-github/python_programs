import numpy as np

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
