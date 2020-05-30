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
    l_arr = [np.array(l).reshape((rows, cols)) for l in [l_rest[field_size*i:field_size*(i+1)] for i in range(0, amount_fields)]]

    d_data = {
        'rows': rows,
        'cols': cols,
        'amount_pieces': amount_pieces,
        'l_amount_group_pieces': l_amount_group_pieces,
        'l_group_pieces': l_group_pieces,
        'length': length,
        'l_group_pieces': l_group_pieces,
        'arr_pcs_idx_pos': arr_pcs_idx_pos,
        'arr_max_height_per_column': arr_max_height_per_column,
        'l_arr': l_arr,
    }

    return d_data