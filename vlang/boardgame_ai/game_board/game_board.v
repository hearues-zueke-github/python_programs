module game_board

import datatypes { Set }

struct Pos {
	x int
	y int
}

pub struct PieceIdPosMoveScore {
	piece_id int
	x_curr int
	y_curr int
	x_next int
	y_next int
	score int
}

pub fn PieceIdPosMoveScore.get_string(obj &PieceIdPosMoveScore) string {
	return 'PieceIdPosMoveScore(' +
	'piece_id: ' + unsafe { (&obj).piece_id }.str() +
	', x_curr: ' + unsafe { (&obj).x_curr }.str() +
	', y_curr: ' + unsafe { (&obj).y_curr }.str() +
	', x_next: ' + unsafe { (&obj).x_next }.str() +
	', y_next: ' + unsafe { (&obj).y_next }.str() +
	', score: ' + unsafe { (&obj).score }.str() +
	')'
}

enum PieceType {
	normal_piece
	king_piece
}

@[heap]
struct Piece {
	piece_id int
	player_id int
mut:
	x int
	y int
	piece_type PieceType
}

@[heap]
struct Player {
	player_id int
mut:
	set_piece_id_on_board Set[int]
	set_piece_id_removed Set[int]
}

@[heap]
pub struct GameBoard {
	m_cols int
	n_rows int
mut:
	arr_player_id []int
	map_player_id_to_player map[int]Player

	arr_piece_id []int
	map_piece_id_to_piece map[int]Piece
	
	map_player_id_to_map_piece_type_to_map_curr_pos_to_arr_next_pos map[int]map[PieceType]map[string][]Pos

	arr_board_player_id []int
	arr_board_piece_id []int

	player_id_won int
	is_game_started bool
	is_game_finished bool

	current_player_id int
	map_player_id_to_next_player_id map[int]int
}

fn (mut gb GameBoard) generate_player(player_id int) {
	mut player := Player {
		player_id: player_id
		set_piece_id_on_board: Set[int]{}
		set_piece_id_removed: Set[int]{}
	}
	gb.arr_player_id << player_id
	gb.map_player_id_to_player[player_id] = player
	println('gb.arr_player_id: ${gb.arr_player_id}')
}

fn (mut gb GameBoard) generate_player_pieces_row(player_id int, y int, piece_id_counter &int) {
	mut player := unsafe { &(gb.map_player_id_to_player[player_id]) }

	mut x := int(0)
	for _ in 0..gb.m_cols {
		piece := Piece {
			piece_id: *piece_id_counter
			player_id: player_id
			x: x
			y: y
			piece_type: PieceType.normal_piece
		}

		gb.arr_piece_id << *piece_id_counter
		gb.map_piece_id_to_piece[*piece_id_counter] = piece
		
		player.set_piece_id_on_board.add(*piece_id_counter)

		gb.arr_board_player_id[y * gb.m_cols + x] = player_id
		gb.arr_board_piece_id[y * gb.m_cols + x] = *piece_id_counter

		unsafe { *piece_id_counter += 1 }
		x += 1
	}
}

fn (gb &GameBoard) generate_map_curr_pos_next_pos(arr_diff_pos []Pos) map[string][]Pos {
	mut map_curr_pos_to_arr_next_pos := map[string][]Pos{}

	for y_curr in 0..gb.n_rows {
		for x_curr in 0..gb.m_cols {
			mut arr_next_pos := []Pos{}

			for diff_pos in arr_diff_pos {
				if diff_pos.x == 0 && diff_pos.y == 0 {
					continue
				}

				x_next := x_curr + diff_pos.x
				if x_next < 0 || x_next >= gb.m_cols {
					continue
				}

				y_next := y_curr + diff_pos.y
				if y_next < 0 || y_next >= gb.n_rows {
					continue
				}

				arr_next_pos << Pos {
					x: x_next
					y: y_next
				}
			}
			
			if arr_next_pos.len == 0 {
				continue
			}

			map_curr_pos_to_arr_next_pos['${x_curr},${y_curr}'] = arr_next_pos
		}
	}

	return map_curr_pos_to_arr_next_pos
}

fn (mut gb GameBoard) generate_all_next_possible_moves() {
	arr_diff_pos_player_1_normal := [
		Pos{x: -1, y: 1},
		Pos{x: 0, y: 1},
		Pos{x: 1, y: 1},
	]

	arr_diff_pos_player_2_normal := [
		Pos{x: -1, y: -1},
		Pos{x: 0, y: -1},
		Pos{x: 1, y: -1},
	]

	arr_diff_pos_king := [
		Pos{x: -1, y: -1},
		Pos{x: 0, y: -1},
		Pos{x: 1, y: -1},
		Pos{x: -1, y: 0},
		Pos{x: 1, y: 0},
		Pos{x: -1, y: 1},
		Pos{x: 0, y: 1},
		Pos{x: 1, y: 1},
	]

	{
		mut map_piece_type_to_map_curr_pos_to_arr_next_pos := map[PieceType]map[string][]Pos{}

		map_piece_type_to_map_curr_pos_to_arr_next_pos[.normal_piece] = gb.generate_map_curr_pos_next_pos(arr_diff_pos_player_1_normal)
		map_piece_type_to_map_curr_pos_to_arr_next_pos[.king_piece] = gb.generate_map_curr_pos_next_pos(arr_diff_pos_king)

		gb.map_player_id_to_map_piece_type_to_map_curr_pos_to_arr_next_pos[1] = map_piece_type_to_map_curr_pos_to_arr_next_pos.move()
	}

	{
		mut map_piece_type_to_map_curr_pos_to_arr_next_pos := map[PieceType]map[string][]Pos{}

		map_piece_type_to_map_curr_pos_to_arr_next_pos[.normal_piece] = gb.generate_map_curr_pos_next_pos(arr_diff_pos_player_2_normal)
		map_piece_type_to_map_curr_pos_to_arr_next_pos[.king_piece] = gb.generate_map_curr_pos_next_pos(arr_diff_pos_king)

		gb.map_player_id_to_map_piece_type_to_map_curr_pos_to_arr_next_pos[2] = map_piece_type_to_map_curr_pos_to_arr_next_pos.move()
	}
}

pub fn GameBoard.new(m_cols int, n_rows int) GameBoard {
	mut arr_board_player_id := []int{len: m_cols * n_rows, init: 0}
	mut arr_board_piece_id := []int{len: m_cols * n_rows, init: 0}

	mut gb := GameBoard {
		m_cols: m_cols
		n_rows: n_rows

		arr_player_id: []int{}
		map_player_id_to_player: map[int]Player{}

		arr_piece_id: []int{}
		map_piece_id_to_piece: map[int]Piece{}

		map_player_id_to_map_piece_type_to_map_curr_pos_to_arr_next_pos: map[int]map[PieceType]map[string][]Pos{}

		arr_board_player_id: arr_board_player_id
		arr_board_piece_id: arr_board_piece_id

		player_id_won: 0
		is_game_started: true
		is_game_finished: false
	}

	mut piece_id_counter := int(1)

	{
		player_id := int(1)
		gb.generate_player(player_id)
		y := int(1)
		gb.generate_player_pieces_row(player_id, y, &piece_id_counter)
	}

	{
		player_id := int(2)
		gb.generate_player(player_id)
		y := n_rows - 2
		gb.generate_player_pieces_row(player_id, y, &piece_id_counter)
	}

	gb.generate_all_next_possible_moves()

	gb.current_player_id = gb.arr_player_id[0]

	gb.map_player_id_to_next_player_id = map[int]int{}
	for i in 0..gb.arr_player_id.len-1 {
		gb.map_player_id_to_next_player_id[gb.arr_player_id[i]] = gb.arr_player_id[i + 1]
	}
	gb.map_player_id_to_next_player_id[gb.arr_player_id[gb.arr_player_id.len-1]] = gb.arr_player_id[0]

	return gb
}

pub fn (gb &GameBoard) print_board_info() {
	println('m_cols: ${gb.m_cols}')
	println('n_rows: ${gb.n_rows}')
	println('arr_player_id: ${gb.arr_player_id}')
	println('map_player_id_to_player:')
	for player_id in gb.map_player_id_to_player.keys() {
		player := unsafe { &(gb.map_player_id_to_player[player_id]) }
		println('- player_id: ${player_id}, player: ${player}')
	}
	println('arr_piece_id: ${gb.arr_piece_id}')
	println('map_piece_id_to_piece:')
	for piece_id in gb.map_piece_id_to_piece.keys() {
		piece := unsafe { &(gb.map_piece_id_to_piece[piece_id]) }
		println('- piece_id: ${piece_id}, piece: ${piece}')
	}
	println('arr_board_player_id: ${gb.arr_board_player_id}')
	println('arr_board_piece_id: ${gb.arr_board_piece_id}')
	println('beautify arr_board_player_id:')
	for j in 0..gb.n_rows {
		for i in 0..gb.m_cols {
			print('${gb.arr_board_player_id[j * gb.m_cols + i]},')
		}
		println('')
	}
	println('beautify arr_board_piece_id:')
	for j in 0..gb.n_rows {
		for i in 0..gb.m_cols {
			print('${gb.arr_board_piece_id[j * gb.m_cols + i]:2},')
		}
		println('')
	}
	println('map_player_id_to_map_piece_type_to_map_curr_pos_to_arr_next_pos:')
	for player_id in gb.map_player_id_to_map_piece_type_to_map_curr_pos_to_arr_next_pos.keys().sorted() {
		println('- player_id: ${player_id}')
		map_piece_type_to_map_curr_pos_to_arr_next_pos := unsafe { &(gb.map_player_id_to_map_piece_type_to_map_curr_pos_to_arr_next_pos[player_id]) }

		for piece_type in map_piece_type_to_map_curr_pos_to_arr_next_pos.keys() {
			println('-- piece_type: ${piece_type}')
			map_curr_pos_to_arr_next_pos := unsafe { &(map_piece_type_to_map_curr_pos_to_arr_next_pos[piece_type]) }
		
			for curr_pos in map_curr_pos_to_arr_next_pos.keys().sorted() {
				print('--- curr_pos: "${curr_pos}", ')
				arr_next_pos := unsafe { &(map_curr_pos_to_arr_next_pos[curr_pos]) }
				print('arr_next_pos: [')
				for next_pos in arr_next_pos {
					print('(${next_pos.x}, ${next_pos.y}), ')
				}
				println(']')
			}
		}
	}
	println('')
	println('player_id_won: ${gb.player_id_won}')
	println('is_game_started: ${gb.is_game_started}')
	println('is_game_finished: ${gb.is_game_finished}')
	println('current_player_id: ${gb.current_player_id}')	
	println('map_player_id_to_next_player_id: ${gb.map_player_id_to_next_player_id}')	
}

pub fn (gb &GameBoard) print_board_state() {
	println('arr_board_player_id: ${gb.arr_board_player_id}')
	println('arr_board_piece_id: ${gb.arr_board_piece_id}')
	println('beautify arr_board_player_id:')
	for j in 0..gb.n_rows {
		for i in 0..gb.m_cols {
			print('${gb.arr_board_player_id[j * gb.m_cols + i]},')
		}
		println('')
	}
	println('beautify arr_board_piece_id:')
	for j in 0..gb.n_rows {
		for i in 0..gb.m_cols {
			print('${gb.arr_board_piece_id[j * gb.m_cols + i]:2},')
		}
		println('')
	}
	println('')
	println('player_id_won: ${gb.player_id_won}')
	println('is_game_started: ${gb.is_game_started}')
	println('is_game_finished: ${gb.is_game_finished}')
	println('current_player_id: ${gb.current_player_id}')	
	println('map_player_id_to_next_player_id: ${gb.map_player_id_to_next_player_id}')	
}

pub fn (gb &GameBoard) generate_for_current_player_next_possible_moves() []PieceIdPosMoveScore {
	if !gb.is_game_started {
		return []PieceIdPosMoveScore{}
	}

	if gb.is_game_finished {
		return []PieceIdPosMoveScore{}
	}

	map_piece_type_to_map_curr_pos_to_arr_next_pos := unsafe { gb.map_player_id_to_map_piece_type_to_map_curr_pos_to_arr_next_pos[gb.current_player_id] }

	mut arr_pos_move_score := []PieceIdPosMoveScore{}

	current_player := unsafe { &(gb.map_player_id_to_player[gb.current_player_id]) }
	for piece_id in current_player.set_piece_id_on_board.array() {
		piece := unsafe { &(gb.map_piece_id_to_piece[piece_id]) }
		map_curr_pos_to_arr_next_pos := unsafe { map_piece_type_to_map_curr_pos_to_arr_next_pos[piece.piece_type] }
		pos_str := '${piece.x},${piece.y}'
		arr_next_pos := unsafe { map_curr_pos_to_arr_next_pos[pos_str] }

		println('piece: ${piece}')
		println('map_curr_pos_to_arr_next_pos: ${map_curr_pos_to_arr_next_pos}')
		println('pos_str: ${pos_str}')
		println('arr_next_pos: ${arr_next_pos}')

		for pos in arr_next_pos {
			// check if the position is occupied or not
			if gb.arr_board_piece_id[pos.y * gb.m_cols + pos.x] == 0 {
				arr_pos_move_score << PieceIdPosMoveScore {
					piece_id: piece_id
					x_curr: piece.x
					y_curr: piece.y
					x_next: pos.x
					y_next: pos.y
					score: 0 // TODO: evaluate the score later
				}
			}
		}
	}

	return arr_pos_move_score
}

pub fn (mut gb GameBoard) restart_game() {
	// TODO: for future
}
