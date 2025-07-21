import gx
import maps
import ui

import rand
import rand.seed
import rand.pcg32

const win_width = 750
const win_height = 500

// fixed, maybe:
// add types of different move sets for each player
// add button for starting a new game
// add promotion for the king piece
// REMOVE this one: add a restart game_field/game button
// find, why piece is not going consistently with the game, per each player!
// show possible moves on the game_field
// find all possible next moves for the current player

// TODO: add a moving animation, when the bot is playing (optional)
// TODO: do a refactoring of current state
// TODO: add move validating for each player
// TODO: add move count and history of each player played
// TODO: create a simple neural network for weights, needed for genetic algorithm
// TODO: implement self play of neural network bots (genetic algorithm)
// TODO: implement serialization of best bots data for useage for new games

fn index_of_first[T](array []T, elem &T) int {
	for i, e in array {
		if *elem == e {
			return i
		}
	}
	return -1
}

@[heap]
struct App {
mut:
	x int
	y int
	window ui.Window
	game_field GameField
	txtfld_player_turn TextField
	txtfld_command TextField
	txtfld_game_finished TextField
	tb_command &ui.TextBox = unsafe { nil }
	btn_do_action &ui.Button = unsafe { nil } // only a test button so far
	
	btn_start_game &ui.Button = unsafe { nil }
	btn_end_game &ui.Button = unsafe { nil }
	btn_random_move &ui.Button = unsafe { nil }
	
	mouse_action MouseAction

	show_field bool
	player_nr_won int
}

@[heap]
struct TextField {
mut:
	x int
	y int
	text string
	color gx.Color
	size int
}

@[heap]
struct GameField {
	field_abs_x int
	field_abs_y int
	
	field_frame_thickness int
	field_frame_color gx.Color

	m_cols int
	n_rows int

	cell_w int
	cell_h int
	cell_frame_thickness int
	
	cell_x_space int
	cell_y_space int
	
	cell_colors []gx.Color
	cell_frame_colors []gx.Color

	cell_color_1 gx.Color
	cell_frame_color_1 gx.Color
	cell_color_2 gx.Color
	cell_frame_color_2 gx.Color

	piece_radius int
	piece_frame_thickness int
	
	piece_colors []gx.Color
	piece_frame_colors []gx.Color

	piece_prev_pos_color gx.Color
	piece_next_pos_color gx.Color
mut:
	map_player_nr_to_players map[int]Player
	map_player_prev map[int]&Player
	map_player_next map[int]&Player
	
	next_piece_id int
	map_piece_id_to_piece map[int]Piece
	
	map_curr_pos_to_next_possible_moves map[string]NextPossibleMove
	arr_curr_pos_to_next_possible_moves []PrevNextPos
	
	current_player &Player = unsafe { nil }

	shared_lock shared []int
	field_array_player_nr []u8
	field_array_piece_id []int
	lut_mouse_rel_pos_int_x []int
	lut_mouse_rel_pos_int_y []int
	is_game_finished bool
	parent_app &App = unsafe { nil }
	rng rand.PRNG
}

// struct PosPiece {
// 	int_x int
// 	int_y int
// }

struct NextPossibleMove {
	int_x int
	int_y int
	arr_remove_piece_id []int
	// remove the opposite pieces
}

struct PrevNextPos {
	prev_x int
	prev_y int
	next_x int
	next_y int
mut:
	piece &Piece = unsafe { nil }
}

enum PieceType {
	normal
	king
}

@[heap]
struct DiffMove {
	diff_x int
	diff_y int
}

// struct Cell {
// 	cell_nr u8
// 	int_x int
// 	int_y int
// 	cell_color &gx.Color
// 	cell_frame_color &gx.Color
// }

@[heap]
struct Piece {
	piece_id int
	player_nr u8
	init_int_x int
	init_int_y int
mut:
	int_x int
	int_y int
	piece_type PieceType
	shared_lock shared []int
}

@[heap]
struct Player {
	player_nr u8
mut:
	arr_piece_id []int
	arr_piece_id_removed []int
	map_piece_type_to_possible_diff_move map[PieceType][]DiffMove
	promotion_y_line int
}

@[heap]
struct MouseAction {
mut:
	is_piece_clicked bool
	piece_id int
	mouse_x int
	mouse_y int
}

fn (mut app App) win_init(win &ui.Window) {
	// init app fields
	app.tb_command = win.get_or_panic[ui.TextBox]('tb_command')
	app.tb_command.set_pos(460, 400)
	app.tb_command.propose_size(180, 20)

	app.btn_do_action = win.get_or_panic[ui.Button]('btn_do_action')
	app.btn_do_action.set_pos(460, 320)
	app.btn_do_action.propose_size(180, 20)

	app.btn_start_game = win.get_or_panic[ui.Button]('btn_start_game')
	app.btn_start_game.set_pos(460, 170)
	app.btn_start_game.propose_size(180, 20)

	app.btn_end_game = win.get_or_panic[ui.Button]('btn_end_game')
	app.btn_end_game.set_pos(460, 195)
	app.btn_end_game.propose_size(180, 20)

	app.btn_random_move = win.get_or_panic[ui.Button]('btn_random_move')
	app.btn_random_move.set_pos(460, 220)
	app.btn_random_move.propose_size(180, 20)
}

fn (mut game_field GameField) place_new_piece(player_nr u8, x int, y int, piece_type PieceType) {
	piece_id := game_field.next_piece_id
	game_field.next_piece_id += 1
	game_field.map_piece_id_to_piece[piece_id] = Piece {
		piece_id: piece_id
		player_nr: player_nr
		int_x: x
		int_y: y
		init_int_x: x
		init_int_y: y
		piece_type: piece_type
	}
	piece := unsafe { &(game_field.map_piece_id_to_piece[piece_id]) }
	game_field.map_player_nr_to_players[player_nr].arr_piece_id << piece.piece_id
	game_field.field_array_player_nr[game_field.m_cols * y + x] = player_nr
	game_field.field_array_piece_id[game_field.m_cols * y + x] = piece.piece_id
}

fn (mut game_field GameField) define_diff_move_per_player_piece(
	side_move_possible bool,
	diag_move_possible bool,
	backward_move_possible bool,
	diag_backward_move_possible bool,
) {
	// define the possible moves for player 1
	mut player1 := unsafe { &(game_field.map_player_nr_to_players[1]) }
	player1.map_piece_type_to_possible_diff_move[.normal] = []DiffMove{}
	player1.map_piece_type_to_possible_diff_move[.king] = []DiffMove{}

	mut player2 := unsafe { &(game_field.map_player_nr_to_players[2]) }
	player2.map_piece_type_to_possible_diff_move[.normal] = []DiffMove{}
	player2.map_piece_type_to_possible_diff_move[.king] = []DiffMove{}

	mut y_dir_mult := 1
	{
		mut diff_move_arr := unsafe { &(player1.map_piece_type_to_possible_diff_move[.normal]) }
		diff_move_arr << DiffMove{diff_x: 0, diff_y: -1 * y_dir_mult}
		if side_move_possible {
			diff_move_arr << DiffMove{diff_x: -1, diff_y: 0 * y_dir_mult}
			diff_move_arr << DiffMove{diff_x: 1, diff_y: 0 * y_dir_mult}
		}
		if diag_move_possible {
			diff_move_arr << DiffMove{diff_x: -1, diff_y: -1 * y_dir_mult}
			diff_move_arr << DiffMove{diff_x: 1, diff_y: -1 * y_dir_mult}	
		}
		if backward_move_possible {
			diff_move_arr << DiffMove{diff_x: 0, diff_y: 1 * y_dir_mult}
		}
		if diag_backward_move_possible {
			diff_move_arr << DiffMove{diff_x: -1, diff_y: 1 * y_dir_mult}
			diff_move_arr << DiffMove{diff_x: 1, diff_y: 1 * y_dir_mult}
		}
	}

	{
		mut diff_move_arr := unsafe { &(player1.map_piece_type_to_possible_diff_move[.king]) }
		diff_move_arr << DiffMove{diff_x: 0, diff_y: -1 * y_dir_mult}
		diff_move_arr << DiffMove{diff_x: 0, diff_y: 1 * y_dir_mult}
		
		diff_move_arr << DiffMove{diff_x: -1, diff_y: 0 * y_dir_mult}
		diff_move_arr << DiffMove{diff_x: 1, diff_y: 0 * y_dir_mult}

		diff_move_arr << DiffMove{diff_x: 1, diff_y: -1 * y_dir_mult}
		diff_move_arr << DiffMove{diff_x: 1, diff_y: 1 * y_dir_mult}
		
		diff_move_arr << DiffMove{diff_x: -1, diff_y: -1 * y_dir_mult}
		diff_move_arr << DiffMove{diff_x: -1, diff_y: 1 * y_dir_mult}
	}

	y_dir_mult = -1
	{
		mut diff_move_arr := unsafe { &(player2.map_piece_type_to_possible_diff_move[.normal]) }
		diff_move_arr << DiffMove{diff_x: 0, diff_y: -1 * y_dir_mult}
		if side_move_possible {
			diff_move_arr << DiffMove{diff_x: -1, diff_y: 0 * y_dir_mult}
			diff_move_arr << DiffMove{diff_x: 1, diff_y: 0 * y_dir_mult}
		}
		if diag_move_possible {
			diff_move_arr << DiffMove{diff_x: -1, diff_y: -1 * y_dir_mult}
			diff_move_arr << DiffMove{diff_x: 1, diff_y: -1 * y_dir_mult}	
		}
		if backward_move_possible {
			diff_move_arr << DiffMove{diff_x: 0, diff_y: 1 * y_dir_mult}
		}
		if diag_backward_move_possible {
			diff_move_arr << DiffMove{diff_x: -1, diff_y: 1 * y_dir_mult}
			diff_move_arr << DiffMove{diff_x: 1, diff_y: 1 * y_dir_mult}
		}
	}

	{
		mut diff_move_arr := unsafe { &(player2.map_piece_type_to_possible_diff_move[.king]) }
		diff_move_arr << DiffMove{diff_x: 0, diff_y: -1 * y_dir_mult}
		diff_move_arr << DiffMove{diff_x: 0, diff_y: 1 * y_dir_mult}
		
		diff_move_arr << DiffMove{diff_x: -1, diff_y: 0 * y_dir_mult}
		diff_move_arr << DiffMove{diff_x: 1, diff_y: 0 * y_dir_mult}

		diff_move_arr << DiffMove{diff_x: 1, diff_y: -1 * y_dir_mult}
		diff_move_arr << DiffMove{diff_x: 1, diff_y: 1 * y_dir_mult}
		
		diff_move_arr << DiffMove{diff_x: -1, diff_y: -1 * y_dir_mult}
		diff_move_arr << DiffMove{diff_x: -1, diff_y: 1 * y_dir_mult}
	}

	println('player1.map_piece_type_to_possible_diff_move[.normal]: ${player1.map_piece_type_to_possible_diff_move[.normal]}')
	println('player1.map_piece_type_to_possible_diff_move[.king]: ${player1.map_piece_type_to_possible_diff_move[.king]}')
	println('')
	println('player2.map_piece_type_to_possible_diff_move[.normal]: ${player2.map_piece_type_to_possible_diff_move[.normal]}')
	println('player2.map_piece_type_to_possible_diff_move[.king]: ${player2.map_piece_type_to_possible_diff_move[.king]}')
}

fn (game_field &GameField) draw_field_cells(mut d ui.DrawDevice, c &ui.CanvasLayout) {
	inner_field_w := game_field.m_cols * game_field.cell_w + game_field.cell_x_space * (game_field.m_cols + 1)
	inner_field_h := game_field.n_rows * game_field.cell_h + game_field.cell_y_space * (game_field.n_rows + 1)

	c.draw_device_rect_surrounded(d, game_field.field_abs_x, game_field.field_abs_y, inner_field_w, inner_field_h, game_field.field_frame_thickness, game_field.field_frame_color)
	
	start_x := game_field.field_abs_x + game_field.cell_x_space // add later the game_field space to the cells too!
	start_y := game_field.field_abs_y + game_field.cell_y_space // add later the game_field space to the cells too!

	// println('before for loops app.txtfld_command: ${app.txtfld_command}')

	for int_x in 0..game_field.m_cols {
		for int_y in 0..game_field.n_rows {
			int_color := (int_x + int_y) % 2

			c.draw_device_rect_filled(d,
				start_x + (game_field.cell_x_space + game_field.cell_w) * int_x,
				start_y + (game_field.cell_y_space + game_field.cell_h) * int_y,
				game_field.cell_w,
				game_field.cell_h,
				game_field.cell_colors[int_color]
			)

			c.draw_device_rect_surrounded(d,
				start_x + (game_field.cell_x_space + game_field.cell_w) * int_x + game_field.cell_frame_thickness,
				start_y + (game_field.cell_y_space + game_field.cell_h) * int_y + game_field.cell_frame_thickness,
				game_field.cell_w - 2 * game_field.cell_frame_thickness,
				game_field.cell_h - 2 * game_field.cell_frame_thickness,
				game_field.cell_frame_thickness,
				game_field.cell_frame_colors[int_color],
			)
		}
	}
}

fn (game_field &GameField) draw_piece_on_canvas(mut d ui.DrawDevice, c &ui.CanvasLayout, piece &Piece, piece_center_x int, piece_center_y int) {
	temp_color := game_field.piece_colors[piece.player_nr - 1]
	temp_color_frame := game_field.piece_frame_colors[piece.player_nr - 1]
	
	c.draw_device_circle_filled(d,
		piece_center_x,
		piece_center_y,
		game_field.piece_radius,
		temp_color_frame,
	)
	c.draw_device_circle_filled(d,
		piece_center_x,
		piece_center_y,
		game_field.piece_radius - game_field.piece_frame_thickness,
		temp_color,
	)
	if piece.piece_type == .king {
		// crown of piece
		c.draw_device_rect_filled(d,
			piece_center_x - 5,
			piece_center_y - 6 - 3,
			10,
			3,
			gx.rgb(0xE0, 0xE0, 0x00),
		)
		c.draw_device_rect_filled(d,
			piece_center_x - 5,
			piece_center_y - 6 - 3 - 3,
			3,
			3,
			gx.rgb(0xE0, 0xE0, 0x00),
		)
		c.draw_device_rect_filled(d,
			piece_center_x - 5 + 4,
			piece_center_y - 6 - 3 - 3,
			2,
			3,
			gx.rgb(0xE0, 0xE0, 0x00),
		)
		c.draw_device_rect_filled(d,
			piece_center_x - 5 + 4 + 3,
			piece_center_y - 6 - 3 - 3,
			3,
			3,
			gx.rgb(0xE0, 0xE0, 0x00),
		)
	}
}

fn (game_field &GameField) draw_field_pieces(mut d ui.DrawDevice, c &ui.CanvasLayout) {
	start_x := game_field.field_abs_x + game_field.cell_x_space // add later the game_field space to the cells too!
	start_y := game_field.field_abs_y + game_field.cell_y_space // add later the game_field space to the cells too!

	start_piece_x := start_x + game_field.cell_w / 2
	start_piece_y := start_y + game_field.cell_h / 2
	piece_to_piece_w := game_field.cell_w + game_field.cell_x_space
	piece_to_piece_h := game_field.cell_h + game_field.cell_y_space

	shared mouse_action := &game_field.parent_app.mouse_action

	for player_id in game_field.map_player_nr_to_players.keys() {
		player := unsafe { &(game_field.map_player_nr_to_players[player_id]) }

		for piece_id in player.arr_piece_id {
			piece := unsafe { &(game_field.map_piece_id_to_piece[piece_id]) }

			rlock mouse_action {
				if mouse_action.is_piece_clicked && mouse_action.piece_id == piece_id {
					continue
				}
			}

			mut piece_center_x := 0
			mut piece_center_y := 0
			rlock piece.shared_lock {
				piece_center_x = start_piece_x + piece.int_x * piece_to_piece_w
				piece_center_y = start_piece_y + piece.int_y * piece_to_piece_h
			}

			game_field.draw_piece_on_canvas(mut d, c, piece, piece_center_x, piece_center_y)
		}
	}

	rlock mouse_action {
		if mouse_action.is_piece_clicked {
			// Draw the moving piece as last!
			piece_center_x := mouse_action.mouse_x
			piece_center_y := mouse_action.mouse_y

			game_field.draw_piece_on_canvas(mut d, c, unsafe { &(game_field.map_piece_id_to_piece[mouse_action.piece_id]) }, piece_center_x, piece_center_y)
		}
	}
}

fn (game_field &GameField) draw_next_possible_moves(mut d ui.DrawDevice, c &ui.CanvasLayout) {
	start_x := game_field.field_abs_x + game_field.cell_x_space // add later the game_field space to the cells too!
	start_y := game_field.field_abs_y + game_field.cell_y_space // add later the game_field space to the cells too!

	start_piece_x := start_x + game_field.cell_w / 2
	start_piece_y := start_y + game_field.cell_h / 2
	piece_to_piece_w := game_field.cell_w + game_field.cell_x_space
	piece_to_piece_h := game_field.cell_h + game_field.cell_y_space

	for prev_next_pos in game_field.arr_curr_pos_to_next_possible_moves {
		piece_center_2_x := start_piece_x + prev_next_pos.next_x * piece_to_piece_w
		piece_center_2_y := start_piece_y + prev_next_pos.next_y * piece_to_piece_h

		c.draw_device_circle_filled(d,
			piece_center_2_x,
			piece_center_2_y,
			6,
			game_field.piece_next_pos_color,
		)
	}

	for prev_next_pos in game_field.arr_curr_pos_to_next_possible_moves {
		piece_center_1_x := start_piece_x + prev_next_pos.prev_x * piece_to_piece_w
		piece_center_1_y := start_piece_y + prev_next_pos.prev_y * piece_to_piece_h

		c.draw_device_circle_filled(d,
			piece_center_1_x,
			piece_center_1_y,
			4,
			game_field.piece_prev_pos_color,
		)
	}
}

fn (mut game_field GameField) move_piece(mut piece Piece, new_x int, new_y int) {
	curr_x, curr_y := piece.get_pos()

	game_field.field_array_player_nr[game_field.m_cols * new_y + new_x] = piece.player_nr
	game_field.field_array_player_nr[game_field.m_cols * curr_y + curr_x] = 0

	game_field.field_array_piece_id[game_field.m_cols * new_y + new_x] = piece.piece_id
	game_field.field_array_piece_id[game_field.m_cols * curr_y + curr_x] = 0

	piece.set_pos(new_x, new_y)

	// check if piece is promoting or not
	if piece.piece_type == .normal && new_y == game_field.current_player.promotion_y_line {
		piece.piece_type = .king
	}

	game_field.current_player = game_field.map_player_next[game_field.current_player.player_nr]
	game_field.empty_and_fill_map_curr_pos_to_next_possible_moves()
}

fn (app &App) draw_field(mut d ui.DrawDevice, c &ui.CanvasLayout) {
	// println('start of draw_field app.txtfld_command: ${app.txtfld_command}')

	game_field := app.game_field
	game_field.draw_field_cells(mut d, c)

	// println('after for loops app.txtfld_command: ${app.txtfld_command}')

	if app.show_field {
		game_field.draw_field_pieces(mut d, c)
		// game_field.draw_next_possible_moves(mut d, c)
	}

	// println('before all printing text app.txtfld_command: ${app.txtfld_command}')
	c.draw_device_text(d, 460, 40, 'Player Nr. 1 turn')
	c.draw_styled_text(460, 55, 'Player Nr. 1 turn, other style', ui.TextStyleParams{
		color: gx.rgb(100, 100, 240), size: 13
	})
	{
		txtfld := &app.txtfld_command
		// println('before printing text app.txtfld_command: ${app.txtfld_command}')
		c.draw_styled_text(txtfld.x, txtfld.y, txtfld.text, ui.TextStyleParams{color: txtfld.color, size: txtfld.size})
	}
	{
		txtfld := &app.txtfld_player_turn
		// println('before printing text app.txtfld_command: ${app.txtfld_command}')
		c.draw_styled_text(txtfld.x, txtfld.y, txtfld.text, ui.TextStyleParams{color: txtfld.color, size: txtfld.size})
	}
	if app.show_field && app.game_field.is_game_finished {
		{
			txtfld := &app.txtfld_game_finished
			// println('before printing text app.txtfld_command: ${app.txtfld_command}')
			c.draw_styled_text(txtfld.x, txtfld.y, txtfld.text, ui.TextStyleParams{color: txtfld.color, size: txtfld.size})
		}
	}
}

fn (mut app App) on_mouse_down_main(window &ui.Window, mouse_event ui.MouseEvent) {
	println('Mouse Down')
	println('- mouse_event.x: ${mouse_event.x}')
	println('- mouse_event.y: ${mouse_event.y}')
	println('- mouse_event.button: ${mouse_event.button}')
	println('- mouse_event.action: ${mouse_event.action}')
	println('- mouse_event.mods: ${mouse_event.mods}')

	game_field := &app.game_field
	if app.show_field && !app.game_field.is_game_finished {
		field_w := game_field.cell_x_space * (game_field.m_cols + 1) + game_field.cell_w * game_field.m_cols
		field_h := game_field.cell_y_space * (game_field.n_rows + 1) + game_field.cell_h * game_field.n_rows

		mouse_rel_x := mouse_event.x - game_field.field_abs_x
		mouse_rel_y := mouse_event.y - game_field.field_abs_y

		println('field_w: ${field_w}, field_h: ${field_h}')
		println('mouse_rel_x: ${mouse_rel_x}, mouse_rel_y: ${mouse_rel_y}')

		if
			(mouse_rel_x < 0) || (mouse_rel_x >= field_w) ||
			(mouse_rel_y < 0) || (mouse_rel_y >= field_h) {
			println('Ignore mouse click!')
			return
		}

		int_x := game_field.lut_mouse_rel_pos_int_x[mouse_rel_x]
		int_y := game_field.lut_mouse_rel_pos_int_y[mouse_rel_y]

		// find piece with the position! TODO: implement a map of coordinates
		shared mouse_action := &app.mouse_action
		for piece_id in game_field.map_piece_id_to_piece.keys() {
			mut piece := unsafe { &(game_field.map_piece_id_to_piece[piece_id]) }
			mut check_x_equal := false
			mut check_y_equal := false

			rlock piece.shared_lock {
				check_x_equal = piece.int_x == int_x
				check_y_equal = piece.int_y == int_y
			}
			if check_x_equal && check_y_equal && piece.player_nr == game_field.current_player.player_nr {
				lock mouse_action {
					mouse_action.is_piece_clicked = true
					mouse_action.piece_id = piece_id
					mouse_action.mouse_x = mouse_event.x
					mouse_action.mouse_y = mouse_event.y
				}
				break
			}
		}

		rlock mouse_action {
			if mouse_action.is_piece_clicked {
				println('Piece was clicked!')
				println('- piece_id: ${mouse_action.piece_id}')
				println('- piece: ${game_field.map_piece_id_to_piece[mouse_action.piece_id]}')
			} else {
				println('A valid piece was not clicked!')
			}
		}
	}
}

fn (mut app App) on_mouse_move_main(window &ui.Window, mouse_event ui.MouseMoveEvent) {
	// println('Mouse Move')
	// println('- mouse_event.x: ${mouse_event.x}')
	// println('- mouse_event.y: ${mouse_event.y}')
	// println('- mouse_event.mouse_button: ${mouse_event.mouse_button}')
	game_field := &app.game_field

	if app.show_field && !app.game_field.is_game_finished {
		shared mouse_action := &app.mouse_action
		lock mouse_action {
			if mouse_action.is_piece_clicked {
				mouse_action.mouse_x = int(mouse_event.x)
				mouse_action.mouse_y = int(mouse_event.y)
			}
		}
	}
}

fn (mut app App) update_txtfld_player_turn() {
	game_field := &app.game_field

	app.txtfld_player_turn.text = 'Turn Player Nr. ${game_field.current_player.player_nr}'

	// check if all pieces in the players array and the game_field array are same!
	mut pieces_pos_player_1 := map[string]bool
	mut pieces_pos_player_2 := map[string]bool
	mut pieces_pos_field := map[string]bool

	for piece_id in game_field.map_player_nr_to_players[1].arr_piece_id {
		p := unsafe { &(game_field.map_piece_id_to_piece[piece_id]) }
		pieces_pos_player_1['${p.int_x},${p.int_y}'] = true
	}
	for piece_id in game_field.map_player_nr_to_players[2].arr_piece_id {
		p := unsafe { &(game_field.map_piece_id_to_piece[piece_id]) }
		pieces_pos_player_2['${p.int_x},${p.int_y}'] = true
	}
	for piece_id in game_field.map_piece_id_to_piece.keys() {
		p := unsafe { &(game_field.map_piece_id_to_piece[piece_id]) }
		pieces_pos_field['${p.int_x},${p.int_y}'] = true
	}

	arr_pos_players := maps.merge(pieces_pos_player_1, pieces_pos_player_2).keys().sorted()
	arr_pos_field := pieces_pos_field.keys().sorted()
	println('pos pieces in player 1 and 2:\n${arr_pos_players}')
	println('pos pieces in game_field:\n${arr_pos_field}')
	println('arr_pos_players == arr_pos_field ? ${arr_pos_players == arr_pos_field}')
}

fn (mut app App) on_mouse_up_main(window &ui.Window, mouse_event ui.MouseEvent) {
	println('Mouse Up')
	println('- mouse_event.x: ${mouse_event.x}')
	println('- mouse_event.y: ${mouse_event.y}')
	println('- mouse_event.button: ${mouse_event.button}')
	println('- mouse_event.action: ${mouse_event.action}')
	println('- mouse_event.mods: ${mouse_event.mods}')

	mut game_field := &app.game_field
	if app.show_field && !app.game_field.is_game_finished {
		field_w := game_field.cell_x_space * (game_field.m_cols + 1) + game_field.cell_w * game_field.m_cols
		field_h := game_field.cell_y_space * (game_field.n_rows + 1) + game_field.cell_h * game_field.n_rows

		mouse_rel_x := mouse_event.x - game_field.field_abs_x
		mouse_rel_y := mouse_event.y - game_field.field_abs_y

		println('field_w: ${field_w}, field_h: ${field_h}')
		println('mouse_rel_x: ${mouse_rel_x}, mouse_rel_y: ${mouse_rel_y}')

		shared mouse_action := &app.mouse_action

		if (
			(mouse_rel_x < 0) || (mouse_rel_x >= field_w) ||
			(mouse_rel_y < 0) || (mouse_rel_y >= field_h)
		) {
			println('Ignore mouse click!')
		} else {
			new_x := game_field.lut_mouse_rel_pos_int_x[mouse_rel_x]
			new_y := game_field.lut_mouse_rel_pos_int_y[mouse_rel_y]

			println('Piece should be placed at: new_x: ${new_x}, new_y: ${new_y}')

			lock mouse_action {
				if mouse_action.is_piece_clicked {
					piece_id := mouse_action.piece_id
					mut piece := unsafe { &(game_field.map_piece_id_to_piece[piece_id]) }
					
					// make a check, if piece was possible to move there
					curr_x, curr_y := piece.get_pos()
					key := '${curr_x},${curr_y},${new_x},${new_y}'
					println('move piece, key: ${key}')
					println('current piece: ${piece}')

					if game_field.field_array_player_nr[game_field.m_cols * new_y + new_x] == 0 && key in game_field.map_curr_pos_to_next_possible_moves {
					// if game_field.field_array_player_nr[game_field.m_cols * new_y + new_x] == 0 && key in game_field.map_curr_pos_to_next_possible_moves {
						app.update_move(key, mut piece, new_x, new_y)
					}

					mouse_action.is_piece_clicked = false
					mouse_action.piece_id = 0
				}
			}
		}

		lock mouse_action {
			if mouse_action.is_piece_clicked {
				mouse_action.is_piece_clicked = false
				mouse_action.piece_id = 0
			}
		}
	}
}

fn (mut app App) on_key_down_main(window &ui.Window, key_event ui.KeyEvent) {
	match key_event.key {
		.a {
			println('Key A was pressed!')
		}
		.d {
			println('Key D was pressed!')
		}
		.s {
			println('Key S was pressed!')
			app.start_new_game()
		}
		.e {
			println('Key E was pressed!')
			app.end_current_game()
		}
		.r {
			println('Key R was pressed!')
			app.do_random_move()
		}
		.w {
			println('Key W was pressed!')
		}
		.enter {
			println('Pressed enter!')
			mut tb_command := &app.tb_command
			text := tb_command.get_text()
			println('Text in command is: "${text}"')
			// tb_command.set_text('')

			if text.len == 0 {
				return
			}

			text_split := text.split(' ')

			cmd := text_split[0]
			if cmd == 'mv' {
				println('cmd: mv')
				if text_split.len < 4 {
					return
				}

				name := text_split[1]
				coord := text_split[2]
				num_str := text_split[3]

				if name == 'command' {
					mut txtfld_command := &app.txtfld_command
					if coord == 'x' {
						txtfld_command.x += num_str.int()
					} else if coord == 'y' {
						txtfld_command.y += num_str.int()
					}
					println('- txtfld_command: ${txtfld_command}')
				} else if name == 'do_action' {
					mut btn_do_action := &app.btn_do_action
					if coord == 'x' {
						btn_do_action.set_pos(btn_do_action.x + num_str.int(), btn_do_action.y)
					} else if coord == 'y' {
						btn_do_action.set_pos(btn_do_action.x, btn_do_action.y + num_str.int())
					} else if coord == 'w' {
						btn_do_action.propose_size(btn_do_action.width + num_str.int(), btn_do_action.height)
					} else if coord == 'h' {
						btn_do_action.propose_size(btn_do_action.width, btn_do_action.height + num_str.int())
					}
					println('- btn_do_action: (x: ${btn_do_action.x}, y: ${btn_do_action.y}, width: ${btn_do_action.width}, height: ${btn_do_action.height})')
				}
			}
		}
		else {
			println('Key Down')
			println('- key_event.key: ${key_event.key}')
			println('- key_event.action: ${key_event.action}')
			println('- key_event.code: ${key_event.code}')
			println('- key_event.mods: ${key_event.mods}')
			println('- key_event.codepoint: ${key_event.codepoint}')
		}
	}
}

fn (mut piece Piece) reset_pos() {
	lock piece.shared_lock {
		piece.int_x = piece.init_int_x
		piece.int_y = piece.init_int_y
	}
}

fn (piece &Piece) get_pos() (int, int) {
	mut int_x := 0
	mut int_y := 0
	rlock piece.shared_lock {
		int_x = piece.int_x
		int_y = piece.int_y
	}
	return int_x, int_y
}

fn (mut piece Piece) set_pos(int_x int, int_y int) {
	lock piece.shared_lock {
		piece.int_x = int_x
		piece.int_y = int_y
	}
}

fn (mut game_field GameField) empty_and_fill_map_curr_pos_to_next_possible_moves() {
	mut map_curr_pos_to_next_possible_moves := unsafe { &(game_field.map_curr_pos_to_next_possible_moves) }
	mut arr_curr_pos_to_next_possible_moves := unsafe { &(game_field.arr_curr_pos_to_next_possible_moves) }

	for key in map_curr_pos_to_next_possible_moves.keys() {
		map_curr_pos_to_next_possible_moves.delete(key)
	}
	arr_len := arr_curr_pos_to_next_possible_moves.len
	for i in 0..arr_len {
		arr_curr_pos_to_next_possible_moves.delete_last()
	}

	map_piece_type_to_possible_diff_move := unsafe { &(game_field.current_player.map_piece_type_to_possible_diff_move) }

	print('all pieces: ')
	for piece_id, mut piece in game_field.map_piece_id_to_piece {
		print('(${piece.int_x}, ${piece.int_y}), ')
	}
	println('')
	print('pieces of player_nr ${game_field.current_player.player_nr}: ')
	for piece_id in game_field.current_player.arr_piece_id {
		piece := unsafe { &(game_field.map_piece_id_to_piece[piece_id]) }
		print('(${piece.int_x}, ${piece.int_y}), ')
	}
	println('')

	for piece_id in game_field.current_player.arr_piece_id {
		piece := unsafe { &(game_field.map_piece_id_to_piece[piece_id]) }

		curr_x := unsafe { piece.int_x }
		curr_y := unsafe { piece.int_y }
		possible_diff_move := &(unsafe { map_piece_type_to_possible_diff_move[ piece.piece_type] })
		for diff_move in possible_diff_move {
			new_x := curr_x + diff_move.diff_x
			new_y := curr_y + diff_move.diff_y

			// check bounds
			if (
				new_x < 0 || new_x >= game_field.m_cols ||
				new_y < 0 || new_y >= game_field.n_rows
			) {
				continue
			}

			// check collisions with other pieces, depence on the rulling
			if game_field.field_array_player_nr[game_field.m_cols * new_y + new_x] == 0 {
				unsafe {
					map_curr_pos_to_next_possible_moves['${curr_x},${curr_y},${new_x},${new_y}'] = NextPossibleMove{
						int_x: new_x
						int_y: new_y
						arr_remove_piece_id: []
					}
				}
				arr_curr_pos_to_next_possible_moves << PrevNextPos{
					prev_x: curr_x
					prev_y: curr_y
					next_x: new_x
					next_y: new_y
					piece: piece
				}
				continue
			}

			// get the piece, which is jumped over
			piece_id_jumped_over := game_field.field_array_piece_id[game_field.m_cols * new_y + new_x]
			piece_jumped_over := unsafe { &(game_field.map_piece_id_to_piece[piece_id_jumped_over]) }

			mut arr_remove_piece_id := []int{}
			if piece_jumped_over.player_nr != game_field.current_player.player_nr {
				arr_remove_piece_id << piece_jumped_over.piece_id
			}

			new_2_x := new_x + diff_move.diff_x
			new_2_y := new_y + diff_move.diff_y

			if (
				new_2_x < 0 || new_2_x >= game_field.m_cols ||
				new_2_y < 0 || new_2_y >= game_field.n_rows
			) {
				continue
			}

			if game_field.field_array_player_nr[game_field.m_cols * new_2_y + new_2_x] == 0 {
				unsafe {
					map_curr_pos_to_next_possible_moves['${curr_x},${curr_y},${new_2_x},${new_2_y}'] = NextPossibleMove{
						int_x: new_2_x
						int_y: new_2_y
						arr_remove_piece_id: arr_remove_piece_id
					}
				}
				arr_curr_pos_to_next_possible_moves << PrevNextPos{
					prev_x: curr_x
					prev_y: curr_y
					next_x: new_2_x
					next_y: new_2_y
					piece: piece
				}
				continue
			}
		}
	}

	if arr_curr_pos_to_next_possible_moves.len == 0 {
		game_field.is_game_finished = true
		game_field.parent_app.player_nr_won = game_field.map_player_prev[game_field.current_player.player_nr].player_nr
		game_field.parent_app.txtfld_game_finished.text = 'Player Nr. ${game_field.parent_app.player_nr_won} Won!'
	}
	println('game_field.is_game_finished ? ${game_field.is_game_finished}')

	println('map_curr_pos_to_next_possible_moves.keys(): ${map_curr_pos_to_next_possible_moves.keys()}')
}

fn (mut game_field GameField) start_new_game() {
	game_field.is_game_finished = false

	for i in 0..game_field.field_array_player_nr.len {
			game_field.field_array_player_nr[i] = 0
	}
	for i in 0..game_field.field_array_piece_id.len {
			game_field.field_array_piece_id[i] = 0
	}
	lock game_field.shared_lock {
		for piece_id, mut piece in game_field.map_piece_id_to_piece {
			piece.reset_pos()
			piece.piece_type = .normal
			game_field.field_array_player_nr[game_field.m_cols * piece.int_y + piece.int_x] = piece.player_nr
			game_field.field_array_piece_id[game_field.m_cols * piece.int_y + piece.int_x] = piece.piece_id
		}
	}
	for player_nr in game_field.map_player_nr_to_players.keys() {
		mut player := unsafe { &(game_field.map_player_nr_to_players[player_nr]) }

		for player.arr_piece_id_removed.len > 0 {
			player.arr_piece_id << player.arr_piece_id_removed.pop()
		}
	}
	game_field.current_player = unsafe { &(game_field.map_player_nr_to_players[1]) }
	game_field.parent_app.txtfld_player_turn.text = 'Turn Player Nr. ${game_field.current_player.player_nr}'

	// fill all possible next moves
	game_field.empty_and_fill_map_curr_pos_to_next_possible_moves()
}

fn (mut app App) btn_do_action_on_click(button &ui.Button) {
	println('fn btn_do_action_on_click() called!!!!!!!!!!!!!!!!')
	// println('button: ${button}')
}

fn (mut app App) start_new_game() {
	app.show_field = true
	app.game_field.start_new_game()
}

fn (mut app App) btn_start_game_on_click(button &ui.Button) {
	app.start_new_game()
	println('fn btn_start_game_on_click() called!!!!!!!!!!!!!!!!')
}

fn (mut app App) end_current_game() {
	app.game_field.is_game_finished = false
	app.show_field = false
	app.txtfld_player_turn.text = ''
}

fn (mut app App) btn_end_game_on_click(button &ui.Button) {
	app.end_current_game()
	println('fn btn_end_game_on_click() called!!!!!!!!!!!!!!!!')
}

fn (mut app App) update_move(key string, mut piece Piece, next_x int, next_y int) {
	mut game_field := &app.game_field

	next_possible_move := unsafe { &(game_field.map_curr_pos_to_next_possible_moves[key]) }
	arr_remove_piece_id := next_possible_move.arr_remove_piece_id

	println('arr_remove_piece_id: ${arr_remove_piece_id}')

	if arr_remove_piece_id.len > 0 {
		for piece_id in arr_remove_piece_id {
			println('piece_id to be removed: ${piece_id}')
			println('piece_id in game_field.map_piece_id_to_piece ? ${piece_id in game_field.map_piece_id_to_piece}')
			piece2 := unsafe { &(game_field.map_piece_id_to_piece[piece_id]) }
			mut player := unsafe { &(game_field.map_player_nr_to_players[piece2.player_nr]) }

			println('before:')
			println('player.arr_piece_id: ${player.arr_piece_id}')
			println('player.arr_piece_id_removed: ${player.arr_piece_id_removed}')
			
			println('Remove piece from game_field: piece2: ${piece2}')
			println('piece is from player_nr ${piece2.player_nr}')

			idx_element := index_of_first(player.arr_piece_id, &(piece2.piece_id))
			player.arr_piece_id.delete(idx_element)
			player.arr_piece_id_removed << (piece_id)

			game_field.field_array_player_nr[game_field.m_cols * piece2.int_y + piece2.int_x] = 0
			game_field.field_array_piece_id[game_field.m_cols * piece2.int_y + piece2.int_x] = 0

			println('after:')
			println('player.arr_piece_id: ${player.arr_piece_id}')
			println('player.arr_piece_id_removed: ${player.arr_piece_id_removed}')
		}
	}

	app.game_field.move_piece(mut piece, next_x, next_y)
	app.update_txtfld_player_turn()

	println('game_field.field_array_player_nr: [')
	for j in 0..game_field.n_rows {
		for i in 0..game_field.m_cols {
			print('${game_field.field_array_player_nr[game_field.m_cols * j + i]}, ')
		}
		println('')
	}
	println(']')

	println('game_field.field_array_piece_id: [')
	for j in 0..game_field.n_rows {
		for i in 0..game_field.m_cols {
			print('${game_field.field_array_piece_id[game_field.m_cols * j + i]:2}, ')
		}
		println('')
	}
	println(']')
}

fn (mut app App) do_random_move() {
	println('-- app.game_field.is_game_finished ? ${app.game_field.is_game_finished}')
	if app.show_field && !app.game_field.is_game_finished {
		mut random_prev_next_pos := app.game_field.rng.element(app.game_field.arr_curr_pos_to_next_possible_moves) or {PrevNextPos{0, 0, 0, 0, unsafe { nil }}}
		println('choosen element is: ${random_prev_next_pos}')
		
		prev_x := random_prev_next_pos.prev_x
		prev_y := random_prev_next_pos.prev_y
		next_x := random_prev_next_pos.next_x
		next_y := random_prev_next_pos.next_y
		key := '${prev_x},${prev_y},${next_x},${next_y}'
		
		app.update_move(key, mut random_prev_next_pos.piece, random_prev_next_pos.next_x, random_prev_next_pos.next_y)
	}
	println('-- app.game_field.is_game_finished ? ${app.game_field.is_game_finished}')
}

fn (mut app App) btn_random_move_on_click(button &ui.Button) {
	println('fn btn_random_move_on_click() called!!!!!!!!!!!!!!!!')
	app.do_random_move()
}

fn main() {
	mut app := &App{}
	app.x = 30
	app.y = 50
	app.game_field = GameField{
		field_abs_x: 30
		field_abs_y: 50
		
		field_frame_thickness: 7
		field_frame_color: gx.rgb(80, 80, 80)

		m_cols: 6,
		n_rows: 6,

		cell_w: 50,
		cell_h: 50,
		cell_frame_thickness: 4,
		
		cell_x_space: 2,
		cell_y_space: 2,
		
		cell_colors: [gx.rgb(118, 150, 86), gx.rgb(238, 238, 210)]
		cell_frame_colors: [gx.rgb(242, 225, 163), gx.rgb(21, 93, 77)]

		piece_radius: 18
		piece_frame_thickness: 4
		piece_colors: [gx.rgb(20, 160, 120), gx.rgb(90, 100, 30)]
		piece_frame_colors: [gx.rgb(0, 0, 0), gx.rgb(0, 0, 0)]

		piece_prev_pos_color: gx.rgb(0xFF, 0x20, 0x20)
		piece_next_pos_color: gx.rgb(0x20, 0xFF, 0x20)

		rng: &rand.PRNG(pcg32.PCG32RNG{})

		next_piece_id: 1
	}
	shared app.mouse_action = MouseAction{
		is_piece_clicked: false
	}

	mut game_field := &app.game_field
	game_field.rng.seed(seed.time_seed_array(pcg32.seed_len))
	game_field.parent_app = app

	game_field.map_player_nr_to_players[1] = Player{player_nr: 1, promotion_y_line: 0}
	game_field.map_player_nr_to_players[2] = Player{player_nr: 2, promotion_y_line: game_field.n_rows - 1}

	{
		keys_sorted := game_field.map_player_nr_to_players.keys().sorted()

		mut keys_sorted_shift := keys_sorted.clone()
		first := keys_sorted_shift[0]
		keys_sorted_shift.delete(0)
		keys_sorted_shift << first
	
		println('keys_sorted: ${keys_sorted}')
		println('keys_sorted_shift: ${keys_sorted_shift}')

		for i in 0..keys_sorted.len {
			player_nr_before := keys_sorted[i]
			player_nr_after := keys_sorted_shift[i]

			game_field.map_player_prev[player_nr_after] = unsafe { &(game_field.map_player_nr_to_players[player_nr_before]) }
			game_field.map_player_next[player_nr_before] = unsafe { &(game_field.map_player_nr_to_players[player_nr_after]) }
		}
	}

	game_field.field_array_player_nr = []u8{len: game_field.n_rows * game_field.m_cols, init: 0}
	game_field.field_array_piece_id = []int{len: game_field.n_rows * game_field.m_cols, init: unsafe { nil }}

	game_field.define_diff_move_per_player_piece(false, true, false, false)

	field_w := game_field.cell_x_space * (game_field.m_cols + 1) + game_field.cell_w * game_field.m_cols
	field_h := game_field.cell_y_space * (game_field.n_rows + 1) + game_field.cell_h * game_field.n_rows

	game_field.lut_mouse_rel_pos_int_x = []int{len: field_w, init: 0}
	game_field.lut_mouse_rel_pos_int_y = []int{len: field_h, init: 0}

	cell_x_space_hlf_1 := game_field.cell_x_space / 2
	cell_x_space_hlf_2 := game_field.cell_x_space - cell_x_space_hlf_1

	cell_y_space_hlf_1 := game_field.cell_y_space / 2
	cell_y_space_hlf_2 := game_field.cell_y_space - cell_y_space_hlf_1

	start_rel_pos_int_x_cell_1 := game_field.cell_x_space + game_field.cell_w + cell_x_space_hlf_1
	start_rel_pos_int_y_cell_1 := game_field.cell_y_space + game_field.cell_h + cell_y_space_hlf_1

	// do the first cell rel_pos_int_x and rel_pos_int_y
	for i in 0..(start_rel_pos_int_x_cell_1) {
		game_field.lut_mouse_rel_pos_int_x[i] = 0
	}
	for i in 0..(game_field.cell_y_space + game_field.cell_w + cell_y_space_hlf_1) {
		game_field.lut_mouse_rel_pos_int_y[i] = 0
	}

	// do the middle cells
	for i in 1..(game_field.m_cols - 1) {
		pos_start := start_rel_pos_int_x_cell_1 + (game_field.cell_w + game_field.cell_x_space) * (i - 1)
		pos_end := pos_start + game_field.cell_w + game_field.cell_x_space
		for pos in pos_start..pos_end {
			game_field.lut_mouse_rel_pos_int_x[pos] = i
		}
	}
	for i in 1..(game_field.n_rows - 1) {
		pos_start := start_rel_pos_int_y_cell_1 + (game_field.cell_h + game_field.cell_y_space) * (i - 1)
		pos_end := pos_start + game_field.cell_h + game_field.cell_y_space
		for pos in pos_start..pos_end {
			game_field.lut_mouse_rel_pos_int_y[pos] = i
		}
	}

	// do the last cell rel_pos_int_x and rel_pos_int_y
	for i in (field_w - (game_field.cell_x_space + game_field.cell_w + cell_x_space_hlf_2))..field_w {
		game_field.lut_mouse_rel_pos_int_x[i] = game_field.m_cols - 1
	}
	for i in (field_h - (game_field.cell_y_space + game_field.cell_h + cell_y_space_hlf_2))..field_h {
		game_field.lut_mouse_rel_pos_int_y[i] = game_field.n_rows - 1
	}

	println('game_field.lut_mouse_rel_pos_int_x: ${game_field.lut_mouse_rel_pos_int_x}')
	println('game_field.lut_mouse_rel_pos_int_y: ${game_field.lut_mouse_rel_pos_int_y}')

	// init the pieces for the players
	for i in 0..game_field.m_cols {
		app.game_field.place_new_piece(1, i, game_field.n_rows - 2, .normal)
		app.game_field.place_new_piece(2, i, 1, .normal)
	}

	app.show_field = false
	game_field.is_game_finished = false

	app.txtfld_command = TextField{
		x: 461
		y: 377
		text: 'Command:'
		color: gx.rgb(0, 0, 0)
		size: 20
	}
	app.txtfld_player_turn = TextField{
		x: 460
		y: 100
		text: ''
		color: gx.rgb(0, 30, 0)
		size: 40
	}

	{
		// field_center_x := app.field_abs_x + ()
		app.txtfld_game_finished = TextField{
			x: 150
			y: 200
			text: ''
			color: gx.rgb(70, 130, 0)
			size: 30
		}
	}

	mut window := ui.window(
		width:  win_width
		height: win_height
		title:  'V UI: Rectangles'
		on_mouse_down: app.on_mouse_down_main,
		on_mouse_move: app.on_mouse_move_main,
		on_mouse_up: app.on_mouse_up_main,
		on_key_down: app.on_key_down_main,
		on_init: app.win_init,
		children: [
			ui.canvas_plus(
				bg_color:      gx.rgb(196, 196, 196)
				bg_radius:     .0
				clipping:      true
				on_draw:       app.draw_field
				// on_click:      app.click_circles
				// on_mouse_move: app.mouse_move_circles
				z_index: 0,
			),
			ui.textbox(id: 'tb_command', z_index: 1)
			ui.button(
				id: 'btn_do_action',
				on_click: app.btn_do_action_on_click,
				z_index: 2,
				text: 'Do Action',
				radius: 5,
				border_color: gx.rgb(0, 128, 0),
				bg_color: gx.rgb(192, 192, 128),
			),
			ui.button(id: 'btn_start_game', radius: 5, z_index: 3, text: 'Start Game', on_click: app.btn_start_game_on_click),
			ui.button(id: 'btn_end_game', radius: 5, z_index: 3, text: 'End Game', on_click: app.btn_end_game_on_click),
			ui.button(id: 'btn_random_move', radius: 5, z_index: 3, text: 'Random Move', on_click: app.btn_random_move_on_click),
		]
	)

	ui.run(window)
}
