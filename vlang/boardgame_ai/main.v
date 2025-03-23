import ui
import gx

const win_width = 750
const win_height = 500

// TODO: add types of different move sets for each player
// TODO: add button for starting a new game
// TODO: REMOVE this one: add a restart field/game button

// TODO: add a moving animation, when the bot is playing
// TODO: do a refactoring of current state
// TODO: add move validating for each player
// TODO: add move count and history of each player played
// TODO: create a simple neural network for weights, needed for genetic algorithm
// TODO: implement self play of neural network bots (genetic algorithm)
// TODO: implement serialization of best bots data for useage for new games
// TODO: find all possible next moves for the current player
// TODO: add promotion for the king piece

@[heap]
struct App {
mut:
	x int
	y int
	window ui.Window
	field Field
	txtfld_player_turn TextField
	txtfld_command TextField
	tb_command &ui.TextBox = unsafe { nil }
	btn_do_action &ui.Button = unsafe { nil } // only a test button so far
	
	btn_start_game &ui.Button = unsafe { nil }
	btn_end_game &ui.Button = unsafe { nil }
	// btn_reset_field &ui.Button = unsafe { nil }
	
	// btn_change_x_plus &ui.Button = unsafe { nil }
	// btn_change_x_minus &ui.Button = unsafe { nil }
	// btn_change_y_plus &ui.Button = unsafe { nil }
	// btn_change_y_minus &ui.Button = unsafe { nil }
	// btn_change_width_plus &ui.Button = unsafe { nil }
	// btn_change_width_minus &ui.Button = unsafe { nil }
	// btn_change_height_plus &ui.Button = unsafe { nil }
	// btn_change_height_minus &ui.Button = unsafe { nil }
	
	mouse_action MouseAction
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
struct Field {
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
mut:
	players []Player
	map_player_next map[int]&Player
	map_curr_pos_to_next_possible_moves map[string]NextPossibleMove
	// current_player_nr int
	current_player &Player = unsafe { nil }
	pieces []Piece
	field_array []u8
	field_array_piece []&Piece
	lut_mouse_rel_pos_int_x []int
	lut_mouse_rel_pos_int_y []int
	show_field bool
	parent_app &App = unsafe { nil }
}

// struct PosPiece {
// 	int_x int
// 	int_y int
// }

struct NextPossibleMove {
	int_x int
	int_y int
	// remove the opposite pieces
}

enum PieceType {
	normal
	king
}

@[heap]
struct DiffMove {
mut:
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
	player_nr u8
	init_int_x int
	init_int_y int
mut:
	int_x int
	int_y int
	piece_type PieceType
}

@[heap]
struct Player {
	player_nr u8
mut:
	pieces []&Piece
	// arr_possible_diff_move [][]DiffMove
	map_piece_type_to_possible_diff_move map[PieceType][]DiffMove
}

@[heap]
struct MouseAction {
mut:
	is_piece_clicked bool
	piece &Piece = unsafe { nil }
	mouse_x int
	mouse_y int
}

fn main() {
	mut app := &App{}
	app.x = 30
	app.y = 50
	app.field = Field{
		field_abs_x: 30
		field_abs_y: 50
		
		field_frame_thickness: 7
		field_frame_color: gx.rgb(80, 80, 80)

		m_cols: 8,
		n_rows: 8,

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
	}
	shared app.mouse_action = MouseAction{
		is_piece_clicked: false
	}

	mut field := &app.field
	field.parent_app = app

	field.players << [Player{player_nr: 1}, Player{player_nr: 2}]
	for i in 0..(field.players.len - 1) {
		player_before := &field.players[i]
		player_after := &field.players[i + 1]
		field.map_player_next[player_before.player_nr] = player_after
	}
	field.map_player_next[field.players[field.players.len - 1].player_nr] = &field.players[0]
	field.map_player_next
	field.field_array = []u8{len: field.n_rows * field.m_cols, init: 0}
	field.field_array_piece = []&Piece{len: field.n_rows * field.m_cols, init: unsafe { nil }}
	field.define_diff_move_per_player_piece(false, true, false, false)

	field_w := field.cell_x_space * (field.m_cols + 1) + field.cell_w * field.m_cols
	field_h := field.cell_y_space * (field.n_rows + 1) + field.cell_h * field.n_rows

	field.lut_mouse_rel_pos_int_x = []int{len: field_w, init: 0}
	field.lut_mouse_rel_pos_int_y = []int{len: field_h, init: 0}

	cell_x_space_hlf_1 := field.cell_x_space / 2
	cell_x_space_hlf_2 := field.cell_x_space - cell_x_space_hlf_1

	cell_y_space_hlf_1 := field.cell_y_space / 2
	cell_y_space_hlf_2 := field.cell_y_space - cell_y_space_hlf_1

	start_rel_pos_int_x_cell_1 := field.cell_x_space + field.cell_w + cell_x_space_hlf_1
	start_rel_pos_int_y_cell_1 := field.cell_y_space + field.cell_h + cell_y_space_hlf_1

	// do the first cell rel_pos_int_x and rel_pos_int_y
	for i in 0..(start_rel_pos_int_x_cell_1) {
		field.lut_mouse_rel_pos_int_x[i] = 0
	}
	for i in 0..(field.cell_y_space + field.cell_w + cell_y_space_hlf_1) {
		field.lut_mouse_rel_pos_int_y[i] = 0
	}

	// do the middle cells
	for i in 1..(field.m_cols - 1) {
		pos_start := start_rel_pos_int_x_cell_1 + (field.cell_w + field.cell_x_space) * (i - 1)
		pos_end := pos_start + field.cell_w + field.cell_x_space
		for pos in pos_start..pos_end {
			field.lut_mouse_rel_pos_int_x[pos] = i
		}
	}
	for i in 1..(field.n_rows - 1) {
		pos_start := start_rel_pos_int_y_cell_1 + (field.cell_h + field.cell_y_space) * (i - 1)
		pos_end := pos_start + field.cell_h + field.cell_y_space
		for pos in pos_start..pos_end {
			field.lut_mouse_rel_pos_int_y[pos] = i
		}
	}

	// do the last cell rel_pos_int_x and rel_pos_int_y
	for i in (field_w - (field.cell_x_space + field.cell_w + cell_x_space_hlf_2))..field_w {
		field.lut_mouse_rel_pos_int_x[i] = field.m_cols - 1
	}
	for i in (field_h - (field.cell_y_space + field.cell_h + cell_y_space_hlf_2))..field_h {
		field.lut_mouse_rel_pos_int_y[i] = field.n_rows - 1
	}

	println('field.lut_mouse_rel_pos_int_x: ${field.lut_mouse_rel_pos_int_x}')
	println('field.lut_mouse_rel_pos_int_y: ${field.lut_mouse_rel_pos_int_y}')

	// init the pieces for the players
	for i in 0..field.m_cols {
		app.field.place_new_piece(1, i, field.n_rows - 2, .normal)
		app.field.place_new_piece(2, i, 1, .normal)
	}

	field.show_field = false

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
	// println('app.txtfld_command: ${app.txtfld_command}')

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
			// ui.button(id: 'btn_reset_field', radius: 5, z_index: 3, text: 'Reset Field', on_click: app.btn_reset_field_on_click),
			
			// ui.button(id: 'btn_change_x_plus', radius: 5, z_index: 3, text: 'X +', on_click: app.btn_change_x_plus_on_click),
			// ui.button(id: 'btn_change_x_minus', radius: 5, z_index: 3, text: 'X -', on_click: app.btn_change_x_minus_on_click),
			// ui.button(id: 'btn_change_y_plus', radius: 5, z_index: 3, text: 'Y +', on_click: app.btn_change_y_plus_on_click),
			// ui.button(id: 'btn_change_y_minus', radius: 5, z_index: 3, text: 'Y -', on_click: app.btn_change_y_minus_on_click),
			// ui.button(id: 'btn_change_width_plus', radius: 5, z_index: 3, text: 'W +', on_click: app.btn_change_width_plus_on_click),
			// ui.button(id: 'btn_change_width_minus', radius: 5, z_index: 3, text: 'W -', on_click: app.btn_change_width_minus_on_click),
			// ui.button(id: 'btn_change_height_plus', radius: 5, z_index: 3, text: 'H +', on_click: app.btn_change_height_plus_on_click),
			// ui.button(id: 'btn_change_height_minus', radius: 5, z_index: 3, text: 'H -', on_click: app.btn_change_height_minus_on_click),
		]
	)

	// println('in main before ui.run app.txtfld_command: ${app.txtfld_command}')

	ui.run(window)
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

	// app.btn_reset_field = win.get_or_panic[ui.Button]('btn_reset_field')
	// app.btn_reset_field.set_pos(460, 220)
	// app.btn_reset_field.propose_size(180, 20)

	// w := 30
	// h := 30

	// x1 := 500
	// x2 := x1 + w + 5
	// y1 := 30
	// y2 := y1 + h + 5
	// y3 := y2 + h + 5
	// y4 := y3 + h + 5

	// app.btn_change_x_plus = win.get_or_panic[ui.Button]('btn_change_x_plus')
	// app.btn_change_x_plus.set_pos(x1, y1)
	// app.btn_change_x_plus.propose_size(w, h)

	// app.btn_change_x_minus = win.get_or_panic[ui.Button]('btn_change_x_minus')
	// app.btn_change_x_minus.set_pos(x2, y1)
	// app.btn_change_x_minus.propose_size(w, h)

	// app.btn_change_y_plus = win.get_or_panic[ui.Button]('btn_change_y_plus')
	// app.btn_change_y_plus.set_pos(x1, y2)
	// app.btn_change_y_plus.propose_size(w, h)

	// app.btn_change_y_minus = win.get_or_panic[ui.Button]('btn_change_y_minus')
	// app.btn_change_y_minus.set_pos(x2, y2)
	// app.btn_change_y_minus.propose_size(w, h)

	// app.btn_change_width_plus = win.get_or_panic[ui.Button]('btn_change_width_plus')
	// app.btn_change_width_plus.set_pos(x1, y3)
	// app.btn_change_width_plus.propose_size(w, h)

	// app.btn_change_width_minus = win.get_or_panic[ui.Button]('btn_change_width_minus')
	// app.btn_change_width_minus.set_pos(x2, y3)
	// app.btn_change_width_minus.propose_size(w, h)

	// app.btn_change_height_plus = win.get_or_panic[ui.Button]('btn_change_height_plus')
	// app.btn_change_height_plus.set_pos(x1, y4)
	// app.btn_change_height_plus.propose_size(w, h)

	// app.btn_change_height_minus = win.get_or_panic[ui.Button]('btn_change_height_minus')
	// app.btn_change_height_minus.set_pos(x2, y4)
	// app.btn_change_height_minus.propose_size(w, h)
}

fn (mut field Field) place_new_piece(player_nr u8, x int, y int, piece_type PieceType) {
	field.pieces << Piece{
		player_nr: player_nr
		int_x: x
		int_y: y
		init_int_x: x
		init_int_y: y
		piece_type: piece_type
	}
	piece := &(field.pieces[field.pieces.len - 1])
	field.players[player_nr - 1].pieces << piece
	field.field_array[field.m_cols * y + x] = player_nr
	field.field_array_piece[field.m_cols * y + x] = piece
}

fn (mut field Field) define_diff_move_per_player_piece(
	side_move_possible bool,
	diag_move_possible bool,
	backward_move_possible bool,
	diag_backward_move_possible bool,
) {
	// define the possible moves for player 1
	mut player1 := &field.players[0]
	player1.map_piece_type_to_possible_diff_move[.normal] = []DiffMove{}
	player1.map_piece_type_to_possible_diff_move[.king] = []DiffMove{}
	// {
	// 	mut arr := []DiffMove{}
	// 	player1.arr_possible_diff_move << arr
	// 	player1.map_piece_type_to_possible_diff_move[.normal] = &arr
	// }
	// {
	// 	mut arr := []DiffMove{}
	// 	player1.arr_possible_diff_move << arr
	// 	player1.map_piece_type_to_possible_diff_move[.king] = &arr
	// }

	mut player2 := &field.players[1]
	player2.map_piece_type_to_possible_diff_move[.normal] = []DiffMove{}
	player2.map_piece_type_to_possible_diff_move[.king] = []DiffMove{}
	// {
	// 	mut arr := []DiffMove{}
	// 	player2.arr_possible_diff_move << arr
	// 	player2.map_piece_type_to_possible_diff_move[.normal] = &arr
	// }
	// {
	// 	mut arr := []DiffMove{}
	// 	player2.arr_possible_diff_move << arr
	// 	player2.map_piece_type_to_possible_diff_move[.king] = &arr
	// }


	mut y_dir_mult := 1
	{
		mut diff_move_arr := &(player1.map_piece_type_to_possible_diff_move[.normal])
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
		mut diff_move_arr := &(player1.map_piece_type_to_possible_diff_move[.king])
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
		mut diff_move_arr := &(player2.map_piece_type_to_possible_diff_move[.normal])
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
		mut diff_move_arr := &(player2.map_piece_type_to_possible_diff_move[.king])
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

fn (field &Field) draw_field_cells(mut d ui.DrawDevice, c &ui.CanvasLayout) {
	inner_field_w := field.m_cols * field.cell_w + field.cell_x_space * (field.m_cols + 1)
	inner_field_h := field.n_rows * field.cell_h + field.cell_y_space * (field.n_rows + 1)

	c.draw_device_rect_surrounded(d, field.field_abs_x, field.field_abs_y, inner_field_w, inner_field_h, field.field_frame_thickness, field.field_frame_color)
	
	start_x := field.field_abs_x + field.cell_x_space // add later the field space to the cells too!
	start_y := field.field_abs_y + field.cell_y_space // add later the field space to the cells too!

	// println('before for loops app.txtfld_command: ${app.txtfld_command}')

	for int_x in 0..field.m_cols {
		for int_y in 0..field.n_rows {
			int_color := (int_x + int_y) % 2

			c.draw_device_rect_filled(d,
				start_x + (field.cell_x_space + field.cell_w) * int_x,
				start_y + (field.cell_y_space + field.cell_h) * int_y,
				field.cell_w,
				field.cell_h,
				field.cell_colors[int_color]
			)

			c.draw_device_rect_surrounded(d,
				start_x + (field.cell_x_space + field.cell_w) * int_x + field.cell_frame_thickness,
				start_y + (field.cell_y_space + field.cell_h) * int_y + field.cell_frame_thickness,
				field.cell_w - 2 * field.cell_frame_thickness,
				field.cell_h - 2 * field.cell_frame_thickness,
				field.cell_frame_thickness,
				field.cell_frame_colors[int_color],
			)
		}
	}
}

fn (field &Field) draw_field_pieces(mut d ui.DrawDevice, c &ui.CanvasLayout) {
	start_x := field.field_abs_x + field.cell_x_space // add later the field space to the cells too!
	start_y := field.field_abs_y + field.cell_y_space // add later the field space to the cells too!

	start_piece_x := start_x + field.cell_w / 2
	start_piece_y := start_y + field.cell_h / 2
	piece_to_piece_w := field.cell_w + field.cell_x_space
	piece_to_piece_h := field.cell_h + field.cell_y_space

	shared mouse_action := &field.parent_app.mouse_action

	// println('before for loop pieces app.txtfld_command: ${app.txtfld_command}')

	for piece in field.pieces {
		// println('piece: ${piece}, app.txtfld_command: ${app.txtfld_command}')
		lock mouse_action {
			if mouse_action.is_piece_clicked && mouse_action.piece == piece {
				continue
			}
		}

		temp_color := field.piece_colors[piece.player_nr - 1]
		temp_color_frame := field.piece_frame_colors[piece.player_nr - 1]

		piece_center_x := start_piece_x + piece.int_x * piece_to_piece_w
		piece_center_y := start_piece_y + piece.int_y * piece_to_piece_h

		c.draw_device_circle_filled(d,
			piece_center_x,
			piece_center_y,
			field.piece_radius,
			temp_color_frame,
		)
		c.draw_device_circle_filled(d,
			piece_center_x,
			piece_center_y,
			field.piece_radius - field.piece_frame_thickness,
			temp_color,
		)
	}

	lock mouse_action {
		if mouse_action.is_piece_clicked {
			temp_color := field.piece_colors[mouse_action.piece.player_nr - 1]
			temp_color_frame := field.piece_frame_colors[mouse_action.piece.player_nr - 1]

			// Draw the moving piece as last!
			piece_center_x := mouse_action.mouse_x
			piece_center_y := mouse_action.mouse_y

			c.draw_device_circle_filled(d,
				piece_center_x,
				piece_center_y,
				field.piece_radius,
				temp_color_frame,
			)
			c.draw_device_circle_filled(d,
				piece_center_x,
				piece_center_y,
				field.piece_radius - field.piece_frame_thickness,
				temp_color,
			)
		}
	}
}

fn (app &App) draw_field(mut d ui.DrawDevice, c &ui.CanvasLayout) {
	// println('start of draw_field app.txtfld_command: ${app.txtfld_command}')

	field := app.field
	field.draw_field_cells(mut d, c)

	// println('after for loops app.txtfld_command: ${app.txtfld_command}')

	if app.field.show_field {
		field.draw_field_pieces(mut d, c)
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
}

fn (mut app App) on_mouse_down_main(window &ui.Window, mouse_event ui.MouseEvent) {
	println('Mouse Down')
	println('- mouse_event.x: ${mouse_event.x}')
	println('- mouse_event.y: ${mouse_event.y}')
	println('- mouse_event.button: ${mouse_event.button}')
	println('- mouse_event.action: ${mouse_event.action}')
	println('- mouse_event.mods: ${mouse_event.mods}')

	field := &app.field
	if field.show_field == true {
		field_w := field.cell_x_space * (field.m_cols + 1) + field.cell_w * field.m_cols
		field_h := field.cell_y_space * (field.n_rows + 1) + field.cell_h * field.n_rows

		mouse_rel_x := mouse_event.x - field.field_abs_x
		mouse_rel_y := mouse_event.y - field.field_abs_y

		println('field_w: ${field_w}, field_h: ${field_h}')
		println('mouse_rel_x: ${mouse_rel_x}, mouse_rel_y: ${mouse_rel_y}')

		if
			(mouse_rel_x < 0) || (mouse_rel_x >= field_w) ||
			(mouse_rel_y < 0) || (mouse_rel_y >= field_h) {
			println('Ignore mouse click!')
			return
		}

		int_x := field.lut_mouse_rel_pos_int_x[mouse_rel_x]
		int_y := field.lut_mouse_rel_pos_int_y[mouse_rel_y]

		// find piece with the position! TODO: implement a map of coordinates
		shared mouse_action := &app.mouse_action
		for piece in &field.pieces {
			if piece.int_x == int_x && piece.int_y == int_y && piece.player_nr == field.current_player.player_nr {
				lock mouse_action {
					mouse_action.is_piece_clicked = true
					mouse_action.piece = piece
					mouse_action.mouse_x = mouse_event.x
					mouse_action.mouse_y = mouse_event.y
				}
				break
			}
		}

		lock mouse_action {
			if mouse_action.is_piece_clicked {
				println('Piece was clicked! piece: ${mouse_action.piece}')
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
	field := &app.field

	if field.show_field == true {
		shared mouse_action := &app.mouse_action
		lock mouse_action {
			if mouse_action.is_piece_clicked {
				mouse_action.mouse_x = int(mouse_event.x)
				mouse_action.mouse_y = int(mouse_event.y)
			}
		}
	}
}

fn (mut app App) on_mouse_up_main(window &ui.Window, mouse_event ui.MouseEvent) {
	println('Mouse Up')
	println('- mouse_event.x: ${mouse_event.x}')
	println('- mouse_event.y: ${mouse_event.y}')
	println('- mouse_event.button: ${mouse_event.button}')
	println('- mouse_event.action: ${mouse_event.action}')
	println('- mouse_event.mods: ${mouse_event.mods}')

	mut field := &app.field
	if field.show_field == true{
		field_w := field.cell_x_space * (field.m_cols + 1) + field.cell_w * field.m_cols
		field_h := field.cell_y_space * (field.n_rows + 1) + field.cell_h * field.n_rows

		mouse_rel_x := mouse_event.x - field.field_abs_x
		mouse_rel_y := mouse_event.y - field.field_abs_y

		println('field_w: ${field_w}, field_h: ${field_h}')
		println('mouse_rel_x: ${mouse_rel_x}, mouse_rel_y: ${mouse_rel_y}')

		// println('app.field.lut_mouse_rel_pos_int_x: ${app.field.lut_mouse_rel_pos_int_x}')
		// println('app.field.lut_mouse_rel_pos_int_y: ${app.field.lut_mouse_rel_pos_int_y}')

		shared mouse_action := &app.mouse_action

		if
			(mouse_rel_x < 0) || (mouse_rel_x >= field_w) ||
			(mouse_rel_y < 0) || (mouse_rel_y >= field_h) {
			println('Ignore mouse click!')
		} else {
			new_x := field.lut_mouse_rel_pos_int_x[mouse_rel_x]
			new_y := field.lut_mouse_rel_pos_int_y[mouse_rel_y]

			println('Piece should be placed at: new_x: ${new_x}, new_y: ${new_y}')

			lock mouse_action {
				if mouse_action.is_piece_clicked {
					mut piece := mouse_action.piece
					
					// make a check, if piece was possible to move there
					curr_x := piece.int_x
					curr_y := piece.int_y
					key := '${curr_x},${curr_y},${new_x},${new_y}'
					println('move piece, key: ${key}')

					if field.field_array[field.m_cols * new_y + new_x] == 0 && key in field.map_curr_pos_to_next_possible_moves {
						field.field_array[field.m_cols * new_y + new_x] = piece.player_nr
						field.field_array[field.m_cols * piece.int_y + piece.int_x] = 0
						field.field_array_piece[field.m_cols * new_y + new_x] = piece
						field.field_array_piece[field.m_cols * piece.int_y + piece.int_x] = unsafe { nil }
						piece.int_x = new_x
						piece.int_y = new_y
						field.current_player = field.map_player_next[field.current_player.player_nr]
						field.empty_and_fill_map_curr_pos_to_next_possible_moves()
						app.txtfld_player_turn.text = 'Turn Player Nr. ${field.current_player.player_nr}'
					}
				}

				println('field.field_array: ${field.field_array}')
			}
		}

		lock mouse_action {
			if mouse_action.is_piece_clicked {
				mouse_action.is_piece_clicked = false
				mouse_action.piece = unsafe { nil }
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
	piece.int_x = piece.init_int_x
	piece.int_y = piece.init_int_y
}

fn (mut field Field) empty_and_fill_map_curr_pos_to_next_possible_moves() {
	mut map_curr_pos_to_next_possible_moves := &(field.map_curr_pos_to_next_possible_moves)
	for key in map_curr_pos_to_next_possible_moves.keys() {
		map_curr_pos_to_next_possible_moves.delete(key)
	}

	map_piece_type_to_possible_diff_move := &(field.current_player.map_piece_type_to_possible_diff_move)
	pieces := &(field.current_player.pieces)

	print('all pieces: ')
	for piece in field.pieces {
		print('(${piece.int_x}, ${piece.int_y}), ')
	}
	println('')
	print('pieces of player_nr ${field.current_player.player_nr}: ')
	for piece in pieces {
		print('(${piece.int_x}, ${piece.int_y}), ')
	}
	println('')

	for piece in pieces {
		curr_x := unsafe { piece.int_x }
		curr_y := unsafe { piece.int_y }
		possible_diff_move := &(unsafe { map_piece_type_to_possible_diff_move[ piece.piece_type] })
		for diff_move in possible_diff_move {
			new_x := curr_x + diff_move.diff_x
			new_y := curr_y + diff_move.diff_y

			// check bounds
			if (
				new_x < 0 || new_x >= field.m_cols ||
				new_y < 0 || new_y >= field.n_rows
			) {
				continue
			}

			// check collisions with othe pieces, depence on the rulling
			// TODO: add one more check, if piece can be jumped over!
			if field.field_array[field.m_cols * new_y + new_x] != 0 {
				continue
			}

			unsafe {
				map_curr_pos_to_next_possible_moves['${curr_x},${curr_y},${new_x},${new_y}'] = NextPossibleMove{
					int_x: new_x
					int_y: new_y
				}
			}
		}
	}

	println('map_curr_pos_to_next_possible_moves.keys(): ${map_curr_pos_to_next_possible_moves.keys()}')
}

fn (mut field Field) start_new_game() {
	field.show_field = true
	for i in 0..field.field_array.len {
			field.field_array[i] = 0
	}
	for mut piece in field.pieces {
		piece.reset_pos()
		field.field_array[field.m_cols * piece.int_y + piece.int_x] = piece.player_nr
	}
	field.current_player = &(field.players[0])
	field.parent_app.txtfld_player_turn.text = 'Turn Player Nr. ${field.current_player.player_nr}'

	// fill all possible next moves
	field.empty_and_fill_map_curr_pos_to_next_possible_moves()
}

fn (mut app App) btn_do_action_on_click(button &ui.Button) {
	println('fn btn_do_action_on_click() called!!!!!!!!!!!!!!!!')
	// println('button: ${button}')
}

fn (mut app App) btn_start_game_on_click(button &ui.Button) {
	app.field.start_new_game()
	
	println('fn btn_start_game_on_click() called!!!!!!!!!!!!!!!!')
}

fn (mut app App) btn_end_game_on_click(button &ui.Button) {
	app.field.show_field = false
	app.txtfld_player_turn.text = ''
	println('fn btn_end_game_on_click() called!!!!!!!!!!!!!!!!')
}

// fn (mut app App) btn_reset_field_on_click(button &ui.Button) {
// 	mut field := &app.field
// 	if field.show_field == true {
// 		for i in 0..field.field_array.len {
// 			field.field_array[i] = 0
// 		}
// 		for mut piece in field.pieces {
// 			piece.reset_pos()
// 			field.field_array[field.m_cols * piece.int_y + piece.int_x] = piece.player_nr
// 		}
// 	}
// 	println('fn btn_reset_field_on_click() called!!!!!!!!!!!!!!!!')
// }

// fn (mut app App) btn_change_x_plus_on_click(button &ui.Button) {
// 	println('Clicked "btn_change_x_plus_on_click"')
// }

// fn (mut app App) btn_change_x_minus_on_click(button &ui.Button) {
// 	println('Clicked "btn_change_x_minus_on_click"')
// }

// fn (mut app App) btn_change_y_plus_on_click(button &ui.Button) {
// 	println('Clicked "btn_change_y_plus_on_click"')
// }

// fn (mut app App) btn_change_y_minus_on_click(button &ui.Button) {
// 	println('Clicked "btn_change_y_minus_on_click"')
// }

// fn (mut app App) btn_change_width_plus_on_click(button &ui.Button) {
// 	println('Clicked "btn_change_width_plus_on_click"')
// }

// fn (mut app App) btn_change_width_minus_on_click(button &ui.Button) {
// 	println('Clicked "btn_change_width_minus_on_click"')
// }

// fn (mut app App) btn_change_height_plus_on_click(button &ui.Button) {
// 	println('Clicked "btn_change_height_plus_on_click"')
// }

// fn (mut app App) btn_change_height_minus_on_click(button &ui.Button) {
// 	println('Clicked "btn_change_height_minus_on_click"')
// }
