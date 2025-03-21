import ui
import gx

const win_width = 750
const win_height = 500

// TODO: do a refactoring of current state
// TODO: add move validating for each player
// TODO: add move count and history of each player played
// TODO: add types of different move sets for each player
// TODO: create a simple neural network for weights, needed for genetic algorithm
// TODO: implement self play of neural network bots
// TODO: implement serialization of best bots data for useage for new games
// TODO: find all possible next moves for the current player

@[heap]
struct App {
mut:
	x int
	y int
	window ui.Window
	field Field
	txtfld_command TextField
	tb_command &ui.TextBox = unsafe { nil }
	btn_do_action &ui.Button = unsafe { nil } // only a test button so far
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
	pieces []Piece
	field_array []u8
	field_array_piece []&Piece
	lut_mouse_rel_pos_int_x []int
	lut_mouse_rel_pos_int_y []int
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

	field.players << [Player{player_nr: 1}, Player{player_nr: 2}]
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


	for i in 0..field.m_cols {
		app.field.place_new_piece(1, i, 1)
		app.field.place_new_piece(2, i, field.n_rows - 2)
	}

	app.txtfld_command = TextField{
		x: 461
		y: 377
		text: 'Command:'
		color: gx.rgb(0, 0, 0)
		size: 20
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
			ui.button(id: 'btn_do_action', z_index: 2, text: 'Do Action',radius: 5, border_color: gx.rgb(0, 128, 0), bg_color: gx.rgb(192, 192, 128))
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


	// println('in win_init app.txtfld_command: ${app.txtfld_command}')
}

fn (mut field Field) place_new_piece(player_nr u8, x int, y int) {
	piece := Piece{
		player_nr: player_nr
		int_x: x
		int_y: y
		piece_type: .normal
	}
	field.pieces << &piece
	field.players[player_nr - 1].pieces << &piece
	field.field_array[field.m_cols * y + x] = player_nr
	field.field_array_piece[field.m_cols * y + x] = &piece
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

fn (app &App) draw_field(mut d ui.DrawDevice, c &ui.CanvasLayout) {
	// println('start of draw_field app.txtfld_command: ${app.txtfld_command}')

	field := app.field
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

	// println('after for loops app.txtfld_command: ${app.txtfld_command}')

	start_piece_x := start_x + field.cell_w / 2
	start_piece_y := start_y + field.cell_h / 2
	piece_to_piece_w := field.cell_w + field.cell_x_space
	piece_to_piece_h := field.cell_h + field.cell_y_space

	shared mouse_action := &app.mouse_action

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
}

fn (mut app App) on_mouse_down_main(window &ui.Window, mouse_event ui.MouseEvent) {
	println('Mouse Down')
	println('- mouse_event.x: ${mouse_event.x}')
	println('- mouse_event.y: ${mouse_event.y}')
	println('- mouse_event.button: ${mouse_event.button}')
	println('- mouse_event.action: ${mouse_event.action}')
	println('- mouse_event.mods: ${mouse_event.mods}')

	field := &app.field

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
		if piece.int_x == int_x && piece.int_y == int_y {
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

fn (mut app App) on_mouse_move_main(window &ui.Window, mouse_event ui.MouseMoveEvent) {
	// println('Mouse Move')
	// println('- mouse_event.x: ${mouse_event.x}')
	// println('- mouse_event.y: ${mouse_event.y}')
	// println('- mouse_event.mouse_button: ${mouse_event.mouse_button}')

	shared mouse_action := &app.mouse_action
	lock mouse_action {
		if mouse_action.is_piece_clicked {
			mouse_action.mouse_x = int(mouse_event.x)
			mouse_action.mouse_y = int(mouse_event.y)
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

	field_w := field.cell_x_space * (field.m_cols + 1) + field.cell_w * field.m_cols
	field_h := field.cell_y_space * (field.n_rows + 1) + field.cell_h * field.n_rows

	mouse_rel_x := mouse_event.x - field.field_abs_x
	mouse_rel_y := mouse_event.y - field.field_abs_y

	println('field_w: ${field_w}, field_h: ${field_h}')
	println('mouse_rel_x: ${mouse_rel_x}, mouse_rel_y: ${mouse_rel_y}')

	shared mouse_action := &app.mouse_action

	if
		(mouse_rel_x < 0) || (mouse_rel_x >= field_w) ||
		(mouse_rel_y < 0) || (mouse_rel_y >= field_h) {
		println('Ignore mouse click!')
	} else {
		int_x := field.lut_mouse_rel_pos_int_x[mouse_rel_x]
		int_y := field.lut_mouse_rel_pos_int_y[mouse_rel_y]

		println('Piece should be placed at: int_x: ${int_x}, int_y: ${int_y}')

		lock mouse_action {
			mut piece := mouse_action.piece
			
			if field.field_array[field.m_cols * int_y + int_x] == 0 {
				field.field_array[field.m_cols * int_y + int_x] = piece.player_nr
				field.field_array[field.m_cols * piece.int_y + piece.int_x] = 0
				field.field_array_piece[field.m_cols * int_y + int_x] = piece
				field.field_array_piece[field.m_cols * piece.int_y + piece.int_x] = unsafe { nil }
				piece.int_x = int_x
				piece.int_y = int_y
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
					}
					println('- btn_do_action: ${btn_do_action}')
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
