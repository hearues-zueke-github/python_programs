import game_board { GameBoard, PieceIdPosMoveScore }
import utils_boardgame_ai { concat_string_from_array }

fn test_generate_new_game_board() {
	gb := GameBoard.new(4, 5)
	gb.print_board_state()
}

fn test_generate_next_moves_for_current_player() {
	gb := GameBoard.new(4, 5)
	gb.print_board_state()

	arr_pos_move_score := gb.generate_for_current_player_next_possible_moves()
	// println('arr_pos_move_score[0]: ${PieceIdPosMoveScore.get_string(arr_pos_move_score[0])}')
	str_arr_pos_move_score := concat_string_from_array(arr_pos_move_score, PieceIdPosMoveScore.get_string)
	println('arr_pos_move_score: ${str_arr_pos_move_score}')
}
