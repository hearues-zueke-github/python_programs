module utils_boardgame_ai

pub fn concat_string_from_array[T](arr []T, get_string fn (&T) string) string {
	mut result := '['

	for v in arr {
		result += get_string(&v) + ', '
	}

	result += ']'

	return result
}
