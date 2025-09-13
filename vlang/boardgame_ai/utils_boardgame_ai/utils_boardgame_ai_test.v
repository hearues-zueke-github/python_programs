import utils_boardgame_ai

struct StructA {
	a int
	b string
}

fn StructA.get_string(obj &StructA) string {
	return '(' + unsafe { (&obj).a.str() } + ', ' + unsafe { (&obj).b } + ')'
}

fn test_concat_string_from_array_empty_array() {
	mut arr := []StructA{}

	str_concat := utils_boardgame_ai.concat_string_from_array(arr, StructA.get_string)
	println('str_concat: ${str_concat}')
	assert str_concat == '[]'
}

fn test_concat_string_from_array_simple() {
	mut arr := []StructA{}

	arr << StructA {
		a: 10
		b: '45'
	}

	arr << StructA {
		a: -56
		b: 'test'
	}

	str_concat := utils_boardgame_ai.concat_string_from_array(arr, StructA.get_string)
	println('str_concat: ${str_concat}')
	assert str_concat == '[(10, 45), (-56, test), ]'
}
