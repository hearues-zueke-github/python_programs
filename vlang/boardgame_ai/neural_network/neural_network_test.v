import rand
import rand.seed
import rand.pcg32

import neural_network { NeuralNetwork }

import arraybyte_serialization { CrossLangSerialization }

fn test_main() {
	// arr_layer := [int(3), 2]
	arr_layer := [i32(5), 6, 4, 3]
	arr_seed := seed.time_seed_array(pcg32.seed_len)

	mut nn := NeuralNetwork.new(arr_layer, arr_seed)

	nn.init_random_weights_biases()

	val := nn.rng.f64_in_range(-1, 1)!
	println('val: ${val}')

	println('arr_layer: ${arr_layer}')

	println('nn.arr_weights: ${nn.arr_weights}')
	println('nn.arr_biases: ${nn.arr_biases}')

	// mut y := []f64{len: int(arr_layer.last())}
	// nn.propagate_forward([f64(3), 4, 9, -1, 5], mut y)

	mut cross_lang_serialization := CrossLangSerialization.new()

	cross_lang_serialization.map_str_to_arr_i32['arr_layer'] = nn.arr_layer
	cross_lang_serialization.map_str_to_arr_u32['arr_seed'] = nn.arr_seed
	
	for i in 0..arr_layer.len - 1 {
		cross_lang_serialization.map_str_to_arr_f64['w${i}'] = nn.arr_weights[i]
		cross_lang_serialization.map_str_to_arr_f64['b${i}'] = nn.arr_biases[i]
	}

	amount_random_tests := int(100)
	mut arr_x := [][]f64{len: amount_random_tests}
	mut arr_y := [][]f64{len: amount_random_tests}

	amount_first_layer := int(nn.arr_layer.first())
	amount_last_layer := int(nn.arr_layer.last())

	for i in 0..amount_random_tests {
		mut x := []f64{len: amount_first_layer}
		mut y := []f64{len: amount_last_layer}

		for j in 0..amount_first_layer {
			x[j] = nn.rng.f64_in_range(-1, 1)!
		}

		for j in 0..amount_last_layer {
			y[j] = nn.rng.f64_in_range(-1, 1)!
		}

		nn.propagate_forward(x, mut y)

		arr_x[i] = x
		arr_y[i] = y
	}

	mut arr_x_one_vector := []f64{len: amount_random_tests * amount_first_layer}
	mut arr_y_one_vector := []f64{len: amount_random_tests * amount_last_layer}

	for i in 0..amount_random_tests {
		x := arr_x[i]
		y := arr_y[i]

		for j in 0..amount_first_layer {
			arr_x_one_vector[i * amount_first_layer + j] = x[j]
		}

		for j in 0..amount_last_layer {
			arr_y_one_vector[i * amount_last_layer + j] = y[j]
		}
	}

	cross_lang_serialization.map_str_to_arr_f64['arr_x_one_vector'] = arr_x_one_vector
	cross_lang_serialization.map_str_to_arr_f64['arr_y_one_vector'] = arr_y_one_vector

	file_path := '/tmp/nn_v_data.arrhex'
	cross_lang_serialization.save_data_to_file(file_path)!
}
