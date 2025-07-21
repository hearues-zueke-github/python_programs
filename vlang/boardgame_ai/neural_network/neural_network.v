module neural_network

import arrays
import rand
import rand.seed
import rand.pcg32

import vsl.vlas.internal.blas

pub struct NeuralNetwork {
pub:
	arr_layer []i32
	max_layer_size int
	arr_seed []u32
pub mut:
	rng rand.PRNG
	arr_weights [][]f64
	arr_biases [][]f64
}

pub fn NeuralNetwork.new(arr_layer []i32, arr_seed []u32) NeuralNetwork {
	return NeuralNetwork{
		arr_layer: arr_layer
		max_layer_size: arrays.max(arr_layer) or { 0 }
		arr_seed: arr_seed
		rng: rand.PRNG(pcg32.PCG32RNG{})
	}
}

pub fn (mut nn NeuralNetwork) init_random_weights_biases() {
	mut rng := &nn.rng
	rng.seed(nn.arr_seed)

	arr_layer := &nn.arr_layer
	mut arr_weights := &nn.arr_weights
	mut arr_biases := &nn.arr_biases
	unsafe { *arr_weights = [][]f64{len: nn.arr_layer.len - 1} }
	unsafe { *arr_biases = [][]f64{len: nn.arr_layer.len - 1} }
	for i in 1..nn.arr_layer.len {
		unsafe { arr_weights[i - 1] = []f64{len: int(arr_layer[i - 1] * arr_layer[i])} }
		unsafe { arr_biases[i - 1] = []f64{len: int(arr_layer[i])} }
	}

	for i in 0..arr_weights.len {
		mut weights := unsafe { arr_weights[i] }
		for j in 0..weights.len {
			weights[j] = rng.f64_in_range(-1, 1) or { f64(0) }
		}
	}

	for i in 0..arr_biases.len {
		mut biases := unsafe { arr_biases[i] }
		for j in 0..biases.len {
			biases[j] = rng.f64_in_range(-1, 1) or { f64(0) }
		}
	}
}

pub fn (nn &NeuralNetwork) propagate_forward(x []f64, mut y []f64) {
	if x.len != nn.arr_layer[0] || y.len != nn.arr_layer.last() {
		println('x.len != nn.arr_weights[0].len')
		return
	}

	mut arr_prev := []f64{len: nn.max_layer_size}
	mut arr_next := []f64{len: nn.max_layer_size}

	arrays.copy(mut arr_next, x)
	
	for i_layer in 0..(nn.arr_layer.len - 1) {
		arrays.copy(mut arr_prev, arr_next)

		m := nn.arr_layer[i_layer + 1]
		n := 1
		k := nn.arr_layer[i_layer]

		lda := k
		ldb := n
		ldc := n

		no_trans := blas.Transpose.no_trans
		blas.dgemm(no_trans, no_trans, m, n, k, 1.0, nn.arr_weights[i_layer], lda, arr_prev, ldb, 0.0, mut &arr_next, ldc)

		for i in 0..nn.arr_layer[i_layer + 1] {
			arr_next[i] += nn.arr_biases[i_layer][i]
		}

		println('-------------------------------')
		println('i_layer: ${i_layer}')
		println('arr_prev: ${arr_prev}')
		println('nn.arr_weights[i_layer]: ${nn.arr_weights[i_layer]}')
		println('nn.arr_biases[i_layer]: ${nn.arr_biases[i_layer]}')
		println('arr_next: ${arr_next}')
	}

	arrays.copy(mut y, arr_next)
}
