import arrays
import os
import time
import math
import rand
import rand.mt19937

struct PRNGOwn {
mut:
	rng rand.PRNG
}

__global (
	max_val = u64(1024*1024*1024*1024*1024*1024*8)
	max_val_half = max_val / u64(2)
)
fn new() PRNGOwn {
	return PRNGOwn {
		rng: &rand.PRNG(mt19937.MT19937RNG{})
	}
}

fn (mut self PRNGOwn) next_f64() f64 {
	return (f64(self.rng.u64() % max_val) - max_val_half) / f64(max_val_half) * f64(2)
}

fn main() {
	mut rng_own := new()

	n := u64(1024)
	len_i := n
	len_j := n
	len_k := n

	len_1 := len_i * len_k
	len_2 := len_k * len_j
	len_3 := len_i * len_j

	mut v_1 := []f64{len: int(len_1)}
	mut v_2 := []f64{len: int(len_2)}
	mut v_3 := []f64{len: int(len_3)}

	for i := u64(0); i < len_1; i += 1 {
		v_1[i] = rng_own.next_f64()
	}

	for i := u64(0); i < len_2; i += 1 {
		v_2[i] = rng_own.next_f64()
	}

	sw := time.new_stopwatch()
	for y := u64(0); y < len_i; y += 1 {
		for z := u64(0); z < len_k; z += 1 {
			for x := u64(0); x < len_j; x += 1 {
				v_3[y * len_j + x] += v_1[y * len_k + z] * v_2[z * len_j + x]
			}
		}
	}
	elapsed := sw.elapsed().microseconds()

	println('elapsed: ${elapsed} us')
}
