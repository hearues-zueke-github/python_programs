import arrays
import os
import time
import math

fn sqrt_int(v_ u64) u64 {
	mut v1 := v_ / 2
	mut v2 := (v1 + v_ / v1) / 2

	for i := 1; i < 10; i += 1 {
		v3 := (v2 + v_ / v2) / 2

		if v1 == v3 || v2 == v3 {
			break
		}

		v1 = v2
		v2 = v3
	}

	return v2
}

fn main() {
	argc := os.args.len
	assert argc >= 3, "usage: ./prime_numbers <max_p> <amount_tests>"
	
	max_p := u64(os.args[1].int())
	amount_tests := u64(os.args[2].int())
	
	mut l_diff := []f64{}
	for i_round := 0; i_round < amount_tests; i_round += 1 {
		sw := time.new_stopwatch()
		mut l := []u64{} l << 2 l << 3 l << 5
		mut l_jump := []u64{} l_jump << 4 l_jump << 2
		
		mut i_jump := 0
		mut p := u64(7)

		// mut max_i := 1
		// mut p_pow_2 := math.powi(i64(l[max_i]), 2)

		for p < max_p {
			max_sqrt_p := sqrt_int(p) + 1

			// is p a prime number? let's test this
			mut is_prime := true
			// for i := 0; i <= max_i; i += 1 {
			for i := 0; l[i] < max_sqrt_p; i += 1 {
				if p % l[i] == 0 {
					is_prime = false
					break
				}
			}

			if is_prime {
				l << p
			}

			p += l_jump[i_jump]
			i_jump = (i_jump + 1) % 2

			// if p > p_pow_2 {
			// 	max_i += 1
			// 	p_pow_2 = math.powi(i64(l[max_i]), 2)
			// }
		}

		elapsed_time := sw.elapsed().seconds()
		l_diff << elapsed_time

		if i_round == 0 {
			mut f := os.create('/tmp/primes_n_${max_p}_vlang.txt') or { panic(err) }
			defer {
				f.close()
			}

			for v in l {
				f.write("${v},".bytes()) or {
					println("Error at writting the file!")
					break
				}
			}
		}

		// println("l: ${l}")
	}

	average_time := (arrays.sum<f64>(l_diff) or {0}) / l_diff.len

	println('l_diff: ${l_diff}')
	println('average_time: ${average_time}')
}
