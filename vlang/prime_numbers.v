import arrays
import os
import time

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
	jump_mult_primes := u8(os.args[3].int())
	
	mut l_diff := []f64{}
	for i_round := 0; i_round < amount_tests; i_round += 1 {
		sw := time.new_stopwatch()
		mut l := []u64{}
		mut p := u64(0)
		mut l_jump := []u64{}
		mut argmax_p_pow_2 := 0
		mut i_start := 0
		mut l_jump_length := 0

		if jump_mult_primes == 2 {
			// jump with primes 2, 3
			l << 2 l << 3 l << 5
			p = u64(7)
			l_jump << 4 l_jump << 2
			argmax_p_pow_2 = 1
			i_start = 2
			l_jump_length = 2
		} else if jump_mult_primes == 3 {
			// // jump with primes 2, 3, 5
			l << 2 l << 3 l << 5 l << 7 l << 11 l << 13 l << 17 l << 19 l << 23 l << 29
			p = u64(31)
			l_jump << 6 l_jump << 4 l_jump << 2 l_jump << 4
			l_jump << 2 l_jump << 4 l_jump << 6 l_jump << 2
			argmax_p_pow_2 = 3
			i_start = 3
			l_jump_length = 8
		} else {
			assert false
		}

		mut i_jump := 0
		mut p_pow_2 := l[argmax_p_pow_2] * l[argmax_p_pow_2]

		for p < max_p {
			// is p a prime number? let's test this
			mut is_prime := true
			for i := i_start; i <= argmax_p_pow_2; i += 1 {
				if p % l[i] == 0 {
					is_prime = false
					break
				}
			}

			if is_prime {
				l << p
			}

			p += l_jump[i_jump]
			i_jump = (i_jump + 1) % l_jump_length

			if p > p_pow_2 {
				argmax_p_pow_2 += 1
				p_pow_2 = l[argmax_p_pow_2] * l[argmax_p_pow_2]
			}
		}

		elapsed_time := sw.elapsed().seconds()
		l_diff << elapsed_time

		if i_round == 0 {
			mut f := os.create('/tmp/primes_n_${max_p}_jump_mult_primes_${jump_mult_primes}_vlang.txt') or { panic(err) }
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
