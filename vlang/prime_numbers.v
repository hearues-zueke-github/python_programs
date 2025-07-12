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
	assert argc >= 3, "usage: ./prime_numbers <max_amount_p> <amount_tests>"
	
	max_amount_p := u64(os.args[1].int())
	amount_tests := u64(os.args[2].int())
	
	mut l_diff_time := []f64{}
	for i_round := 0; i_round < amount_tests; i_round += 1 {
		sw := time.new_stopwatch()
		mut l := []u64{cap: int(max_amount_p)}
		l << 2
		l << 3
		l << 5
		mut l_jump := []u64{} l_jump << 4 l_jump << 2
		amount_l_jump := l_jump.len
		mut i_jump := 0

		mut p := u64(l[l.len - 1] + l_jump[amount_l_jump - 1])
		mut amount_p := l.len

		mut max_i := 1
		mut max_p_i_pow_2 := l[max_i] * l[max_i]

		for amount_p < max_amount_p {
			// is p a prime number? let's test this
			mut is_prime := true
			for i := 0; i <= max_i; i += 1 {

				if p % l[i] == 0 {
					is_prime = false
					break
				}
			}

			if is_prime {
				l << p
				amount_p += 1
			}

			p += l_jump[i_jump]
			i_jump = (i_jump + 1) % amount_l_jump

			if p > max_p_i_pow_2 {
				max_i += 1
				max_p_i_pow_2 = l[max_i] * l[max_i]
			}
		}

		elapsed_time := sw.elapsed().seconds()
		l_diff_time << elapsed_time

		if i_round == 0 {
			mut f := os.create('/tmp/primes_n_max_amount_p_${max_amount_p}_vlang.txt') or { panic(err) }
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

	}

	average_time := (arrays.sum[f64](l_diff_time) or {0}) / l_diff_time.len

	println('l_diff_time: ${l_diff_time}')
	println('average_time: ${average_time}')
}
