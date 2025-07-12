import math

fn calc_next_prime(l_p []i64, l_n []i64, thread_nr int) []i64 {
	mut l := []i64{}
	l << l_p[l_p.len - 2] * l_p[l_p.len - 1]

	println('thread_nr: ${thread_nr}, l: ${l}')
	return l
}

fn main() {
	mut l_p := []i64{}
	l_p << 2
	l_p << 3
	l_p << 5
	mut threads := []thread []i64{}
	start_num := i64(7)
	end_num := start_num + i64(6) * 10
	mut i := i64(start_num)
	mut thread_nr := 0
	for i < end_num {
		l_p << i + 2
		threads << spawn calc_next_prime(l_p, [i + 0, i + 4], thread_nr)
		i += 6
		thread_nr++
	}
	// Join all tasks
	r := threads.wait()
	println('All jobs finished: ${r}')

	for i2 in i64(95)..i64(105) {
		n := math.sqrti(i2)
		println('i2: ${i2}, n: ${n}')
	}
}
