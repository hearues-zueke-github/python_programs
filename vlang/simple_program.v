import os

struct Point {
	x int
	y int
}

struct Line {
	p1 Point
	p2 Point
}

type ObjectSumType = Line | Point

fn main() {
	println("hello world")

	println(os.args)

	a := i32(123)
	assert a == i32(123)

	println("a: ${a}")

	mut l_vals := []int{len: 7, init: it * 3 + 1}
	println("l_vals: ${l_vals}")

	mut object_list := []ObjectSumType{}
	object_list << Point{1, 1}
	object_list << Line{
		p1: Point{3, 3}
		p2: Point{4, 4}
	}
	dump(object_list)

	mut l_vals_rep := l_vals.repeat(2)
	idx1 := int(1)
	idx2 := int(7)
	mut l_vals_rep_2 := l_vals_rep[idx1..idx2]
	println("l_vals_rep: ${l_vals_rep}")
	println("l_vals_rep_2: ${l_vals_rep_2}")	
}
