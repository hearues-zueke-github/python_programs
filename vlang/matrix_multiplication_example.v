import vsl.vlas.internal.blas

fn main() {
	no_trans := blas.Transpose.no_trans

	m := int(2)
	n := int(3)
	k := int(4)

	alpha := f64(1.0)
	beta := f64(0.0)
	
	a := [f64(1),2,8,4, 5,7,1,3]
	b := [f64(1),2,3, 4,2,5, 7,8,9, 1,2,3]

	println('m: ${m}, n: ${n}, k: ${k}')
	println('alpha: ${alpha}, beta: ${beta}')
	println('a: ${a}')
	println('b: ${b}')
	
	lda := k
	ldb := n
	ldc := n

	mut c := []f64{len: m * n, init: 0.0}

	// do the matrix matrix multiplication!
	blas.dgemm(no_trans, no_trans, m, n, k, alpha, a, lda, b, ldb, beta, mut &c, ldc)

	println('c: ${c}')

	c_ref := [f64(69.0), 78.0, 97.0, 43.0, 38.0, 68.0]
	assert c == c_ref
}
