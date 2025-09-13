pub fn sqrt_u64(n: u64) -> u64 {
	if n == 0 {
		return 0;
	} else if n == 1 {
		return 1;
	}

	let mut n_sqrt_prev: u64 = (1 + n / 1) / 2;
	let mut n_sqrt: u64 = (n_sqrt_prev + n / n_sqrt_prev) / 2;
	
	loop {
		let n_sqrt_prev_2: u64 = n_sqrt_prev;
		n_sqrt_prev = n_sqrt;
		n_sqrt = (n_sqrt + n / n_sqrt) / 2;

		if n_sqrt_prev_2 == n_sqrt || n_sqrt_prev == n_sqrt {
			return n_sqrt_prev;
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_sqrt_u64_simple() {
		for n in 0..11 {
			let n_u64: u64 = n as u64;
			let result: u64 = sqrt_u64(n_u64 * n_u64);

			assert_eq!(n_u64, result);
		}
	}
}
