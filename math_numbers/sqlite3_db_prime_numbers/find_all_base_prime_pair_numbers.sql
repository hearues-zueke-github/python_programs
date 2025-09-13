SELECT
	PairNumPrime.base as base_num,
	PairPrimeNum.base AS base_prime,
	PairNumPrime.n,
	PairNumPrime.n_prim,
	PairPrimeNum.prime,
	PairPrimeNum.prime_prim
FROM base_pairs_from_number_n_to_prime_p AS PairNumPrime
INNER JOIN base_pairs_from_prime_p_to_number_n AS PairPrimeNum
WHERE
	PairNumPrime.prime_n = PairPrimeNum.prime AND
	PairNumPrime.prime_n_prim = PairPrimeNum.prime_prim AND
	PairNumPrime.n > 10859307
;
