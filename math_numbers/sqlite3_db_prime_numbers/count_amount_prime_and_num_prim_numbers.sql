SELECT
	T1.base AS base,
	T1.count_prime AS count_prime,
	T2.count_num AS count_num
FROM (
	SELECT
		base,
		count(base) AS count_prime
	FROM base_pairs_from_prime_p_to_number_n AS PairPrimeNum
	GROUP BY base
) AS T1
INNER JOIN (
	SELECT
		base,
		count(base) AS count_num
	FROM base_pairs_from_number_n_to_prime_p AS PairNumPrime
	GROUP BY base
) AS T2
WHERE
	T1.base = T2.base
;
