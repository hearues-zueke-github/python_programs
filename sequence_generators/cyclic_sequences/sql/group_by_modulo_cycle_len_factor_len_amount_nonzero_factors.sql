SELECT
	modulo, cycle_len, factor_len, amount_nonzero_factors, COUNT(amount_nonzero_factors) count_amount_nonzero_factors
FROM
	cyclic_2_factor_sequence
-- WHERE
-- 	factor_len = 9
-- 	cycle_len = modulo*modulo
GROUP BY
	modulo, cycle_len, factor_len, amount_nonzero_factors
ORDER BY
	modulo, cycle_len, factor_len, amount_nonzero_factors
;

SELECT
	factor_amount, factor_len, modulo, cycle_len, COUNT(factor_len) count_factor_len
FROM
	cyclic_n_factor_sequence
-- WHERE
-- 	factor_amount = 2
-- 	cycle_len = modulo*modulo
GROUP BY
	factor_amount, factor_len, modulo, cycle_len
ORDER BY
	factor_amount, modulo, factor_len, cycle_len
-- 	factor_amount, factor_len, modulo, cycle_len
;

SELECT
	factor_amount, factor_len, modulo, cycle_len, amount_nonzero_factors, COUNT(factor_len) count_group_by
FROM
	cyclic_n_factor_sequence
-- WHERE
-- 	factor_amount = 2
-- 	cycle_len = modulo*modulo
GROUP BY
	factor_amount, factor_len, modulo, cycle_len, amount_nonzero_factors
ORDER BY
	factor_amount, modulo, factor_len, cycle_len, amount_nonzero_factors
-- 	factor_amount, factor_len, modulo, cycle_len, amount_nonzero_factors
;
