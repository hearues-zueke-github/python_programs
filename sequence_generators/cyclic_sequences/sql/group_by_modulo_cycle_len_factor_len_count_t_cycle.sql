SELECT
	modulo,
	cycle_len,
	factor_len,
	COUNT(t_cycle) count_t_cycle
FROM cyclic_2_factor_sequence
WHERE
-- 	factor_len = 9 and modulo = 13
-- 	modulo = 9
-- 	modulo = 9 and cycle_len = 64
	cycle_len = modulo * modulo
GROUP BY
	modulo, cycle_len, factor_len
ORDER BY
	modulo, factor_len, cycle_len
-- 	modulo, cycle_len, factor_len
	--COUNT(t_cycle) DESC, modulo, cycle_len, factor_len
;