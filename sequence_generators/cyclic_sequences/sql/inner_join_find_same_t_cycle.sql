SELECT
	t1.modulo,
	t1.t_cycle,
	t1.t_factor t_factor_1,
	t1.factor_len factor_len_1,
	t2.t_factor t_factor_2,
	t2.factor_len factor_len_2
FROM
	cyclic_2_factor_sequence t1
INNER JOIN
	cyclic_2_factor_sequence t2
ON
	t1.modulo = t2.modulo AND
	t1.t_cycle = t2.t_cycle AND
	t1.factor_len < t2.factor_len
ORDER BY
	t1.modulo, t1.factor_len, t1.t_factor
;

SELECT
	t1.modulo,
	t1.factor_len factor_len_1,
	t2.factor_len factor_len_2,
	COUNT(t1.factor_len) amount_factor_pair
FROM
	cyclic_2_factor_sequence t1
INNER JOIN
	cyclic_2_factor_sequence t2
ON
	t1.modulo = t2.modulo AND
	t1.t_cycle = t2.t_cycle AND
	t1.factor_len < t2.factor_len
GROUP BY
	t1.modulo, t1.factor_len, t2.factor_len
ORDER BY
	t1.modulo, t1.factor_len, t2.factor_len
;
