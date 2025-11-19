CREATE TABLE IF NOT EXISTS cyclic_2_factor_sequence (
	"cycle_len" INTEGER NOT NULL,
	"t_cycle" TEXT NOT NULL,
	"t_factor" TEXT NOT NULL,
	UNIQUE(cycle_len, t_cycle, t_factor)
);
