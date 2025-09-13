CREATE TABLE "base_pairs_prim_number_n_prime_p" (
	"base_num"	INTEGER NOT NULL,
	"base_prime"	INTEGER NOT NULL,
	"n"	INTEGER NOT NULL,
	"n_prim"	INTEGER NOT NULL,
	"prime"	INTEGER NOT NULL,
	"prime_prim"	INTEGER NOT NULL,
	UNIQUE(base_num, base_prime, n, n_prim)
);

