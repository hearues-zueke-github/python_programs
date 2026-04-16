#! /bin/bash

set -e

time make

sqlite_file_path=cyclic_fac_n_data_modulo_prime.sqlite3
thread_count_available=8

factor_amount=1
# factor_len=4
# factor_len=5
# factor_len=6
# factor_len=7

tries_per_thread=500000
# factor_len=81

# randomize last_a_factor
rand_a=$(od -N 8 -t uL -An /dev/urandom | tr -d " ")
rand_a=$(($rand_a - ($rand_a % 4) + 1))
last_a_factor=$(printf "%016X" $rand_a)

# fixed last_a_factor
# rand_a=1234252348
# rand_a=$(($rand_a - ($rand_a % 4) + 1))
# last_a_factor=$(printf "%016X" $rand_a)

# fixed last_c_factor
# rand_c=541233
# rand_c=$(($rand_c - ($rand_c % 2) + 1))
# last_c_factor=$(printf "%016X" $rand_c)

# time ./cyclic_multi_factor_pow_seq tries_per_thread=10000 factor_amount=$factor_amount modulo=2 factor_len=$factor_len amount_nonzero_factors=5 values_a=123456789abcdef1,0203010203040501,0000000000000001 file_path=/tmp/test_cpp_out_2.txt thread_count_available=10 values_c=1212121212121215,1543623485734987,$last_c_factor

# for modulo in $(seq 3 10); do
# for modulo in $(seq 3 30); do
# arr_factor_len=( 8 9 8 9 8 9 8 9 8 9 )
# arr_factor_len=( 6 7 6 7 6 7 6 7 6 7 6 7 6 7 )
# arr_factor_len=( 2 3 4 5 6 7 8 9 2 3 4 5 6 7 8 9 2 3 4 5 6 7 8 9 )
# arr_factor_len=( 2 3 4 5 6 7 8 9 2 3 4 5 6 7 8 9 2 3 4 5 6 7 8 9 )
# arr_factor_len=( 2 3 4 5 6 7 8 9 2 3 4 5 6 7 8 9 2 3 4 5 6 7 8 9 2 3 4 5 6 7 8 9 2 3 4 5 6 7 8 9 )
# arr_factor_len=( 2 3 4 5 6 7 8 9 )
arr_factor_len=( 2 )
arr_modulo=( 3 5 7 11 13 )
for factor_len in "${arr_factor_len[@]}"; do
	# randomize last_a_factor
	rand_c=$(od -N 8 -t uL -An /dev/urandom | tr -d " ")
	rand_c=$(($rand_c - ($rand_c % 2) + 1))
	last_c_factor=$(printf "%016X" $rand_c)
	
	# rand_c=$(($rand_c + 2))
	# last_c_factor=$(printf "%016X" $rand_c)
	
	# for modulo in $(seq 17 19); do
	# for modulo in $(seq 19 19); do
	# for modulo in $(seq 3 80); do
	for modulo in "${arr_modulo[@]}"; do

	# for modulo in $(seq 3 40); do
	# for modulo in $(seq 41 80); do
	# for modulo in $(seq 3 10); do
	# for modulo in $(seq 11 20); do
	# for modulo in $(seq 21 30); do
	# for modulo in $(seq 22 30); do
	# for modulo in $(seq 8 9); do
		rand_c=$(($rand_c + 2))
		last_c_factor=$(printf "%016X" $rand_c)
		time ./cyclic_multi_factor_pow_seq tries_per_thread=$tries_per_thread factor_amount=$factor_amount modulo=$modulo factor_len=$factor_len amount_nonzero_factors=5 values_a=123456789abcdef1,0203010203040501,$last_a_factor file_path=/tmp/test_cpp_out_2.txt thread_count_available=$thread_count_available values_c=1212121212121215,1543623485734987,$last_c_factor sqlite_file_path=$sqlite_file_path
	done
done
