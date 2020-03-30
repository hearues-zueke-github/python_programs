.DEFAULT_GOAL := default
.PHONY: default
# CC=clang++
CC=g++
# CFLAGS=-Wall -std=c++17
CFLAGS=-O3 -Wall -std=c++17

utils_primes:
	$(CC) $(CFLAGS) utils_primes.cpp -c
concat_number_to_prime: utils_primes
	$(CC) $(CFLAGS) -pthread concat_number_to_prime.cpp utils_primes.o -o concat_number_to_prime.o
generate_primes: utils_primes
	$(CC) $(CFLAGS) -pthread generate_primes.cpp utils_primes.o -o generate_primes.o
default: utils_primes
	$(CC) $(CFLAGS) -pthread generate_primes.cpp utils_primes.o -o generate_primes.o
