.DEFAULT_GOAL := default
.PHONY: default
# CC=clang++
CC=g++
# CFLAGS=-Wall -std=c++17
CFLAGS=-O3 -Wall -std=c++17

utils:
	$(CC) $(CFLAGS) ../cpp_programs/primes/utils.cpp -c
default: utils
	$(CC) $(CFLAGS) tetris_deep_search_path_optimization.cpp utils.o -o tetris_deep_search_path_optimization.o
