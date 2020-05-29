.DEFAULT_GOAL := default
.PHONY: default
CC=g++
CFLAGS=-O3 -Wall -std=c++17
BUILD=build/
# HEADERS=-I/usr/include/python3.7m
# LINKER=-lpython3.7m

before:
	mkdir -p $(BUILD)
utils: before
	$(CC) $(CFLAGS) ../cpp_programs/primes/utils.cpp -c -o $(BUILD)utils.o
utils_tetris: before
	$(CC) $(CFLAGS) utils_tetris.cpp -c -o $(BUILD)utils_tetris.o
pystring:
	(cd pystring && make)
default: before utils utils_tetris pystring
	$(CC) $(CFLAGS) tetris_deep_search_path_optimization.cpp $(BUILD)utils.o $(BUILD)utils_tetris.o pystring/pystring.o -o $(BUILD)tetris_deep_search_path_optimization.o
# 	$(CC) $(CFLAGS) $(HEADERS) tetris_deep_search_path_optimization.cpp $(BUILD)utils.o $(BUILD)utils_tetris.o $(LINKER) -o $(BUILD)tetris_deep_search_path_optimization.o
