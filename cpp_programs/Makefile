.PHONY: utils
CC=g++
CFLAGS=-Wall -std=c++17

utils:
	$(CC) $(CFLAGS) utils.cpp -c
modulo_sequence:
	$(CC) $(CFLAGS) modulo_sequence.cpp utils.cpp -o modulo_sequence.o
test_vector:
	$(CC) $(CFLAGS) test_vector.cpp utils.cpp -o test_vector.o
test_vector_2:
	$(CC) $(CFLAGS) test_vector_2.cpp -o test_vector_2.o
test_vector_3: utils
	$(CC) $(CFLAGS) test_vector_3.cpp utils.o -o test_vector_3.o
test_vector_sort: utils
	$(CC) $(CFLAGS) test_vector_sort.cpp utils.o -o test_vector_sort.o
hex_output: utils
	$(CC) $(CFLAGS) hex_output.cpp utils.o -o hex_output.o
all:
	$(CC) $(CFLAGS) simple_hashing_algorithm.cpp -o simple_hashing_algorithm.o
