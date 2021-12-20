.DEFAULT_GOAL := all

CC=g++
CFLAGS=-Werror -Wall -std=c++20 -pthread -O3
# CFLAGS=-Werror -Wall -std=c++20 -O2 -pthread -fmodules-ts
# CFLAGS=-Wall -std=c++20

iostream:
	$(CC) $(CFLAGS) -x c++-system-header iostream -c
utils: utils.cpp
	$(CC) $(CFLAGS) utils.cpp -c
# thread_data_generic: utils utils.o thread_data_generic.cppm
# 	$(CC) $(CFLAGS) thread_data_generic.cppm -c
thread_data_generic: utils utils.o thread_data_generic.cpp
	$(CC) $(CFLAGS) thread_data_generic.cpp -c
multi_linear_sequences: utils utils.o thread_data_generic thread_data_generic.o multi_linear_sequences.cpp
	$(CC) $(CFLAGS) multi_linear_sequences.cpp -c
main: main.cpp
main: utils utils.o thread_data_generic thread_data_generic.o multi_linear_sequences multi_linear_sequences.o main.cpp
	$(CC) $(CFLAGS) utils.o thread_data_generic.o multi_linear_sequences.o main.cpp -o main.o
#	$(CC) $(CFLAGS) utils.cpp multi_linear_sequences.cpp main.cpp -o main.o
# 	$(CC) $(CFLAGS) utils.cpp thread_data_generic.cpp multi_linear_sequences.cpp main.cpp -o main.o
# 	$(CC) $(CFLAGS) utils.o thread_data_generic.o multi_linear_sequences.o main.cpp -o main.o
all: main

clean:
	rm utils.o
	rm thread_data_generic.o
	rm main.o
