.DEFAULT_GOAL := all

CC=g++
OPTIMIZATION=-O2
# CFLAGS=-Werror -Wall -std=c++20 -pthread -O3
CFLAGS=-Werror -Wall -std=c++20 $(OPTIMIZATION) -pthread -fmodules-ts
# CFLAGS=-Wall -std=c++20

iostream:
	$(CC) $(CFLAGS) -x c++-system-header iostream -c
utils.o: utils.cpp utils.h
	$(CC) $(CFLAGS) utils.cpp -c
thread_data_generic.o: thread_data_generic.cpp thread_data_generic.h
	$(CC) $(CFLAGS) thread_data_generic.cpp -c
multi_linear_sequences.o: multi_linear_sequences.cpp multi_linear_sequences.h
	$(CC) $(CFLAGS) multi_linear_sequences.cpp -c
main.o: utils.o thread_data_generic.o multi_linear_sequences.o main.cpp
	$(CC) $(CFLAGS) utils.o thread_data_generic.o multi_linear_sequences.o main.cpp -o main.o
all: main.o

clean:
	rm utils.o
	rm thread_data_generic.o
	rm main.o
