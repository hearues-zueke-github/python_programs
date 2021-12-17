.DEFAULT_GOAL := all

CC=g++
# CFLAGS=-Werror -Wall -std=c++20 -g -pthread
CFLAGS=-Werror -Wall -std=c++20 -O2 -pthread -fmodules-ts
# CFLAGS=-Wall -std=c++20

iostream:
	$(CC) $(CFLAGS) -x c++-system-header iostream
main: main.cpp
	$(CC) $(CFLAGS) main.cpp -o main.o
all: main
clean:
	rm main.o
