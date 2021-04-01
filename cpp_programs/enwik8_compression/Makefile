.PHONY: default
CC=g++
CFLAGS=-Wall -std=c++17

utils:
	$(CC) $(CFLAGS) utils.cpp -c
default: utils
	$(CC) $(CFLAGS) main.cpp utils.o -o main.o
