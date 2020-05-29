.PHONY: default
CC=g++
CFLAGS=-Wall -std=c++17

Number:
	$(CC) $(CFLAGS) Number.cpp -c -o Number.o
Matrix:
	$(CC) $(CFLAGS) Matrix.cpp -c -o Matrix.o
default: Number Matrix
	$(CC) $(CFLAGS) main.cpp Number.o Matrix.o -o main.o
