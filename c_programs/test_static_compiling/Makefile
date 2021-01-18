CC=gcc
CFLAGS=-Wall

simple_library:
	$(CC) $(CFLAGS) -c simple_library.c

main: simple_library
	$(CC) $(CFLAGS) main.c simple_library.o -o main.o

simple_library_O2:
	$(CC) $(CFLAGS) -O2 -c simple_library.c -o simple_library_O2.o

main_O2: simple_library_O2
	$(CC) $(CFLAGS) -O2 main.c simple_library_O2.o -o main_O2.o
