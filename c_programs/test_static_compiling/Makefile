CC=gcc
CFLAGS=-Wall # -fPIC # -static
OBJ_DIR=objs/

.DEFAULT_GOAL=all

create_obj_dir:
	mkdir -p $(OBJ_DIR)

main_empty: create_obj_dir main_empty.c
	$(CC) $(CFLAGS) main_empty.c -o $(OBJ_DIR)main_empty.o

main_include_stdio: create_obj_dir main_include_stdio.c
	$(CC) $(CFLAGS) main_include_stdio.c -o $(OBJ_DIR)main_include_stdio.o


simple_library: create_obj_dir simple_library.c
	$(CC) $(CFLAGS) -c simple_library.c -o $(OBJ_DIR)simple_library.o

main_with_simple_library: create_obj_dir main_with_simple_library.c simple_library
	$(CC) $(CFLAGS) main_with_simple_library.c $(OBJ_DIR)simple_library.o -o $(OBJ_DIR)main_with_simple_library.o

main_without_simple_library: create_obj_dir main_without_simple_library.c
	$(CC) $(CFLAGS) main_without_simple_library.c -o $(OBJ_DIR)main_without_simple_library.o


simple_library_O2: create_obj_dir simple_library.c
	$(CC) $(CFLAGS) -O2 -c simple_library.c -o $(OBJ_DIR)simple_library_O2.o

main_with_simple_library_O2: create_obj_dir main_with_simple_library.c simple_library_O2
	$(CC) $(CFLAGS) -O2 main_with_simple_library.c $(OBJ_DIR)simple_library_O2.o -o $(OBJ_DIR)main_with_simple_library_O2.o

main_without_simple_library_O2: create_obj_dir main_without_simple_library.c
	$(CC) $(CFLAGS) -O2 main_without_simple_library.c -o $(OBJ_DIR)main_without_simple_library_O2.o


all_empty_main:
	make main_empty
	make main_include_stdio

all_simple_normal:
	make simple_library
	make main_with_simple_library
	make main_without_simple_library

all_optimized:
	make simple_library_O2
	make main_with_simple_library_O2
	make main_without_simple_library_O2


all:
	make all_simple_normal
	make all_empty_main
	make all_optimized
