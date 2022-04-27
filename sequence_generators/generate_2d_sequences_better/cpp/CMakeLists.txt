cmake_minimum_required(VERSION 3.0)

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_FLAGS_RELEASE "-O3")

project (CppPrograms CXX)

set (INC_FMT $ENV{HOME}/libs/fmt-8.1.1)

add_executable (Generate2dSequence ./generate_2d_sequence_images.cpp)

target_include_directories (Generate2dSequence PUBLIC ./CImg)
target_include_directories(Generate2dSequence PUBLIC ${INC_FMT}/include)

target_link_libraries (Generate2dSequence PUBLIC -lpng)
target_link_libraries (Generate2dSequence PUBLIC -lX11)
target_link_libraries (Generate2dSequence PUBLIC ${INC_FMT}/build/libfmt.a)
