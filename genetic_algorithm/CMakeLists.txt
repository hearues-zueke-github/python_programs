cmake_minimum_required(VERSION 3.16)
project(genetic_algorithm)

set(CMAKE_CXX_STANDARD 17)

add_library(utils ../cpp_programs/primes/utils.cpp)
add_library(utils_tetris utils_tetris.cpp)

#add_executable(tetris_deep_search_path_optimization build/tetris_deep_search_path_optimization.o)
add_executable(genetic_algorithm tetris_deep_search_path_optimization.cpp)
target_link_libraries(genetic_algorithm utils)
target_link_libraries(genetic_algorithm utils_tetris)
#target_link_libraries(tetris_deep_search_path_optimization utils)
