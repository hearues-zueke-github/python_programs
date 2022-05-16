cmake_minimum_required(VERSION 3.21)
project(OwnPRNG)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-no-pie -O2")
# set(CMAKE_CXX_FLAGS "-no-pie")

set(INC_FMT $ENV{HOME}/libs/fmt-8.1.1)

add_executable(PRNGinCPP prng_in_cpp.cpp)
target_include_directories(PRNGinCPP PUBLIC ${INC_FMT}/include)
target_link_libraries(PRNGinCPP ${INC_FMT}/build/libfmt.a)
