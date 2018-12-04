CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT(DistributedRF CXX)
set(CMAKE_CXXFLAGS "-Wall -Werror -Wextra -pedantic -std=c++17")

find_package(MPI REQUIRED)

set(
    sources

    src/distributed_rf.cc
    src/tree.cc
    src/main.cc
)

add_executable(distributed_rf ${sources})
target_include_directories(distributed_rf PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_compile_options(distributed_rf PRIVATE ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(distributed_rf ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
