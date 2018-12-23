CMAKE_MINIMUM_REQUIRED(VERSION 3.9)

PROJECT(DistributedRF CXX)
set(CMAKE_CXX_FLAGS "-Wall -Werror -Wextra -pedantic -std=c++17 -g")
set(CMAKE_CXX_COMPILER "mpicxx")
set(PROJECT_NAME diraf)

find_package(MPI REQUIRED)

set(
    sources

    src/distributed_rf.cc
    src/error_functions.cc
    src/tree.cc
    src/utils.cc

)

add_library(_${PROJECT_NAME} SHARED ${sources})
add_executable(distributed_rf src/main.cc)
add_executable(test_diraf src/test.cc)

target_include_directories(distributed_rf PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_compile_options(distributed_rf PRIVATE ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(distributed_rf ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS} _${PROJECT_NAME})

target_include_directories(test_diraf PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_compile_options(test_diraf PRIVATE ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(test_diraf _${PROJECT_NAME} ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
