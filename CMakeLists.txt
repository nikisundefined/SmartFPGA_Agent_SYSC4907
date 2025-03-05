cmake_minimum_required(VERSION 3.15)
project(nengocpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)

# TODO: Include build source for python 3.10 and pybind11

find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

add_library(nengocpp src/nengo.cpp src/nengo.h)
target_include_directories(nengocpp BEFORE PUBLIC ${EIGEN3_INCLUDE_DIRS})
pybind11_add_module(nengopy src/nengo.cpp THIN_LTO)
target_include_directories(nengopy PRIVATE ${EIGEN3_INCLUDE_DIRS})

add_custom_command(TARGET nengopy
        POST_BUILD
        COMMAND bash -c "PYTHONPATH=\"${nengocpp_BINARY_DIR}\" ${_Python_EXECUTABLE} -m pybind11_stubgen --ignore-all-errors --numpy-array-remove-parameters --output-dir ${CMAKE_SOURCE_DIR}/nengopy nengocpp"
        WORKING_DIRECTORY ${nengocpp_BINARY_DIR}
)