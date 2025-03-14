cmake_minimum_required(VERSION 3.15)
project(smart_agent LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)

include(FetchContent)
FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG master
        FIND_PACKAGE_ARGS CONFIG
)
FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG master
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
        FIND_PACKAGE_ARGS CONFIG
)
FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json
        GIT_TAG master
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
        FIND_PACKAGE_ARGS
)

# TODO: find any other options that should be set when using this method of including projects
set(BUILD_TESTING OFF)
set(EIGEN_BUILD_TESTING OFF)
set(EIGEN_MPL2_ONLY ON)
set(EIGEN_BUILD_PKGCONFIG OFF)
set(EIGEN_BUILD_DOC OFF)

# TODO: Convert the find_package calls to FetchContent
find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
FetchContent_MakeAvailable(pybind11 nlohmann_json)

add_library(rectified_linear STATIC src/hls/rectified_linear.cpp)
target_include_directories(rectified_linear PRIVATE /opt/Xilinx/Vitis_HLS/2023.2/include)

# Testing Executables
add_executable(pathfinding src/pathfinding.cpp src/pathfinding.h)
target_compile_options(pathfinding PRIVATE -fconstexpr-ops-limit=4294967296)

# Common Headers
add_library(neurons INTERFACE src/nengo.h)
add_library(sim INTERFACE src/point.h src/player.h src/path_cache.h src/path_pair.h src/direction.h src/arena.h src/pathfinding.h)
add_library(share INTERFACE src/shared_point.h src/shared_player.h src/shared_arena.h src/shared_path_cache.h)

# Helper function to create pybind11 module and auto link all libraries
function(pybind11_module name)
    # Create pybind11 python module
    pybind11_add_module(${ARGV} THIN_LTO)
    # Attach Eigen library and internal headers
    target_link_libraries(${name} PRIVATE Eigen3::Eigen sim share neurons)
    # Attach json library if found
    if(${nlohmann_json_FOUND})
        target_include_directories(${name} PRIVATE ${nlohmann_json_INCLUDE_DIRS})
        target_compile_definitions(${name} PRIVATE SMART_AGENT_JSON)
    endif()
    # Configure stub generation after build
    add_custom_command(TARGET ${name}
            POST_BUILD
            COMMAND bash -c "PYTHONPATH=\"${${name}_BINARY_DIR}\" ${Python3_EXECUTABLE} -m pybind11_stubgen --ignore-all-errors --numpy-array-remove-parameters --output-dir ${CMAKE_SOURCE_DIR}/smart_agent ${name}"
            WORKING_DIRECTORY ${${name}_BINARY_DIR}
    )
endfunction()

# Python Modules
pybind11_module(nengocpp src/bindings/nengo.cpp)
pybind11_module(simulationcpp src/bindings/simulation.cpp)
pybind11_module(sharedcpp src/bindings/shared.cpp)
