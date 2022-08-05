cmake_minimum_required(VERSION 3.18)
project(test)

set(CMAKE_CXX_STANDARD 17)

# dependencies
find_package(OpenMP)
find_package(pybind11 REQUIRED)

# OpenMesh
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(OpenMesh EXCLUDE_FROM_ALL)

# core library
add_library(trimem SHARED src/mesh.cpp
                             src/mesh_properties.cpp
                             src/energy.cpp
                             src/flips.cpp
                             src/flip_utils.cpp
                             src/omp_guard.cpp
                             src/nlists/nlist.cpp
                             src/util.cpp)
target_include_directories(trimem PUBLIC src OpenMesh/src)
target_link_libraries(trimem PUBLIC OpenMeshCore
                                    OpenMP::OpenMP_CXX)

# python module
pybind11_add_module(core MODULE src/bindings.cpp
                                src/mesh_py.cpp)
target_link_libraries(core PRIVATE trimem)

install(TARGETS core trimem OpenMeshCore DESTINATION .)
set_target_properties(core trimem PROPERTIES INSTALL_RPATH
                                      "$ORIGIN/${CMAKE_INSTALL_LIBDIR}")

# c++ debug app
add_subdirectory(src/debug)
