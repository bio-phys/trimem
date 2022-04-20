cmake_minimum_required(VERSION 3.18)
project(test)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP)
find_package(pybind11 REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(OpenMesh EXCLUDE_FROM_ALL)

pybind11_add_module(core MODULE src/bindings.cpp
                                src/mesh.cpp
                                src/params.cpp
                                src/mesh_properties.cpp
                                src/mesh_util.cpp
                                src/energy.cpp
                                src/flips.cpp
                                src/flip_utils.cpp
                                src/omp_guard.cpp
                                src/nlists/nlist.cpp)
target_include_directories(core PRIVATE src OpenMesh/src)

target_link_libraries(core PUBLIC OpenMeshToolsStatic
                                  OpenMeshCoreStatic
                                  OpenMP::OpenMP_CXX)


install(TARGETS core DESTINATION .)
