cmake_minimum_required(VERSION 3.18)
project(test)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenMP)

# use EXCLUDE_FROM_ALL to avoid the install targets defined upstream OpenMesh,
# instead add dependency of test on opemesh-python explicitly below
add_subdirectory(openmesh-python EXCLUDE_FROM_ALL)

pybind11_add_module(_core MODULE src/bindings.cpp
                          src/params.cpp
                          src/mesh_properties.cpp
                          src/mesh_util.cpp
                          src/energy.cpp
                          src/flips.cpp)
target_include_directories(_core PRIVATE openmesh-python/src)
target_include_directories(_core PRIVATE openmesh-python/OpenMesh/src)

target_link_libraries(_core PUBLIC OpenMeshToolsStatic
                                   OpenMeshCoreStatic
                                   OpenMP::OpenMP_CXX)

add_dependencies(_core openmesh)

install(TARGETS _core openmesh DESTINATION .)
