cmake_minimum_required(VERSION 3.18)
project(test)

find_package(Python COMPONENTS Interpreter Development)
find_package(OpenMP)

# use EXCLUDE_FROM_ALL to avoid the install targets defined upstream OpenMesh,
# instead add dependency of test on opemesh-python explicitly below
add_subdirectory(openmesh-python EXCLUDE_FROM_ALL)

pybind11_add_module(test src/energy.cpp)
target_include_directories(test PRIVATE openmesh-python/src)
target_include_directories(test PRIVATE openmesh-python/OpenMesh/src)

target_link_libraries(test PRIVATE OpenMeshToolsStatic
                                   OpenMeshCoreStatic
                                   OpenMP::OpenMP_CXX)

add_dependencies(test openmesh)

install(TARGETS test openmesh LIBRARY DESTINATION .)
