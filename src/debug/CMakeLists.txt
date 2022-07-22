add_executable(testd test.cpp)

target_include_directories(testd PRIVATE
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/OpenMesh/src>
)

target_link_libraries(testd PRIVATE trimem)

set_target_properties(testd PROPERTIES INSTALL_RPATH
                                      "$ORIGIN/${CMAKE_INSTALL_LIBDIR}")

install(TARGETS testd DESTINATION .)
